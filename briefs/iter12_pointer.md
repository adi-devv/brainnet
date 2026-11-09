# Iteration 12: relational pointer layer with a local two-phase rule

Work in `D:\Aadit\claudez\brainnet` only. Read `iter10.py`, `iter8.py`, `iter5.py`, `hybrid_mixer.py`, then create `iter12.py` (at most 130 lines). Create nothing else except `results/iter12_*`. Do not change any existing file. No network, no pip, no git. Run Python only as `"C:\Users\aadit\AppData\Local\Programs\Python\Python312\python.exe" -X utf8 -B ...`. Never stop any process. Smoke check on tiny budgets only (under 60 s total); do not run the full commands. Rename or delete nothing that existed before you started; a smoke run must write to `results/iter12_smoke_<variant>.json`, never to the real result names. Reply with at most 8 lines.

No `.backward()`, `torch.autograd`, `torch.optim`; `torch.set_grad_enabled(False)`.

## Why
The renamed test replaces Python identifiers with random letters. Char-keyed tables learn nothing about those. This layer is an attention head built from relations only: it sees whether characters or tokens MATCH, never which ones they are, so its weights are name-invariant by construction.

## The head
At position t the model predicts the char after x[t]. Candidates are j in 0..t-1; candidate j proposes char x[j+1]. Each pair (t, j) maps to one integer row of a weight table. A head has `W` (float32, n_rows, init -5.0), a null weight `w0` (init 0.0), Adagrad state `G` (init 1e-8).

    a_j    = softmax over {null, all valid j}, score W[row(t,j)] for j, w0 for null
    q(c)   = a_null * f(c) + sum_j a_j * [x[j+1] == c]      f = fallback distribution
    r_j    = a_j * [x[j+1] == y] / q(y)                     clamped phase
    r_null = a_null * f(y) / q(y)
    delta[row] = sum over examples and j of (r_j - a_j)     per row; same form for w0
    G += delta**2;  W += 0.2 * delta / sqrt(G)              only rows with delta != 0; only when y is given

Updates are per position t, so per step like the mixer. Normalise q to sum 1 and floor f at 1e-6.

Two heads, each with its own W, w0, G. d = t-j; `dist = bucketize(d, [2,4,8,16,32,64])`, 7 buckets. Out-of-range lags compare as not equal.

- **Head 1, char match.** 8 bits: bit l is `x[t-l] == x[j-l]`, l=0..7. Row = bits*7 + dist. 1792 rows.
- **Head 2, token match.** A word char is a letter, digit or underscore (vocab from `data/meta.json`; pad V is non-word). A token is a maximal run of word chars; every other char is its own token. Compute with a vectorised loop over t, causal: `prefix[t]` = rolling hash of the current token up to and including t; `prev1[t]`, `prev2[t]` = hashes of the last and second-last complete tokens; `length[t]` = min(chars in current token, 7). Bits: prefix equal, prev1 equal, prev2 equal, length equal, x[j+1] is word, x[t] is word. Row = bits*7 + dist. 448 rows.

Pre-compute rows per batch as int16 (b,t,t) with a sentinel for j >= t, plus the candidate chars; invalid candidates get a_j = 0. Evaluation flattens in chunks of 1024, so override `logprobs` accordingly.

## Variants: `iter12.py {learned,frozen,pure} [--budget N]`
- `learned`: subclass `iter10.Model` with variant `clock` unchanged, then add the two heads with f = the highest-order slow prediction (`ps[4]`). 18 inputs: iter 10's 16, then head 1, head 2. First layer `Neuron(r, 18, lr)` for r in (1, V+1, 4096, 16, 12, 16), second `Neuron(V+1, 6, lr)`. New sixth selector, 16 rows: `bucketize(1-a_null_head1,[.1,.5,.9])*4 + bucketize(1-a_null_head2,[.1,.5,.9])`.
- `frozen`: same as `learned` but W, w0, G never update. Control for whether the local rule learns anything.
- `pure`: no slow tables, no fast tables, no clocked tables. f = histogram of x[0..t] in the window, +0.5 smoothing, excluding pad. 3 inputs: f, head 1, head 2. First layer `Neuron(r, 3, lr)` for r in (1, V+1, 16) with the same 16-row selector, second `Neuron(V+1, 3, lr)`. Answers: how far does the relational learner go with no char-keyed memory beyond the mixer's own weights.

Run one variant at a time. Same setup as iter 10: `iter8.fit`-style loop, `tr[:cut]`, one pass, bs=976, seed 0, cuda, allocator cap 2.7 GB, lr 1e-4 for the mixer. For `pure` make sure no slow table is built or written.

## Required checks (assert all, in every run)
1. Brute-force reference: on a tiny example, explicit Python loops reproduce head 1 and head 2 rows, a, q and the delta of one update.
2. Invariance: apply a random letter-to-letter bijection to x; head 1 and head 2 rows must be identical.
3. Causality: changing x at positions >= 100 leaves logprobs at t < 100 unchanged, plus `iter10.reference()` for `learned`.
4. Evaluation leaves all tables, mixer, W, w0, G unchanged (`I.stamp`).

## Output
`results/iter12_<variant>.json`: test_loss, renamed_loss, val_loss, renamed_gap, train_seconds, chars, peak_allocated_mib, head1_null_mean, head2_null_mean, settings. `pure` may omit what does not apply.

## Decision rules (save in settings, do not act on them)
- Reference: iter 10 `clock` test 1.1711, renamed 1.6269, gap 0.4559. Backprop at 16M chars: test 1.1238, renamed 1.5298, gap 0.406.
- `learned` beats `frozen` by >= 0.01 on renamed loss, else the local rule is not learning.
- `learned` renamed loss <= 1.58 and gap < 0.43: name dependence improved, continue this learner.
- `learned` test <= 1.16: continue.
- `pure`: report only.
