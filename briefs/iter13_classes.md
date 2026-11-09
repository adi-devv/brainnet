# Iteration 13: character-class head in the relational pointer layer

Work in `D:\Aadit\claudez\brainnet` only. Read `iter12.py`, `iter10.py`, `iter5.py`, `hybrid_mixer.py`, then create `iter13.py` (at most 100 lines). Create nothing else except `results/iter13_*`. Do not change any existing file. No network, no pip, no git. Run Python only as `"C:\Users\aadit\AppData\Local\Programs\Python\Python312\python.exe" -X utf8 -B ...`. Never stop any process. Smoke check on tiny budgets only (under 60 s total); do not run the full commands. A smoke run must write to `results/iter13_smoke_<variant>.json`, never to the real result names. Reply with at most 8 lines.

No `.backward()`, `torch.autograd`, `torch.optim`; `torch.set_grad_enabled(False)`.

## Why
Iter 12's heads see only whether chars match. They cannot tell that a candidate proposes a `(` versus a letter, or that x[t] is a space. A third head adds what KIND of character is involved, while staying blind to which letter it is.

## The third head
Same `iter12.Head` class and rule (W init -5.0, w0 init 0.0, Adagrad 0.2). Heads 1 and 2 stay exactly as in iter 12. Head 3's row for pair (t, j), j < t, built from a key function `k` over the vocab plus pad:

    row = (k[x[t]]*K + k[x[j+1]])*8 + eq0 + 2*eq1 + 4*prefix)*7 + dist

`eq0` = x[t]==x[j]; `eq1` = t>=1 and j>=1 and x[t-1]==x[j-1]; `prefix` = token-prefix hash equal (`h` in `iter12.rows`); `dist` is iter 12's 7 buckets; the candidate char and pad handling are iter 12's (`nxt`, last position proposes pad). Store rows as int32 with -1 for j >= t.

## Variants: `iter13.py {classes,literal} [--budget N] [--smoke]`
- `classes`: K = 6. k: letter (any case) 0, digit 1, underscore 2, space 3, newline 4, everything else including pad 5.
- `literal`: letter (any case) 0, digit 1, then every other vocab char and pad gets its own key in vocab order, so K = 2 + count of those. Punctuation, space and newline keep their identity because the renamed test never changes them.

Both are otherwise iter 12 `learned`: `iter10.Model` clock, f = `ps[4]` for all three heads, 19 inputs (iter 12's 18 then head 3), first layer `Neuron(r, 19, lr)` for r in (1, V+1, 4096, 16, 12, 16), second `Neuron(V+1, 6, lr)`. The sixth selector is unchanged, from heads 1 and 2 only. Subclass or reuse `iter12` rather than copying it. Same setup: one pass, `tr[:cut]`, bs=976, seed 0, cuda, allocator cap 2.7 GB, mixer lr 1e-4, evaluation chunks of 1024. Run one variant at a time.

## Required checks (assert all, in every run)
1. `iter12.reference(vocab)` and `iter10.reference()` still pass.
2. Brute force: on a tiny example, explicit Python loops reproduce head 3 rows for both key functions.
3. Invariance: a random bijection on letters (any letter to any letter, across case) leaves head 3 rows identical.
4. Causality: `R.R.causality` as in iter 12.
5. Evaluation leaves all tables, mixer, W, w0, G unchanged (`I.stamp`).

## Output
`results/iter13_<variant>.json`: test_loss, renamed_loss, val_loss, renamed_gap, train_seconds, chars, peak_allocated_mib, head1_null_mean, head2_null_mean, head3_null_mean, head3_rows, settings.

## Decision rules (save in settings, do not act on them)
- Reference: iter 12 `learned` test 1.1651, renamed 1.6058, gap 0.4407. Backprop at 16M chars: test 1.1238, renamed 1.5298, gap 0.406.
- Beats iter 12 `learned` by >= 0.01 on renamed loss: the class head adds something.
- Renamed <= 1.58 and gap < 0.43: name dependence improved, continue this learner.
- Test <= 1.16: continue.
- If none of the last two hold: stop stacking on count tables.
