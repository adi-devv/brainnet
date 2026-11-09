# Iteration 9: recency-weighted fast memory and a copy predictor

Work in `D:\Aadit\claudez\brainnet` only. Read `iter8.py`, `iter5.py`, `hybrid_mixer.py`, `benchmark.py`, then create `iter9.py` (at most 130 lines). Create nothing else except `results/iter9_*`. Do not change any existing file. No network, no pip, no git. Run Python only as `"C:\Users\aadit\AppData\Local\Programs\Python\Python312\python.exe" -X utf8 -B ...`. Never stop any process. Smoke check on tiny budgets only (under 60 s total); do not run the full commands. Reply with at most 10 lines: what you wrote and the smoke result.

## Why
Iter 8's window-local counts gave the largest gain since iter 5 (test 1.1957 vs control 1.2541). Two weaknesses remain. All earlier matches count equally, although the most recent use of a name is usually the best guide. And hashed fixed orders cannot say "I have been copying from position j for 20 chars". No `.backward()`, `torch.autograd`, `torch.optim`; `torch.set_grad_enabled(False)`.

## Reuse
Subclass `iter8.Model` and reuse `iter8.fit` shape, `iter5.batches`, `iter5.evaluate`, `iter5.rename`, `iter5.stamp`, `iter5.B.load`, and the same setup: `tr[:cut]`, `cut=int(len(tr)*.95)`, one pass, bs=976, lr 1e-4, seed 0, cuda, allocator cap 2.7 GB. Keep the iter 8 causality test, extended to all new inputs. Evaluation keeps slow tables, mixer and copy tables frozen.

## Mixer inputs (16, always the same structure)
5 slow + 5 fast (exactly iter 8) + 5 recency + 1 copy.

## Recency inputs
Same `match` matrix as iter 8 (`j < t`, same window, same hash) for each order. Let rank(t, j) = number of j' with j < j' < t and match(t, j'). So the most recent match has rank 0. Weight of match j = `0.5 ** rank`. Recency counts at (b, t, c) = sum of those weights over j with `x[b, j+1] == c`. Use float32, sub-batches of 32.

Distributions: same backoff chain as iter 8, restarting from the highest-order slow distribution `ps[-1]`: for each order ascending, total = weighted sum, q = (c + 0.5*p) / (total + 0.5), p = q where total > 0, else unchanged. Append each p.

## Copy predictor
For each window, longest-match DP in `t` order: `L_t[j] = (x[t] == x[j]) * (L_{t-1}[j-1] + 1)` for `j < t`, 0 otherwise. At position t pick the j with the largest `L_t[j]`, most recent j on ties; length = that `L_t[j]`, none if 0. The predicted char is `x[j+1]`, and since `j+1 <= t` it is always known and causal.

Length bucket (1..8): 1, 2, 3, 4-5, 6-8, 9-16, 17-32, 33+.
Agree flag: 1 if the predicted char equals `ps[-1].argmax(-1)`, else 0.
Copy selector (17 rows): 0 if no match, else `(bucket-1)*2 + agree + 1`.

Confidence: two global tensors `hit` and `seen` of 17 rows, on the model, starting at zero. During training, after predicting and before writing, for positions with a match do `seen[row] += 1` and `hit[row] += (predicted == target)`. `conf = (hit + 0.5) / (seen + 1)`.

Copy distribution with `p = ps[-1]`: where no match, the input is p unchanged. Where a match exists with predicted char k: `q[k] = conf`, and for other chars `q[c] = (1 - conf) * p[c] / (1 - p[k])`, guarding against a zero denominator. Renormalise.

Tables are never updated during evaluation. Extend the frozen-state check so a change in `hit` or `seen` raises, in addition to `iter5.stamp`.

## Selectors and layers
Fifth selector is the copy selector. Selectors: `m.selectors(x)`, the iter 8 fourth selector, the copy selector. First layer `Neuron(r, 16, lr)` for r in (1, V+1, 4096, 16, 17). Second layer `Neuron(V+1, 5, lr)`. Weights moved to cuda.

## Variants: `iter9.py <name> [--budget N]`, names `all`, `no_recency`, `no_copy`
- `all`: as above.
- `no_recency` (ablation): the 5 recency inputs each equal `ps[-1]`. Same 16 inputs and 5 neurons.
- `no_copy` (ablation): the copy input equals `ps[-1]`, copy selector is always 0, tables stay zero. Same 16 inputs and 5 neurons.

## Output
Save `results/iter9_<name>.json` with test_loss, renamed_loss, val_loss, renamed_gap, train_seconds, chars, peak_allocated_mib, hit_rate as in iter 8, copy_rate (fraction of test evaluation positions with a match), copy_accuracy (fraction of those where the predicted char was right), settings.

`--budget` limits training chars for smoke checks.

Reference checks to run in smoke: on a small hand-built window, assert the DP length and predicted char, and the recency weights, against a plain Python loop. Raise on failure.

## Decision rules (save in settings, do not act on them)
- Reference: iter 8 `fast` test 1.1957, renamed 1.6485, gap 0.4528. Backprop at 16M chars: test 1.1238, gap 0.406.
- `all` test must be at most 1.165, or further tuning of the fast memory stops.
- `all` renamed gap must be below 0.43, or it is not fixing name dependence.
- Recency stays only if `all` beats `no_recency` by at least 0.01 test. Copy stays only if `all` beats `no_copy` by at least 0.01 test.
