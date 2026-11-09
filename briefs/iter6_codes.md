# Iteration 6: learned predictive-state codes

Work in `D:\Aadit\claudez\brainnet` only. Read `iter5.py`, `hybrid_mixer.py`, `benchmark.py`, then create `iter6.py` (at most 200 lines). Create nothing else except `results/iter6_*`. Do not change any existing file. No network, no pip, no git. Run Python only as `"C:\Users\aadit\AppData\Local\Programs\Python\Python312\python.exe" -X utf8 -B ...`. Smoke check on tiny budgets only (under 60 s total); do not run the full commands. Reply with at most 10 lines: what you wrote and the smoke result.

## Why
Current counts key on exact strings, so `foo(x` and `bar(y` share nothing. Test whether closed-form codes, learned from what follows each context, give keys that generalise. Linear algebra only. No `.backward()`, `torch.autograd`, `torch.optim`; `torch.set_grad_enabled(False)`.

## Reuse
`import iter5`, and reuse `iter5.Counts`, `iter5.Model`, `iter5.batches`, `iter5.fit`-style training, `iter5.evaluate`, `iter5.rename`, `iter5.slot` as far as possible. Do not modify `iter5.py`. Same split as `mix_full`: train on `tr[:cut]`, `cut=int(len(tr)*.95)`, `one_pass`, bs=976, lr 1e-4, seed 0, cuda. Total GPU memory under 2.7 GB (the existing five tables take about 1.3 GB, so the new tables use 2**23 entries each).

## Fitting the code (on the training chars only, never val or test)
The past feature vector phi(t) uses only chars strictly before target position t. It is sparse, in F = 2**16 hashed dims:
- (position p, char) for p = 1..8 exact, then (bucket 9-16, char) and (bucket 17-32, char).
- (p, char bigram) for p = 1..8.
- Pad index V for positions before the start. Compute phi the same way in fitting and in use, from the tokens available.

The future vector psi(t) is sparse, in G = 2**12 hashed dims: (j, char) for j = 0..3 ahead starting at the target char, plus the hashed trigram of the next 3 chars.

Diagonal whitening: count feature occurrences in one pass; every feature value is divided by sqrt(count + 1) on its own side (CCA with diagonal covariance).

Randomized SVD of the cross-count matrix C = sum phi(t) psi(t)^T, streamed with `index_add_`, never stored dense:
1. Omega = G x 80 Gaussian, seeded. Y = C Omega, a stream pass. QR gives Q (F x 80).
2. One power iteration: Z = C^T Q (stream), then Y = C Z (stream), QR again.
3. B = Q^T C (stream, 80 x G), SVD of B. W = Q U, keeping dims 1..64 (drop dim 0, the mean direction). W is F x 64.

Code(t) = sum of W rows over the active features of phi(t), with the whitening applied. A code uses only past chars.

## Keys
Bits = code > per-dim median (medians from a seeded sample of training positions). Nested keys use the first B bits for B in (8, 12, 16, 20, 24); each key goes through a mix hash, then the `slot` scheme. Backoff chain coarse to fine with the same alpha=0.5 recipe as `Counts`. Five tables of 2**23 entries, predictions before writes. The rolling exact-context tables are unchanged.

## Mixer
`mix_two_layer` as in `iter5.Model`, with layer 1 taking 10 inputs: the 5 exact-context distributions and the 5 code-key distributions.

## Commands (`iter6.py <name> [--budget N]`), each saving `results/iter6_<name>.json` with test_loss, renamed_loss, val_loss, fit_seconds, train_seconds, chars, peak_allocated_mib, settings
| Name | What |
|---|---|
| `mix_psc` | Learned W as above |
| `mix_rand` | Control: W replaced by a seeded Gaussian F x 64 matrix, same features, bits and tables. Isolates learning from having more inputs |

`mix_psc` also saves `singular_values` (top 64) and a sanity list: for 8 fixed common contexts of train, the 5 nearest training positions by code, each decoded as the last 24 chars.

`--budget` limits both fitting and training chars, for smoke checks.

## Decision rules (save in settings, do not act on them)
- Reference `mix_full`: test 1.2528, renamed 1.7547, gap 0.502.
- `mix_psc` must beat `mix_full` by at least 0.05 on test, or the approach is dropped.
- `mix_psc` must beat `mix_rand` by at least 0.03, or learning is not what helps.
- A lower renamed gap than 0.502 means less dependence on memorised names.
