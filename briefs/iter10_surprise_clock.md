# Iteration 10: surprise-clocked contexts

Work in `D:\Aadit\claudez\brainnet` only. Read `iter8.py`, `iter9.py` (for how it overrides `logprobs` and reuses `iter8.fit`), `iter5.py`, `hybrid_mixer.py`, `benchmark.py`, then create `iter10.py` (at most 130 lines). Create nothing else except `results/iter10_*`. Do not change any existing file. No network, no pip, no git. Run Python only as `"C:\Users\aadit\AppData\Local\Programs\Python\Python312\python.exe" -X utf8 -B ...`. Never stop any process. Smoke check on tiny budgets only (under 60 s total); do not run the full commands. Reply with at most 10 lines: what you wrote and the smoke result.

## Why
Every table so far keys on exact runs of adjacent chars, so one predictable char that differs breaks the match. Here, a position only advances a slower context when it was surprising. Two contexts that differ only in predictable filler then share statistics. No `.backward()`, `torch.autograd`, `torch.optim`; `torch.set_grad_enabled(False)`.

## Reuse
Subclass `iter8.Model`; recency and copy from iter 9 are not used. Reuse `iter8.fit`, `iter5.batches`, `iter5.evaluate`, `iter5.rename`, `iter5.stamp`, `iter5.slot`, `iter5.B.load`, and the same setup: `tr[:cut]`, `cut=int(len(tr)*.95)`, one pass, bs=976, lr 1e-4, seed 0, cuda, allocator cap 2.7 GB. Keep the iter 8 causality test and also assert the clocked hashes at positions 0..99 are unchanged. Evaluation keeps all tables and the mixer frozen.

## Surprise
In `prepare`, before the `t` loop, with tables as they stand (nothing written yet): for each window position `t >= 1`, `s[t] = -log ps[-1][x[t]]`, where `ps[-1]` is the highest-order slow distribution for the context ending at `x[t-1]`. So `s[t]` uses only chars up to `t`. `s[0] = inf`; this also avoids indexing with the pad value `V` at position 0. Predict in chunks of 1024 to stay under the cap.

Two clock levels: A fires at t when `s[t] > 1.0`, B fires at t when `s[t] > 3.0`.

## Clocked contexts
Per level keep the last 3 fired chars of the window, a shift register starting at the pad value `V`, updated at t if the level fires at t. Local part: `h0 = (x[t-1] + 1) * 1000003 + x[t] + 1`, using pad `V` for `x[-1]`. For n in 1, 2, 3: `h_n = h_{n-1} * 1000003 + fired[n-th most recent] + 1`. This gives 6 hashes per position, (A,1..3) and (B,1..3).

## Clocked tables
6 tables, each `C` and `T` int32 with `1 << 23` entries, stored on the model. Index and tag as in `iter5.Counts`: `ix, tag = slot(h * 1000003 + c + 1)`, then `ix = ix & ((1 << 23) - 1)`. Same tag check on read and same replace-on-collision on write. Write them inside `output` when `y is not None`, after the mixer has predicted, so `iter8.fit` can be reused unchanged.

Distributions: per level, a backoff chain starting from `ps[0]` (order 2 slow distribution): for n ascending, total = c.sum, q = (c + 0.5*p) / (total + 0.5), p = q where total > 0, else unchanged. Append each p.

## Mixer
16 inputs: 5 slow + 5 fast (exactly iter 8) + 6 clocked.
Fifth selector (12 rows): `gap` = chars since the last level-A firing, bucket 0..5 for 0, 1, 2, 3-4, 5-8, 9+. `hit` = 1 if the (A,3) context has total > 0. Row = `gap_bucket * 2 + hit`. First layer `Neuron(r, 16, lr)` for r in (1, V+1, 4096, 16, 12). Second layer `Neuron(V+1, 5, lr)`. Weights moved to cuda.

## Variants: `iter10.py <name> [--budget N]`, names `clock`, `clock_off`, `clock_random`
- `clock`: as above.
- `clock_off` (control): the 6 clocked inputs each equal `ps[-1]`, fifth selector always 0, clocked tables never written. Same 16 inputs and 5 neurons.
- `clock_random` (control): same as `clock`, but level A and B firing are pseudo-random at the rates measured by `clock`, read from `results/iter10_clock.json` (raise if missing, or if its `chars` differs from this run). Fire A when `slot(cod[4][b, t])[0] % 100000 < rate_A * 100000`, and B likewise with a different salt: `slot(cod[4][b, t] + 77)`. Still causal, deterministic, and uses no surprise.

## Output
Save `results/iter10_<name>.json` with test_loss, renamed_loss, val_loss, renamed_gap, train_seconds, chars, peak_allocated_mib, hit_rate as in iter 8, fire_rate_A and fire_rate_B (fraction of test evaluation positions firing), settings.

`--budget` limits training chars for smoke checks; a smoke `clock_random` may use the smoke `clock` file.

Reference check to run in smoke: on a small hand-built window and fixed fake surprises, assert the shift registers and clocked hashes against a plain Python loop. Raise on failure.

## Decision rules (save in settings, do not act on them)
- Reference: iter 8 `fast` test 1.1957, gap 0.4528. Iter 9 `all` test 1.1843, gap 0.437. Backprop at 16M chars: test 1.1238, gap 0.406.
- `clock_off` should land within 0.005 of 1.1957, else the comparison is suspect.
- `clock` must beat `clock_off` by at least 0.03 test, or the approach is dropped.
- `fire_rate_A` of `clock` must be at most 0.70, or it is not sparse.
- `clock` must beat `clock_random` by at least 0.01 test, or surprise is not what helps.
- `clock` renamed gap below 0.43 means less dependence on names.
