# Iteration 8: two-speed memory

Work in `D:\Aadit\claudez\brainnet` only. Read `iter5.py`, `hybrid_mixer.py`, `benchmark.py`, then create `iter8.py` (at most 130 lines). Create nothing else except `results/iter8_*`. Do not change any existing file. No network, no pip, no git. Run Python only as `"C:\Users\aadit\AppData\Local\Programs\Python\Python312\python.exe" -X utf8 -B ...`. Another GPU job may be running: never stop any process. Smoke check on tiny budgets only (under 60 s total); do not run the full commands. Reply with at most 10 lines: what you wrote and the smoke result.

## Why
Every input today comes from global tables of the training data. Nothing looks only at the current 128-char window, so a name used 20 chars ago is unknown unless the training set also had it. Add a fast memory that starts empty for every window and is built only from that window. No `.backward()`, `torch.autograd`, `torch.optim`; `torch.set_grad_enabled(False)`.

## Reuse
Subclass `iter5.Model`. Reuse `iter5.batches`, `iter5.evaluate`, `iter5.rename`, `iter5.stamp`, `iter5.B.load`. Same setup as `mix_full`: train on `tr[:cut]`, `cut=int(len(tr)*.95)`, one pass, bs=976, lr 1e-4, seed 0, cuda, allocator cap 2.7 GB as in `iter5.run`. Evaluation keeps the slow tables and mixer frozen.

## Fast memory
Use the existing rolling hashes `cod = m.base.codes(x)`, for every order in `m.base.lens`, ascending. At window position t the target is `x[t+1]` (`y[t]`), and `cod[order][:, t]` is the context ending at `x[t]`.

For each order, fast counts at (b, t, c) = number of j < t in the same window with `cod[order][b, j] == cod[order][b, t]` and `x[b, j+1] == c`. Only chars inside the window and strictly before the target are used, so no `y` is needed and the same code runs in training and evaluation. Vectorised: `match = (h[:, :, None] == h[:, None, :])`, keep only j < t, one-hot the next chars with V+1 classes and drop the pad class V, matmul in sub-batches so memory stays under the cap. Compute for the whole batch before the t loop. Store as uint8 or similar. These tensors are local to the call and never stored on the model, so `iter5.stamp` is unaffected.

Fast distributions, per position:
- p = the last (highest order) slow distribution `ps[-1]` from `m.base.predict`.
- For each order ascending: c = fast counts, total = c.sum; q = (c + 0.5*p) / (total + 0.5); p = q where total > 0, else unchanged. Append p as a fast input.

The mixer gets 10 inputs: 5 slow + 5 fast.

Fourth selector: longest = index (1..5) of the highest order with total > 0, 0 if none; distinct = number of chars with c > 0 at that order, clamped to 1..3. Selector = 0 if longest is 0, else (longest-1)*3 + distinct. So 16 rows.

Mixer structure: first layer `Neuron(r, 10, lr)` for r in (1, V+1, 4096, 16), second layer `Neuron(V+1, 4, lr)`, weights moved to cuda. `m.selectors(x)` plus the fourth selector goes into `mix`.

## Variants: `iter8.py <name> [--budget N]`, names `fast` and `fast_off`
- `fast`: as above.
- `fast_off` (control): identical structure, but all fast counts are zero. Same 10 inputs and 4 neurons. This isolates the extra mixer parameters from the memory itself.

## Output
Save `results/iter8_<name>.json` with test_loss, renamed_loss, val_loss, renamed_gap, train_seconds, chars, peak_allocated_mib, hit_rate (for each order, the fraction of test evaluation positions with total > 0), settings. Assert via `iter5.stamp` that evaluation wrote no state.

Causality test, run for every command before saving and also in smoke: take 4 evaluation windows, get `logprobs`, replace `x[:, 100:]` with other random valid chars, and assert the log-probs at positions 0..99 are exactly equal. Raise on failure.

`--budget` limits training chars for smoke checks.

## Decision rules (save in settings, do not act on them)
- Best so far: iter7 `boost` test 1.2451. `mix_full` test 1.2528, renamed 1.7547, gap 0.502. Backprop at 16M chars: test 1.1238, gap 0.406.
- `fast` test must be at most 1.195, or the approach is dropped.
- `fast` renamed gap must be below 0.45, or it is not fixing name dependence.
- `fast` must beat `fast_off` by at least 0.03 on test, or the memory is not what helps.
