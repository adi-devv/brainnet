# Iteration 11: surprise clock + iter 9 inputs

Work in `D:\Aadit\claudez\brainnet` only. Read `iter9.py`, `iter10.py`, `iter8.py`, `iter5.py`, `hybrid_mixer.py`, then create `iter11.py` (at most 100 lines). Create nothing else except `results/iter11_*`. Do not change any existing file. No network, no pip, no git. Run Python only as `"C:\Users\aadit\AppData\Local\Programs\Python\Python312\python.exe" -X utf8 -B ...`. Never stop any process. Smoke check on tiny budgets only (under 60 s total); do not run the full command. Reply with at most 8 lines: what you wrote and the smoke result.

## Build
One variant: `iter11.py combo [--budget N]`. No `.backward()`, `torch.autograd`, `torch.optim`; `torch.set_grad_enabled(False)`.

Subclass `iter9.Model` with `variant='all'` behaviour (recency and copy both on). Add exactly what `iter10.Model` does for `variant='clock'`: surprise computed in `prepare` before training writes (thresholds 1.0 and 3.0), `iter10.clocked`, the 6 clocked tables of `1 << 23` entries, the same backoff chains from `ps[0]`, written only when `y is not None`, and iter 10's 12-row selector (`gap_bucket*2 + A3 hit`).

22 inputs in this order: iter 9's 16, then the 6 clocked. First layer `Neuron(r, 22, lr)` for r in (1, V+1, 4096, 16, 17, 12). Second layer `Neuron(V+1, 6, lr)`. Weights on cuda. Lr 1e-4.

Hint: overriding `mix` to append the clocked distributions and selector before calling the parent is one way to reuse `iter9.Model.output` unchanged, including its copy-table update after mixing. Do not copy-paste iter 9's logic if you can reuse it.

Same setup as iter 10: `iter8.fit`, `tr[:cut]`, one pass, bs=976, seed 0, cuda, allocator cap 2.7 GB. Run `iter9.reference(m)` and `iter10.reference()`. Causality must cover iter 9's checks and iter 10's clocked-hash check for positions 0..99. Evaluation must leave slow tables, clocked tables, mixer and copy confidence unchanged (assert it).

## Output
`results/iter11_combo.json`: test_loss, renamed_loss, val_loss, renamed_gap, train_seconds, chars, peak_allocated_mib, hit_rate, copy_rate, copy_accuracy, fire_rate_A, fire_rate_B, settings.

## Decision rules (save in settings, do not act on them)
- Reference: iter 9 `all` test 1.1843, gap 0.4373. Iter 10 `clock` test 1.1711, gap 0.4559. Backprop at 16M chars: test 1.1238, gap 0.406.
- `combo` test <= 1.160 means the two sets of inputs add up; above that they overlap and input stacking stops.
- `combo` renamed gap < 0.43 means name dependence improved.
