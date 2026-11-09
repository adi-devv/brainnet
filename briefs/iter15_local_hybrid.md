# Iteration 15: exact local gradient, and tables + local net (fast memory + slow learner)

Work in `D:\Aadit\claudez\brainnet` only. Read `iter14.py` (whole file), `iter12.py` (whole file) and `iter5.py` (`batches`, `evaluate`, `rename`, `stamp`, `ROOT`, `B`). Create `iter15.py` (at most 170 lines). Import and reuse `iter14` and `iter12`; do not copy them. Create nothing else except `results/iter15_*`. Do not change any existing file. No network, no pip, no git. Run Python only as `"C:\Users\aadit\AppData\Local\Programs\Python\Python312\python.exe" -X utf8 -B ...`. Never stop any process. Smoke check on tiny budgets only (under 90 s total); do not run the full commands. Smoke runs write `results/iter15_smoke_<variant>.json` (mode 'x'). Reply with at most 8 lines.

No `.backward()`, `torch.autograd`, `torch.optim`; `torch.set_grad_enabled(False)`. All updates are written by hand.

## Why
Iter 14 (DFA, random fixed feedback) learned but lost to backprop at 6.2M chars (1.549 vs 1.33). Count tables (iter 12 `learned`) win at 6.2M (1.165) but do not generalize to renamed code (gap 0.44). Two ideas:
1. Exact local gradient: each layer's hidden error comes from the layer's OWN readout (`R_l`), not a random matrix. This is the true gradient of that layer's own loss; still no chain rule across layers.
2. Brain-style two systems: tables = one-shot fast memory; a slow local-rule net learns only what the tables miss (its logits are added to the tables' log-probs).

## Variants: `iter15.py {local,hybrid,tables} [--budget N | --full] [--smoke]`
Default budget 999_424; `--full` = all training chars (`budget=None`). Result file `results/iter15_<variant>_<1m|full>.json` (smoke: `results/iter15_smoke_<variant>.json`).

- `local`: iter14 `Model` with L=3, bs 16, lr 1e-3, identical to iter14 `dfa` except the hidden error for layer l is `delta_l = (err_l @ R_l.T) * (1 - h_l^2)` using the CURRENT `R_l` (computed before this step's update). Everything else as iter14.
- `tables`: iter12 `Model(V,'learned')` trained exactly as iter12 `run` does (bs 976, per position, head updates + `m.base.write`). Control.
- `hybrid`: same tables, plus an iter14-style net (L=3, D=256, local rule as `local`). Per training batch of 976 windows:
  1. `lp = tables.logprobs(x)` BEFORE this batch's writes (prequential: the net only sees table predictions for text not yet written).
  2. Train the tables on the batch exactly as `tables`.
  3. Train the net on the same batch in sub-batches of 64 windows (one Adam step each), with the table output as a fixed prior: layer l logits are `lp + z_l`, final logits `lp + sum_l g_l z_l`; `err_l = softmax(lp + z_l) - onehot(y)` and the final error for `g` uses `lp + sum g z`.
  - `R_l`, `c_l` start at ZERO, so the untrained hybrid equals the tables.
  - Prediction: `log_softmax(lp + sum_l g_l z_l)` with `lp` from the trained tables.
  - Keep the 2.7 GiB allocator limit from iter12; chunk anything that does not fit.

## New metric: in-context use of new names (sample efficiency)
On the renamed test windows (`B.eval_windows` of renamed and of original test give aligned windows; renaming keeps lengths), take target positions whose char differs from the original (inside a renamed identifier). Identifier = maximal run of `[A-Za-z0-9_]` in the window. Label a position `repeat` if the same renamed identifier string occurred as a complete run earlier in the same window, else `first`. Report mean per-char CE `novel_first_loss`, `novel_repeat_loss` and their counts. A model that learns a new name from one sighting has a low repeat loss.

## Required checks (assert in every run)
1. No autograd, grad mode off, parameters plain tensors.
2. Finite difference: for `local` and `hybrid`, the hidden delta of each layer equals the numeric derivative of that layer's own CE (with the `lp` prior in hybrid; use a random fixed `lp`) with respect to `h_l` pre-tanh activation, on a tiny float64 input, a few entries.
3. Causality: changing x at position 47 leaves the final log-probs at positions < 47 unchanged (all variants; hybrid includes tables + net).
4. Evaluation changes nothing (`I.stamp` on the model, tables included).
5. Full runs only: mean training loss over the last 10% of net steps is below the first 10% (`local`, `hybrid`).

## Output
test_loss, renamed_loss, renamed_gap, val_loss, novel_first_loss, novel_repeat_loss, novel_counts, per_layer_test_loss (local/hybrid), g, train_seconds, chars, steps, peak_allocated_mib, settings (incl. decision rules).

## Decision rules (save in settings, do not act on them)
- References: backprop 1M 2.330; backprop 6.2M 1.329; backprop 16M 1.124 (renamed 1.530); iter14 dfa 1M 1.992, 6.2M 1.549; iter12 tables 6.2M 1.165 (renamed 1.606, gap 0.441).
- `local` 1M <= 1.962 (dfa - 0.03): the exact local gradient beats random feedback; run `local --full`.
- `hybrid` beats `tables` at the same budget by >= 0.02 test or >= 0.04 renamed: the net learns what tables miss; run `hybrid --full`.
- `hybrid --full` test < 1.124: beats backprop at 16M with 2.6x less data, without backprop.
- Novel-name learning: (first - repeat) larger for hybrid than tables = better one-shot use of new names.
