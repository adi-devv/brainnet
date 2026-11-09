# Iteration 7: boosted closed-form stages

Work in `D:\Aadit\claudez\brainnet` only. Read `iter5.py`, `iter6.py`, `hybrid_mixer.py`, `benchmark.py`, then create `iter7.py` (at most 200 lines). Create nothing else except `results/iter7_*`. Do not change any existing file. No network, no pip, no git. Run Python only as `"C:\Users\aadit\AppData\Local\Programs\Python\Python312\python.exe" -X utf8 -B ...`. Smoke check on tiny budgets only (under 60 s total); do not run the full commands. Reply with at most 10 lines: what you wrote and the smoke result.

## Why
Lookup tables have saturated: learned codes added 0.002 over random codes. Test whether stacked closed-form logit stages, each fitted to the errors of the model so far, close the gap. Each stage is one linear solve; there is no chain rule through stages. No `.backward()`, `torch.autograd`, `torch.optim`; `torch.set_grad_enabled(False)`.

## Stage 0
Exactly `mix_full`: `iter5.Model(V, 1e-4)`, one pass over `tr[:cut]`, `cut=int(len(tr)*.95)`, bs=976, seed 0, cuda. Its test loss must come out within 0.005 of 1.2528; report it.

During that pass keep the online (predict-before-write) log-probs for targets in the second half of the pass, on CPU as float32, with their x and y windows. Stages fit only on these. Reason: replaying training chars through tables that already contain them would give almost no residual, and early online predictions come from nearly empty tables. Evaluation stays frozen, as in `iter5`.

## Stage k = 1..3
Scores: S_0 = stage 0 log-probs, S_k = S_(k-1) + eta_k * X_k W_k, p_k = softmax(S_k).

Fit with the Bohning bound on the multinomial Hessian, which gives one shared matrix and V right-hand sides:
- X = sparse binary features, D = 2**13 hashed dims, always including a bias feature.
- A = 0.5 * X^T X, accumulated with `index_add_` in sub-batches small enough for GPU memory. Dense D x D float32 on the GPU.
- b = X^T (onehot(y) - p_(k-1)), D x V.
- W = (A + lam I)^-1 b by Cholesky. Check the factorisation succeeded and report if not.
- Choose lam from (1e-3, 1e-2, 1e-1) * mean(diag A), and eta from (1, 2, 4), by loss on the `val` split (`tr[cut:]`), at each stage. Save the choice.

Features:
- Stage 1: `iter6.features(x, V)` reduced to D by `& (D-1)`, plus bias.
- Stages 2 and 3 in variant `boost`: for t, let top1 and top2 be the two highest chars of S_(k-1), and conf = bucket of max p_(k-1) in 4 equal bins. Features = each stage 1 feature hashed with top1 (through `iter6.mix`), plus hashed (top1, top2), (top1, conf), (top1, last char), plus bias.
- Stages 2 and 3 in variant `boost_flat` (control): the stage 1 features again, so this is plain repeated logistic regression, isolating what the crossing adds.

Features use only chars strictly before the target, computed identically at fit and at evaluation.

GPU memory under 2.7 GB, including the count tables. Free A and b between stages. Use the allocator cap from `iter6.run`.

## Commands (`iter7.py <name> [--budget N]`), names `boost` and `boost_flat`
Each saves `results/iter7_<name>.json` with: `stages`, a list for k = 0..3 of test_loss, renamed_loss, val_loss, lam, eta, fit_seconds; top-level test_loss, renamed_loss, val_loss taken from the last stage; train_seconds, chars, stage_chars, peak_allocated_mib, settings. Use `iter5.evaluate` and `iter5.rename`, and assert via `iter5.stamp` that evaluation wrote no state.

`--budget` limits base training chars for smoke checks.

## Decision rules (save in settings, do not act on them)
- Reference `mix_full`: test 1.2528, renamed 1.7547, gap 0.502. Backprop at 16M chars: test 1.1238, gap 0.406.
- `boost` final test must beat 1.2528 by at least 0.05, or the approach is dropped.
- Stage 3 must add at least 0.02 on test, or depth is not helping.
- `boost` must beat `boost_flat` by at least 0.02, or the crossing is not what helps.
- A renamed gap below 0.502 means less dependence on memorised names.
