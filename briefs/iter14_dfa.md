# Iteration 14: table-free stacked local learner (kill test at 1M chars)

Work in `D:\Aadit\claudez\brainnet` only. Read `iter5.py` (`batches`, `evaluate`, `rename`, `stamp`, `ROOT`, `B`) and `iter12.py` (how `run` evaluates, renames, writes results), then create `iter14.py` (at most 150 lines). Create nothing else except `results/iter14_*`. Do not change any existing file. No network, no pip, no git. Run Python only as `"C:\Users\aadit\AppData\Local\Programs\Python\Python312\python.exe" -X utf8 -B ...`. Never stop any process. Smoke check on tiny budgets only (under 60 s total); do not run the full commands. Smoke runs write `results/iter14_smoke_<variant>.json`, never the real names (open results in mode 'x'). Reply with at most 8 lines.

No `.backward()`, `torch.autograd`, `torch.optim`; `torch.set_grad_enabled(False)`. All updates are written by hand.

## Why
Iters 5-13 are count tables plus hand-picked features; nothing learns that two contexts are alike, which is what renaming tests. Iter 12 with tables off scored 2.53 at 6.2M chars. This iteration drops tables entirely: dense learned vectors, depth, and a local rule with no chain rule across layers. It is a cheap kill test at 1M chars. Backprop transformer at 1M chars: test 2.330 (mean of 3 seeds).

## Model (V = vocab size, pad id = V, input ids in 0..V, outputs V classes)
Everything is computed for all positions of a batch of windows at once (x: [bs,128]); position t sees only x[:, :t+1].
- Causal matrices: fixed lower-triangular `M_k[t,s] = (1-d_k) * d_k^(t-s)` for s<=t, decays d = (0.5, 0.8, 0.95), applied along time by matmul.
- Layer 1 (width D=256): `a1 = sum_{lag=0..5} W1[lag][x_{t-lag}]` (table lookup rows of size D; lag beyond the window start uses pad) `+ sum_k M_k(E[x]) @ Pk` where `E` is a V+1 by 32 table and `Pk` a 32 by D matrix; plus bias. `h1 = tanh(a1)`.
- Layers 2..L: input `u = concat(h_{l-1}, M_1 h_{l-1}, M_2 h_{l-1}, M_3 h_{l-1})` (4*D wide, causal); `a_l = u @ W_l + b_l`; `h_l = tanh(a_l)`. Previous-layer activations are constants (no gradient flows to lower layers).
- Each layer l has its own readout `R_l` (D by V, bias) giving logits_l from h_l.
- Final prediction: `log_softmax(sum_l g_l * logits_l)`, `g` init 1/L, learned.

## Local learning rule (direct feedback alignment)
Per step, per layer l, with `err_l = softmax(logits_l) - onehot(y)` on valid positions (y>=0), averaged over valid positions:
- `R_l` update: the exact single-layer delta rule (h_l^T err_l).
- Hidden update: `delta_l = (err_l @ B_l) * (1 - h_l^2)` with `B_l` a FIXED random V by D matrix (gaussian, std 1/sqrt(V), seeded, never trained). Then update W_l / lookup rows / E / Pk by the outer product of their inputs with `delta_l` (for E: push delta through Pk and the M_k transposes — that is only a within-layer linear map, still local to layer 1).
- `g` update: manual gradient of the final CE with respect to g.
- Optimizer: hand-written Adam-style (or Adagrad) per parameter tensor, lr 1e-3 default (flag `--lr`), no weight decay. Gradient clip each update tensor to norm 1.
- Data: `I.batches(tr[:cut], budget, bs, True, V)` with `cut = int(len(tr)*.95)`, one pass, bs windows = 16 by default (flag `--bs`), seed 0, cuda.

## Variants: `iter14.py {dfa,shallow} [--budget N] [--bs N] [--lr X] [--smoke]`
- `dfa`: L = 3.
- `shallow`: L = 1 (control for whether depth helps).
Default budget 999_424 (the baseline's 1M). Run one at a time.

## Required checks (assert in every run)
1. No autograd: model parameters are plain tensors with `requires_grad=False`; grad mode off.
2. Finite-difference check of the delta rule: on a tiny random input, the `R_l` update equals the analytic gradient of the layer's CE (compare with a numeric central difference on a few entries).
3. Causality: changing x at position t0 leaves outputs at positions < t0 unchanged (every layer).
4. Evaluation leaves every parameter, Adam state and `g` unchanged (`I.stamp`).
5. Loss decreases: mean training loss over the last 10% of steps is below the first 10% (assert in full runs only, not smoke).

## Output
`results/iter14_<variant>.json`: test_loss, renamed_loss, renamed_gap, val_loss (on tr[cut:]), per_layer_test_loss (list, each layer's own readout alone), g, train_seconds, chars, steps, peak_allocated_mib, settings (incl. decision rules below). Evaluate with `I.evaluate`/`I.rename` as in iter 12.

## Decision rules (save in settings, do not act on them)
- Backprop at 1M chars: test 2.330. Iter 12 `pure` at 6.2M chars: 2.53 / renamed 2.59 / gap 0.061.
- `dfa` test < 2.33: local rule learns representations at least as fast as backprop here; scale it.
- `dfa` test < `shallow` test by >= 0.03: depth helps without backprop; keep stacking.
- Otherwise: the local rule cannot learn representations here; next is a different local rule or hybrid.
