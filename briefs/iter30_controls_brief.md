# Build iter30.py: two decisive controls on iter 27

Repo: this folder. Read `iter22.py`, `iter24.py`, `iter27.py` (and what they import). Do NOT edit existing files. Do NOT run full training (GPU shared); only the smoke commands below.

Goal: isolate (a) the learning rule and (b) fixed EMAs vs learned attention. Everything not named below stays exactly as iter 27 seed 0: widths 256/256/320, init, sampler, batch 16, T 128, steps, LR schedule, Adam, copy heads, logit mixing g, evals.

## CLI `--mode {local,backprop,attn}`, `--seed`, `--smoke N`
- `local`: iter 27 unchanged (plus the curve logging below). Must reproduce iter27 within GPU nondeterminism.
- `backprop`: same model, same losses (the same per-layer readout losses iter 27 trains on, summed), but gradients flow ACROSS layers: no detach between layers, every layer receives gradients from all losses above it. Standard order: full forward, then one backward (torch.autograd is fine), then one Adam update of all params per step (no interleaved per-layer updates). Same Adam/LR/schedule/clipping as iter 27.
- `attn`: iter 27 + one causal multi-head softmax attention block feeding the TOP layer (layer 2): queries/keys/values from layer 1's output (256), 4 heads x 64, RoPE on q,k, causal mask within the T=128 window, output projection O (256x256) zero-init. Its 256-d output is concatenated to layer 2's input, so W2 gets 256 extra input rows (init by iter 27's W2 rule, drawn from a separate generator so every iter27 param stays identical). Learning stays LOCAL: attention params and W2 learn only from layer 2's own loss; inputs from layer 1 are detached (no gradient to layers 0-1). Within-layer gradients exact (autograd on detached inputs is fine). Same Adam and LR as the network.

## Curve logging (all modes)
Every 10% of steps, evaluate validation loss on a fixed subset (first 1/4 of the val windows, same each time), record `curve=[(chars_seen, val_subset_loss)]`. Exclude eval time from `train_seconds` (synchronize before/after), report `eval_seconds` separately. Final full val/test/renamed evals as iter 27.

## Checks (startup, fast)
- All modes: at init every iter27 param equals a fresh iter27 seed model exactly.
- `attn`: at init (O=0) forward outputs match iter27 within atol 1e-6; causality (changing a future token leaves past outputs unchanged at every layer); layer-2 local gradient matches float64 autograd on a tiny case; layers 0-1 get no gradient from layer 2.
- `backprop`: on a tiny float64 case, with cross-layer flow blocked (detach) its per-layer grads equal iter 27's local grads; with flow enabled, lower-layer grads differ (nonzero cross-layer term).

## Output
`results/iter30_{mode}.json` (seed>0: `iter30_{mode}_seed{seed}.json`). Keep iter27's eval fields, add `curve`, `eval_seconds`, `mode`, param counts, peak memory, and `iter27_reference=dict(val_loss=1.45627, test_loss=1.3232, renamed_loss=1.6232)`.

## Verify
Smoke each mode: `/c/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B iter30.py --mode MODE --smoke 204800`. If your sandbox cannot run Python, still write iter30.py completely; skip the smoke and I will run it. Report in `briefs/iter30_codex_out.md`: summary, smoke outputs, any deviation.
