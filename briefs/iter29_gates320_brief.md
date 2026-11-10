# Build iter29.py: iter 27 (widths 256/256/320) + iter 26's EMA gates

Repo: this folder. Read `iter24.py`, `iter26.py`, `iter27.py`, `briefs/iter29_astra_out.md` (item 2). Do NOT edit existing files. Do NOT run full training (GPU shared); only the smoke command below.

## Spec
- New file `iter29.py`. Combine exactly two existing changes on top of iter 24:
  - iter 27's top-layer widening (W2 1536x320, b2 320, R2 320xV, its dedicated-generator redraw and init rules), and
  - iter 26's content-dependent softmax gates at layers 1 and 2: zero-init `G1,G2: 256x3`, `d1,d2: 3`, `pi = softmax(lower @ G + d)`, each EMA branch scaled by `3*pi_k`, exact layer-local gate gradients, gate LR 0.1x network LR.
- Prefer reuse: e.g. subclass `iter27.Model` and port iter26's `_pi/_layer/_local_grads/_update` (or a mixin), so behavior matches each parent exactly where they don't interact. Anything in the gate code hard-coded to 256 for layer 2's output must use 320.
- Init: gates are zero-init, so no extra RNG draws; all non-gate params must equal a fresh iter27 model of the same seed exactly.
- Everything else unchanged: tanh, local losses, logit mixing g, copy heads, LR schedule, sampler, bs 16, T 128.

## Checks (startup)
- At init, every non-gate param equals a fresh iter27 seed-0 model exactly; at init (zero gates, pi=1/3, scale 1) the forward outputs match iter27 within atol 1e-6 (GPU index_add_ is nondeterministic ~2e-8, never use atol 0 across runs).
- Layer-1 and layer-2 local gradients (incl. G, d) match torch.autograd on a tiny float64 case.
- Causality: changing a future token does not change past outputs at any layer.
- Reuse iter26/iter27 checks where they still hold. If you take a param "stamp" that includes `vars(obj)`, take it after any monkeypatching.

## Output
Result `results/iter29_gates320.json` (seed>0: `iter29_gates320_seed{seed}.json`), `open('x')`; keep iter27's eval fields + iter26's gate stats (mean pi per layer, saturation fractions), settings record widths and gates, and add `iter27_reference=dict(val_loss=1.45627, test_loss=1.3232, renamed_loss=1.6232)`. CLI `--smoke N`, `--seed`.

## Verify
Smoke: `/c/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B iter29.py --smoke 204800`. If your sandbox can't run Python with torch, say so and stop; I'll run it. Report in `briefs/iter29_codex_out.md`: diff summary, smoke output if run, any deviation from spec.
