# Build iter27.py: iter 24 with only the final layer widened to 320

Repo: this folder. Read `iter24.py`, `iter22.py`, `iter15.py`, `iter26.py` (pattern to follow), `briefs/iter27_astra_out.md`. Do NOT run full training (GPU shared); only run the smoke command below.

## Spec
- New file `iter27.py`, `import iter24 as S`, `Model(S.Model)` subclass, same layout as `iter26.py`.
- Widen ONLY layer 2 (the top layer) from 256 to 320 units: `W2` 1536x320, `b2` 320, `R2` 320xV, Adam states for each matched (zeros). Layers 0 and 1, `E`, `g`, copy gate: unchanged.
- Init: same rules as the existing layer-2 init (same distributions; readout scaled 1/sqrt(320) where the original uses 1/sqrt(256); lag rows 1024:1536 of W2 zero). Global RNG stream must be unchanged: build the iter24 model first, then redraw the layer-2 tensors from a dedicated `torch.Generator` seeded from the current seed, so layers 0-1 AND the gate built afterwards are bit-identical to iter 24.
- Anything hard-coded to 256 for layer 2 (forward, local grads, delta, checks, eval) must use 320. Layer 1 output stays 256 and still feeds layer 2 (input width 1536 unchanged).
- Keep local losses, logit mixing g, copy heads, LR schedule, sampler, bs 16, T 128 unchanged.

## Checks (startup, like iter24/iter26)
- Shapes of W2/b2/R2 and their states; W2 lag rows zero.
- At init, every non-layer-2 parameter and the gate equal a fresh iter24 seed-0 model exactly.
- Layer-2 local gradient matches torch.autograd on a tiny float64 case (exact within-layer gradient).
- Causality: changing a future token does not change past outputs at any layer.
- One training step on a real batch runs and leaves layers 0-1 identical to iter24 for the first step (layers 0-1 updates do not depend on layer 2).
- Reuse iter24's checks where they still hold; adapt any that assumed 256 at layer 2.

## Output
Result `results/iter27_width320.json` (seed>0: `iter27_width320_seed{seed}.json`), `open('x')`, settings record widths [256,256,320]; keep iter24's eval fields and add `iter24_reference=dict(val_loss=1.46686, test_loss=1.3320, renamed_loss=1.6298)`. CLI `--smoke N`, `--seed`.

## Verify
Run: `/c/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B iter27.py --smoke 204800` and make it pass. Report: diff summary, smoke output (losses, seconds), any deviation from spec.
