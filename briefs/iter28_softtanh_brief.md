# Build iter28.py: iter 24 with a leaky tanh in the upper layers

Repo: this folder. Read `iter24.py`, `iter22.py`, `iter15.py`, `iter26.py` (pattern to follow), `briefs/iter27_astra_out.md` (item 2). Another agent is writing `iter27.py` at the same time: do NOT read, edit or run it. Do NOT run full training (GPU shared); only run the smoke command below.

## Spec
- New file `iter28.py`, `import iter24 as S`, `Model(S.Model)` subclass, same layout as `iter26.py`.
- Layers 1 and 2 only: `h = 0.95*tanh(a) + 0.05*a`. Layer-local derivative `0.95*(1 - tanh(a)**2) + 0.05`, computed from tanh(a) (not from h). Layer 0 stays plain tanh.
- Everything else unchanged: init (bit-identical to iter 24, no extra RNG draws), local losses, logit mixing g, copy heads, LR schedule, sampler, bs 16, T 128.

## Checks (startup)
- Layer 0 output and all params at init equal a fresh iter24 seed-0 model exactly.
- Layer-1 and layer-2 local gradients match torch.autograd on a tiny float64 case.
- Causality: changing a future token does not change past outputs at any layer.
- Reuse iter24's checks where they still hold; adapt any that assumed tanh at layers 1-2.

## Output
Result `results/iter28_softtanh.json` (seed>0: `iter28_softtanh_seed{seed}.json`), `open('x')`; keep iter24's eval fields, settings record the activation, and add `iter24_reference=dict(val_loss=1.46686, test_loss=1.3320, renamed_loss=1.6298)`. Also record, on the validation windows, per upper layer: fraction |tanh(a)|>.99, mean derivative, mean |h|, max |h|. CLI `--smoke N`, `--seed`.

## Verify
Run: `/c/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B iter28.py --smoke 204800` and make it pass. Report: diff summary, smoke output (losses, seconds), any deviation from spec.
