# Ask: iter 29 (iter 27 + EMA gates) seed 0, and what next

Same project as before (local learning, no cross-layer backprop; goal: faster AND better than backprop; data/one pass/budget fixed). Read: `iter24.py`, `iter26.py`, `iter27.py`, `iter29.py`, `briefs/iter29_astra_out.md` (your last advice), `results/iter29_gates320.json`, `results/iter27_width320*.json`, `results/iter26_ema_gates*.json`. Do not edit files or run training.

## Result (seed 0)
| | val | test | renamed | train s |
|---|---|---|---|---|
| iter 27 s0 | 1.45627 | 1.3232 | 1.6232 | 70.9 |
| iter 29 s0 | 1.44868 | 1.3162 | 1.6148 | 107.3 |
- Paired val gain 0.0076 (your forecast: ~0.006). Gate means: layer1 pi [.674,.102,.223], layer2 [.587,.105,.309].
- Laptop was on AC (checked right after; GPU idle at 300 MHz then). Iter 26 gates on iter 24 cost ~4 s back-to-back; here +36 s vs iter 27 seed 0. Not yet verified with a same-session control.

## Questions (short, ranked, with odds)
1. From the code: is +36 s plausible as real compute (e.g., gate gradient doing 3 branch projections against W at width 320, extra cats/stacks, memory), or must it be machine noise? Point to the specific lines that cost the most and estimate the real overhead.
2. If real: a concrete cheaper implementation of the same math (exact, layer-local) that should cut the overhead — e.g., reuse quantities already computed in the W gradient, fuse projections, avoid materializing gated blocks. Give a spec Codex can implement and a check that it's numerically identical.
3. Decision now: same-session timing control (iter 27 vs iter 29), optimize first, or kill? And the promotion rule to apply (yours: mean gain ≥0.005, all 3 seeds improve, controlled median ≤90 s).
