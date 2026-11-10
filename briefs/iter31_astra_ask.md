# Ask: iter 29 timing control done; what next

Same project (local learning, no cross-layer backprop; goal: faster AND better than backprop; dataset, one pass, budget fixed). Read: `briefs/iter30_astra_out.md` (your last advice), `iter27.py`, `iter29.py`, `results/timing/run*_iter*.json`, `results/iter27_width320*.json`, `results/iter16_backprop_6215526.json`, `README.md` findings. Do not edit files or run training.

## AB/BA timing, seed 0, same session, on AC (separate processes)
| run | model | val | train s |
|---|---|---|---|
| 1 | iter27 | 1.45641 | 70.9 |
| 2 | iter29 | 1.44855 | 92.3 |
| 3 | iter29 | 1.44868 | 83.0 |
| 4 | iter27 | 1.45640 | 80.1 |
Median: iter27 75.5 s, iter29 87.6 s (+12 s, within your 5-18 s estimate). Paired val gain 0.0078. Passes your rule on seed 0 with ~2 s headroom.

## Where we stand vs backprop (6.2M chars, one pass)
iter27 3-seed: test 1.3215, renamed 1.621, ~75 s, ~136 MB. Backprop: test 1.332 (2 seeds), renamed 1.667, 180 s, 321 MB; 1.124 test at 16M chars. Iter 27 has no learning curve logged; backprop has 11 points. Backprop got no schedule tuning comparable to ours.

## Questions (short, ranked, with odds/expected gains)
1. Is +12 s (16%) a problem? Decision: run iter29 seeds 1-2 as is, or first implement your exact cheaper gate gradient (fused projection, cached pi, fewer syncs), then seeds? Expected time after optimization?
2. Bigger picture: the strongest next 2-3 iterations toward a clear win over backprop (quality, not just speed). Consider: fair backprop control (same hold+cosine schedule), 16M-char scaling run, logging learning curves for sample efficiency, and architecture/learning-rule ideas. Rank by value per GPU minute.
3. Anything in the evidence we're misreading.
