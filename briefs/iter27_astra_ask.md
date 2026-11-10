# Ask: iter 26 rerun result, and what next

You are advising a research project. Goal: a transformer-like architecture that learns without cross-layer backprop and is both faster and better than backprop. We are researching new approaches, not proven ones. Dataset, one pass over 6.2M chars, and budget are fixed (no "add more data").

Read first: `README.md`, `PLAN.md`, `iter24.py`, `iter26.py`, `briefs/iter26_astra_out.md` (your previous advice), `results/iter26_ema_gates.json` (rerun), `results/iter26_ema_gates_run1_slow.json` (first run). Do not edit any files, and do not run training (the GPU is shared).

## What iter 26 is
Iter 24 (base) + content-dependent softmax gates over the 3 fixed causal EMA branches at each upper layer: `pi = softmax(lower @ G + d)`, zero-init, each EMA block scaled by `3*pi_k`, exact layer-local gradient, gate LR 0.1x network.

## Results (seed 0)
| | iter 26 | iter 24 |
|---|---|---|
| val | 1.4583 | 1.4669 (seeds 1,2: 1.4693, 1.4686) |
| test | 1.3213 | 1.3320 |
| renamed-identifier | 1.6197 | 1.6298 |
| train time, back-to-back same session | 82.5 s | 78.2 s |

- First run took 111 s; that was machine state. After a reboot, a back-to-back timing shows the gates cost about 4 s (~5%).
- Kill rule (yours): each run <90 s and mean paired val improvement >= 0.010. Seed 0 improvement is 0.0085: it fails the val bar narrowly. Seeds 1, 2 not run yet.
- Gate stats (test): layer1 mean pi [.674,.102,.223], layer2 [.608,.108,.284]; tanh saturation fraction (|h|>.99) per layer [.015,.412,.586] (was this high in iter 24? check if you can from code).
- Backprop baseline: test 1.3289 (seed 1 1.3352), renamed 1.6675, 180 s, 321 MB. Same-sampler backprop: test 1.3407, renamed 1.6863.

## Questions (short answers, ranked, with odds)
1. Run seeds 1, 2 for iter 26, or kill it now? Is 0.0085 at seed 0 likely real given iter 24's seed spread?
2. Is the high saturation at layers 1-2 a problem worth fixing (and how, cheaply), or noise?
3. What is the single best next iteration toward "faster AND better than backprop" given everything so far? Include width 320 in the ranking, and anything new you think is more promising. Give a concrete spec, expected gain, and a kill rule.
