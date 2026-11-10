# Ask: iter 27/28 results, and what next

You are advising a research project. Goal: a transformer-like architecture that learns without cross-layer backprop and is both faster and better than backprop. We are researching new approaches, not proven ones. Dataset, one pass over 6.2M chars, and budget are fixed (no "add more data").

Read first: `README.md`, `PLAN.md`, `iter24.py`, `iter26.py`, `iter27.py`, `iter28.py`, `briefs/iter27_astra_out.md` (your previous advice), `results/iter27_width320*.json`, `results/iter28_softtanh.json`, `results/iter26_ema_gates*.json`. Do not edit any files, and do not run training (the GPU is shared).

## Results (val; iter 24 vals seed0/1/2: 1.46686 / 1.4693 / 1.4686)
| | seed | val | gain | test | renamed | train s |
|---|---|---|---|---|---|---|
| iter 27: widths 256/256/320 | 0 | 1.4563 | .0106 | 1.3232 | 1.6232 | 70.9 |
| | 1 | 1.4589 | .0104 | 1.3210 | 1.6234 | 83.2 |
| | 2 | 1.4595 | .0091 | 1.3203 | 1.6166 | 88.7 |
| iter 28: leaky tanh .95tanh+.05a, layers 1-2 | 0 | 1.4803 | -.0135 | 1.3435 | 1.6434 | 98.0* |
| iter 26: EMA gates (not promoted) | 0/1/2 | mean gain .0089 | | ~1.321 | ~1.62 | ~79-91 |

- Iter 27 mean gain 0.01005 → passes the rule (<90 s each, mean ≥0.010); it is the new base. Killed iter 28 on val.
- *Timing noise: the laptop was off charger for some runs, which slows it heavily (iter 28's extra 27 s is machine, not code; it adds only a few elementwise ops). Seed spread of time 71-89 s is likely the same cause. On charger, iter 24 is ~65-78 s.
- Backprop baseline: test 1.3289 (seed 1 1.3352), renamed 1.6675, 180 s, 321 MB. Same-sampler backprop: test 1.3407, renamed 1.6863. All iter 27 seeds beat it on test and renamed at ~2-2.5x speed.

## Questions (short answers, ranked, with odds)
1. Why did leaky tanh hurt so much? Does it tell us anything (e.g., the bounded nonlinearity matters, saturation is useful)?
2. Next iteration from iter 27: gates (iter 26) on top of width 320 (gates cost ~4 s; time margin is thin), wider top layer (384?), widening layer 1, or something new you think is more promising. Give a concrete spec, expected gain, and a kill rule.
3. Is the 90 s / 0.010 rule still the right bar now, given timing noise from power state? Should we measure time differently (e.g., per-step GPU time, or a same-session control)?
4. Toward the end goal ("faster AND better than backprop"): what claims are now justified, and what evidence is still missing (e.g., a stronger/tuned backprop baseline, validation-matched comparison)?
