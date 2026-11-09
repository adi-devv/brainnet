# Consult only: what should brainnet try next?

Do not edit, create or run anything. You may read the repo (`iter14.py`, `iter15.py`, `iter17.py`, `baseline.py`). Answer in under 300 words.

## Goal
A new transformer-like architecture that learns without backprop and is both faster and better than backprop. This is research: we want approaches worth trying, not proven recipes. Evolution strategies, RL² and "lifetimes" are out of scope.

## Setup
Character-level code language model. Test loss in nats/char, lower is better. GTX 1650 4 GB, one seed unless noted. Every run should take minutes.

## Results
| Method | Data | Train time | Test | Renamed-identifier test | Gap | Memory |
|---|---|---|---|---|---|---|
| Backprop GPT (1.23M params, width 128, 6 layers) | 2.5M chars | 73 s | 1.726 | 1.985 | 0.259 | 321 MB |
| Backprop GPT | 6.2M | 180 s | 1.329 (seed 1: 1.335) | 1.667 | 0.339 | 321 MB |
| Backprop GPT | 16M | 465 s | 1.124 | 1.530 | 0.406 | 321 MB |
| DFA stacked learner (iter 14) | 6.2M | 70 s | 1.549 | 1.862 | 0.313 | |
| `local` (iter 15), 1 pass | 6.2M | 66-73 s | 1.492 | 1.819 | 0.327 | 105 MB |
| `local`, 2 passes | 12.4M | 131 s | 1.382 | 1.755 | 0.373 | 105 MB |
| `local`, 2.5 passes, constant lr | 15.5M | 163 s | 1.353 | 1.720 | 0.367 | 105 MB |
| `local`, 2.77 passes, cosine lr to lr/10 | 17.2M | 175 s | 1.377 | 1.739 | 0.363 | 105 MB |
| Hashed count tables + learned pointer heads (iter 12) | 6.2M | 244 s | 1.165 | 1.606 | 0.441 | ~1.9 GB |
| Tables + local net (hybrid) | 6.2M | 707 s | 1.155 | 1.596 | 0.441 | ~2 GB |

`local` is 3 layers, width 256, no attention: fixed causal EMAs plus 6 lag lookups as input. Each layer has its own linear readout and updates from its own cross-entropy only, with no cross-layer gradient. Per-layer test loss at 2.5 passes: 1.446, 1.392, 1.376.

New-identifier loss, first sight vs a repeat inside the same window:
- Backprop 6.2M: 5.64 vs 5.21
- `local`: 5.77 vs 5.45 (1 pass), 5.99 vs 5.69 (2.5 passes)
- Tables at 1M: 5.98 vs 3.20

So tables are one-shot memories and copy new names well, but cost much more memory. No net copies well.

Scorecard: backprop still wins equal time at 180 s and equal data. `local` only wins equal time at 73 s (1.492 vs 1.726).

## Question
If you had to pick exactly one next experiment, what is it?

Give:
1. The pick, and the mechanism that makes it plausibly beat backprop at both equal time and equal data.
2. The exact smallest test, with numeric pass/fail bars against the table above.
3. Why not your two closest alternatives.
4. What in these results you distrust, if anything.

Say plainly if you think the current path cannot reach the goal.
