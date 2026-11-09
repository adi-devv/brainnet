# Iteration 3: pay more attention to new things than frequent ones

Same rules, data, 1M budget, evaluation and limits as `briefs/iter2_hybrid.md`. No backprop. Do not modify `benchmark.py`, `baseline.py`, `LOCK.json`, or `data/`.

## Change
Other iteration 2 runs are still using `hybrid.py`, so do not edit it or `test_hybrid.py`. Copy `hybrid.py` to `hybrid_novelty.py` and `test_hybrid.py` to `test_novelty.py` (pointing at the new module), and make all changes in the copies. Run Python only as `"C:\Users\aadit\AppData\Local\Programs\Python\Python312\python.exe" -X utf8 ...`.

Add these options to `hybrid_novelty.py`, all off by default, so its behaviour matches `hybrid.py` when they are off.

1. **Novelty scaling (`novelty`):** keep a visit count per neuron. Scale the Hebbian update and the memory write by `1 / sqrt(1 + visits)` of the active neurons, so frequently seen patterns change the model less and new ones change it more.
2. **Surprise gate (`gate`):** if the model already gave the true next character probability above a threshold (default 0.9), skip the Hebbian update and memory write for that position. Count how many positions were skipped.

## Known risk to measure
The memory predicts from counts, so down-weighting or skipping frequent patterns can make its probabilities wrong (common characters get under-counted). Report whether test loss gets worse for this reason. If it does, try one fix within the time limit: keep updating a small separate frequency table for all positions, and use it only for the backoff/smoothing.

## Base settings
Use the settings of the `full` variant from `results/iter2_full.json`, and that file's test loss for the recheck. Call this `base`. Use at most 4 CPU cores in total, since other runs are still active.

## Variants (parallel, one process each, seed 0, CPU only)
| Name | Setup |
|---|---|
| `base_recheck` | `base` with both options off. Must match the iteration 2 score within 0.002, otherwise stop and report |
| `novel_scale` | `base` + novelty |
| `surprise_gate` | `base` + gate |
| `novel_gate` | `base` + novelty + gate |

## Checks
Run `test_novelty.py` first, and it must pass. Add a check that the gate really skips positions on a repeating pattern.

## Output
Save `results/iter3_<name>.json` and `results/iter3_summary.json`: test loss, train seconds, chars per second, and fraction of positions skipped. End with a plain summary table, and state whether novelty or the gate improved test loss, speed, or both, and anything that failed.
