Created only [iter20.py](D:/Aadit/claudez/brainnet/iter20.py). No git commands or full experiments.

Ran all three modes with `--smoke 32769` using the specified Python and flags; `learned_input` used `--seed 1`. Total runtime was about 30 seconds. Each run completed 17 steps, and all startup checks passed.

| Mode | Test loss | Renamed loss | Training seconds |
|---|---:|---:|---:|
| assoc | 4.1262 | 4.2228 | 0.301 |
| noretrieve | 4.3037 | 4.4103 | 0.321 |
| learned_input | 3.8143 | 3.9195 | 0.564 |

Smoke reports were cleaned up. Timing bars passed; loss bars failed at this short budget.

Interpretation: `INPUT_SCALE` multiplies completed `s[0]`, documented in settings. Otherwise, no deviations.