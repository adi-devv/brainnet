Created only [iter18.py](D:/Aadit/claudez/brainnet/iter18.py). No git commands or full experiments run.

Using the specified Python with `-X utf8 -B`, ran both modes with `--smoke 32769`:

| Smoke output | copy | backprop_same |
|---|---:|---:|
| Characters trained | 32,769 | 32,641 |
| Test loss | 2.25199 | 3.90526 |
| Renamed loss | 2.33262 | 3.91084 |
| Validation loss | 2.50119 | 4.12432 |
| Novel first / repeat | 4.84214 / 2.58999 | 4.59711 / 4.59061 |
| Training seconds | 0.444 | 1.335 |
| Peak allocated MiB | 54.96 | 271.70 |

Copy’s pure-net test loss was **2.86458**, difference **+1.37238** from the full-run reference `1.4922`.

Match, finite-difference, and causality checks passed. Additional 4,097-character checks verified the no-copy switch, normalized gate weights, and unchanged network/Adam updates within observed GPU rounding variation.

Smoke pass bars: copy passed time, memory, and repeat-name; backprop passed time and memory. Neither passed test/renamed thresholds at this small budget.

Temporary results were cleaned up. Deviations: none.