Added [iter27.py](C:/Users/aadit/AppData/Local/Temp/claude/2e162b2224a69697/71efcaa5-cf65-4cbc-90fa-3206306ba725/scratchpad/bn/iter27.py): only layer 2 widened to 320, dedicated RNG, matching Adam states, startup checks, and requested result naming/metadata.

Requested smoke passed—all checks, 100 steps:

| Metric | Value |
|---|---:|
| Validation loss | 2.223613 |
| Test loss | 2.051383 |
| Renamed loss | 2.147160 |
| Training seconds | 2.140126 |

No full training ran. Smoke results use cleaned temporary storage, matching iter26.

Implementation note: exact first-step comparison enables deterministic operations only during that check; PyTorch emits CuBLAS warnings. Normal training settings remain unchanged. No model-spec deviations.