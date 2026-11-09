Created `iter8.py` (112 lines) with both variants, causal fast memory, hit rates, and saved decision rules.
Smoke passed for both: 129 training chars, 4 evaluation windows, 5.57 seconds total.
Reference counts/selectors, exact causality, and frozen-state checks passed; peak allocation: 1,295 MiB.
Saved smoke-only JSONs under `results/iter8_*`. No full runs or existing-file changes.