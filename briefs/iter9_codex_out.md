Created `iter9.py` (128 lines) with recency memory, copy prediction, both ablations, frozen-state checks, and decision rules.

Smoke passed for all three variants: 128 training chars and 2 evaluation windows each, 19 seconds total. DP/recency references, causality, padding, distributions, and frozen-state checks passed.

Saved `results/iter9_{all,no_recency,no_copy}.json`. No existing files changed.