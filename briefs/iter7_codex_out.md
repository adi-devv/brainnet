Created `iter7.py` (156 lines) with both variants, validation selection, JSON output, and frozen-state assertions. Existing files unchanged.

Smoke passed both variants; production D=8192 passed all nine Cholesky solves. Peak GPU allocation: 2063 MiB. Total smoke time stayed under 60 seconds.

Stage 0 smoke loss: 4.066634 at 128 chars. Full-pass reference check (1.2528 ± 0.005) is implemented but unrun, as requested.