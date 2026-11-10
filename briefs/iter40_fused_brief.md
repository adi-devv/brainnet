# Build iter40.py: fused sparse tables for speed (no backprop)

Repo: this folder. Read `README.md` (iter38-39 findings), `iter39.py`, `briefs/iter39_codex_out.md`, `briefs/iter39_flaky_out.md`, and `prof39.py`. Create `iter40.py` as a copy of `iter39.py` and change ONLY what is needed for speed. Do NOT edit existing files. Do NOT run full training.

## Status (compact, --admit none, --age-chunks 0, 64 MiB, seed 0)
| run | test | renamed | train s | peak |
|---|---|---|---|---|
| iter39 | 1.1713 | 1.5450 | 221.9 | 117 MiB |
| BP reference | 1.256 | 1.571 | 79.4 | 151 MiB |
| target | <=1.185 | <=1.550 | <=67 | <=151 MiB |

Smoke profile (409,600 targets, synchronized timers): train 10.6 s. `Buckets.write` 8.9 s and `Buckets.read` 3.3 s, compared with 7.6 s total for tagged 1x. Both are launch-bound: 8 sparse tables, each doing its own small kernels per chunk; GPU utilization ~30%. The full iter39 run (221.9 s) is slower than iter38 (193 s). Check whether `sum_rows` (sort + segment_reduce in the mixer gradients) caused that, and make it cheaper while keeping it bit-deterministic.

## Changes
1. **One flat store:** concatenate all sparse tables' count/tag arrays into one storage, with per-table bucket offsets (keep each table's slot count). Hash each table's keys as before, add the offset, and do ONE gather for all tables' reads per chunk, and ONE write (hit coalescing, miss ranking, victim election, scatter) over the concatenated records. Since the offsets make tables disjoint, the result must be bit-exact with iter39 per table.
2. **Fewer launches:** no per-table Python loops in score/learn_write for sparse tables. Encode/decode, hashing and the unsafe-bucket TorchScript recurrence each run once per chunk over all tables. Keep stats on device.
3. **Optional, only if exact:** CUDA graphs or `torch.compile` for the fixed-shape chunk path. Skip it if it complicates exactness.
4. Keep everything else identical: data, splits, scoring before writes, mixer (deterministic), evals, metrics, timing scope, memory accounting, allocation probe, diagnostics. The train peak stays <=151 MiB.

## Verify
- `--self-test`: iter40's tables are bit-identical to iter39's after a random stream with heavy collisions (CPU and CUDA), plus all inherited self-tests and configured_checks.
- Smoke command (CAP=0): `/c/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B iter40.py --mode tables --smoke 409600 --storage compact --budget-mib 64 --admit none --age-chunks 0`

If your sandbox can't run Python, write everything completely and skip running. Report in `briefs/iter40_codex_out.md`: summary, what changed, the exactness argument, the expected speedup, and the exact full-run command.
