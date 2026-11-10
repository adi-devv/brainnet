# Build iter39.py: fast associative write, no doorkeeper (no backprop)

Repo: this folder. Read `README.md`, `iter38.py`, `briefs/iter38_storage_brief.md`, `briefs/iter38_codex_out.md`, and `results/iter38_tables_compact_*_seed0.json`. Create `iter39.py` as a copy of `iter38.py`, then change ONLY what is listed below. Do NOT edit existing files. Do NOT run full training; smoke runs only.

## Measured root causes (iter38, compact storage, 64 MiB, seed 0)
| run | test | renamed | train s |
|---|---|---|---|
| doorkeeper | 1.2003 | 1.5585 | 209.6 |
| no admission | 1.1867 | 1.5605 | 194.4 |
| target (16x tables) | <=1.185 | <=1.550 | <=~67 |
| tagged 1x (iter35) | 1.2225 | 1.5878 | 66.9 |

- **Quality:** the doorkeeper drops 36-72% of writes for orders >=8. Tables are only 2-30% occupied, with almost no evictions, so admission throws away information without saving any space. Aging (halving every 256 chunks) also deletes 40k-570k live entries per table. **No-aging run (`--admit none`, aging off, 64 MiB): test 1.1713, renamed 1.5450, peak 116.7 MiB, 193 s. That meets the quality target, so make aging OFF the default in iter39 (keep the flag for ablation). Only speed remains.**
- **Time:** on a 409,600-target smoke with synchronized timers, `Buckets.write` takes 17.6 s of 18.9 s train (the doorkeeper is only 2.2 s of that). Tagged's write takes 3.9 s of 7.6 s. `Buckets.write` is launch-bound: Python loops over the ways (resident check and hit check, 4 each), a 4-turn miss-election loop, and `put_count` calling `LogCount.encode` plus `uniform_hash`. That's ~150 small kernels per table per chunk, across 8 sparse tables. GPU utilization is ~32%.

## Changes
1. **Default `--admit none`.** Keep the doorkeeper flag for ablation only; remove the resident check when admission is off (iter38 already skips it, so just verify).
2. **Vectorized `Buckets.write`, same semantics as iter38 (bit-exact with `--admit none` on a fixed stream):**
   - Gather all W ways at once: `ix = bucket[:,None]*W + arange(W)`, compare tags and counts in one op. No per-way Python loop.
   - Resident hits: one gather, one encode, one scatter.
   - Misses: compute each miss's rank within its bucket (already done) and the victim ordering of the bucket's ways (ascending encoded count, tie by lowest way) via one `sort`/`argsort` over `[n_bucket_misses, W]`. Then assign victim = ordered_way[rank] for rank < W, all in ONE scatter. Match iter38's sequential turn-by-turn result exactly. Note that in iter38 a later turn sees counts written by earlier turns. Reproduce that ordering, or prove the batched assignment is identical (each new entry gets a distinct victim way, chosen in the order of lowest old count). Rank-r miss takes the r-th lowest old way. If an earlier turn's newly written entry could become the argmin for a later turn, document whether that differs, and make it match.
   - `LogCount.encode`: call it ONCE per write over the concatenated (hit, miss) values, not per turn.
   - Aging: one vectorized pass over the whole table (or a few large slabs), not 65,536-slot slabs, if it fits in the memory budget. Keep the train peak <=151 MiB.
3. **Fewer launches elsewhere where it's free:** avoid `.sum()` stats syncs per write (keep stats on device and add tensors; no `.item()`). Optionally batch all sparse tables' writes into one call by concatenating with per-table offsets into one flat storage. Do it only if it stays simple and exact.
4. Keep everything else from iter38 identical: data, splits, scoring before writes, mixer, evals, metrics, timing scope, memory accounting, allocation probe, and diagnostics (still sampled).

## Verify
- Bit-exact: iter39's associative write with `--admit none` produces tables identical to iter38's `Buckets.write` on a random stream with heavy collisions and evictions (CPU and CUDA). Add this to `--self-test`.
- Self-tests from iter38 still pass.
- Smoke: `/c/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B iter39.py --mode tables --smoke 409600 --storage compact --budget-mib 64` (with `CAP=0`). Report train_seconds against iter38's 18.9 s and tagged's 7.6 s on the same smoke.

If your sandbox can't run Python, write everything completely and skip running. Report in `briefs/iter39_codex_out.md`: summary, what changed, the exactness argument, and the exact full-run commands.
