# Build iter38.py: memory-efficient, growable count storage (no backprop)

Repo: this folder. Read `README.md`, `iter37.py` (iter35 tables plus a `CAP` env knob), `results/iter37_tables_cap{2,4}.json`, `results/iter35_tables.json`, and `results/iter30_backprop.json`. Create `iter38.py` as a copy of `iter37.py`, then change ONLY the storage. Do NOT edit existing files. Do NOT run full training; smoke runs only.

## Why
The tables plateau because they fill up, not because the method stops working:

| tables | test | renamed | persistent memory |
|---|---|---|---|
| 1x (iter35) | 1.2225 | 1.5878 | ~52 MB (train peak 103 MiB) |
| 4x | 1.1886 | 1.5553 | train peak 240 MiB |
| 16x | 1.1768 | 1.5441 | 776 MB (train peak 792 MiB) |
| BP iter30 | 1.2562 | 1.5714 | train peak 151 MiB |

Each `Tagged` slot is direct-mapped and costs 8 bytes (int32 count + int32 tag). Every write replaces the slot, so most of the space goes to (context, next-char) pairs seen only once. Goal: **reach 16x quality (test ≤1.185, renamed ≤1.550) with train peak ≤151 MiB**, at a train time ≤ iter35's (~67 s), with no BP.

## Changes, each behind a flag so we can ablate (`--storage`, `--admit`, `--budget-mib`)
1. **Compact entries (`--storage compact`).** Use a uint8 saturating count with probabilistic (Morris/log) increments above a threshold, plus an int16 fingerprint: 3 bytes per entry instead of 8. Return the decoded count estimates to the mixer exactly where raw counts went before.
2. **Set-associative buckets with frequency eviction.** Use 4-way (configurable) buckets instead of direct-mapped. On a miss, replace the lowest-count way, and decay or age counts so stale entries can leave. Keep the deterministic collision handling within a chunk (same idea as `Tagged.commit`: no nondeterministic scatter races).
3. **Admission (`--admit doorkeeper`).** A small bit-array Bloom "doorkeeper" per table (TinyLFU style): the first sighting of a (context, next) pair only sets bits, and the second sighting admits it into the table. Reset the doorkeeper periodically. Apply it to orders ≥3; raw1/raw2 stay dense.
4. **Budget allocation / growth (`--budget-mib B`).** Given a total persistent budget, size each table's slots. Allocation is proportional to its measured useful entries, meaning entries that got hit (read with a matching fingerprint) at least once. Grow by doubling with a rehash while under budget: store enough hash bits to rehash, or recompute the address from the stored full key hash. A one-shot allocation from a warm-up first 10% is also acceptable if growth is too complex; document what you did.
5. Keep everything else from iter37 identical: data, splits, scoring before writes, mixer, evals, metrics, timing scope, and memory accounting.

## Diagnostics per table (cheap, sampled; NOT a per-chunk `unique` inside train timing)
Hit rate, eviction count, admitted fraction, final occupancy, and bytes. Also move iter35's per-chunk staleness `unique` diagnostic out of the train timing (sample it every N chunks, or time it separately), and report train_seconds both ways.

## Output
`results/iter38_{tag}.json`, where tag encodes the flags. Write it with mode `"w"`; smoke runs write `*_smoke.json`. Include val/test/renamed, the val_curve at 10%…100%, train_seconds, train peak MiB, persistent bytes per table, the diagnostics, and `references=dict(bp_test=1.2562, bp_renamed=1.5714, bp_peak_mib=151, tables16x_test=1.1768, tables16x_renamed=1.5441)`.

## Verify
- Tiny-case unit checks: compact count decode is monotonic and within 10% of the true count up to 1e4; buckets and the doorkeeper match a dict reference on a small stream with no evictions; outputs are deterministic across two runs.
- Smoke: `/c/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B iter38.py --mode tables --smoke` (or whatever smoke flag iter37 uses), with `--storage compact --admit doorkeeper --budget-mib 64`.

If your sandbox can't run Python, write everything completely and skip running. Report in `briefs/iter38_codex_out.md`: summary, flags, exact commands to run the full ablation, and any deviation.
