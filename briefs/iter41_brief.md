# iter41: fit iter40 into the memory and time budget, staying bit-exact

Start from `iter40.py` (read `briefs/iter40_fused_brief.md` and `briefs/iter40_codex_out.md` first). Write `iter41.py`; do not edit iter40.py.

iter40 full run (seed 0, compact, --admit none, --age-chunks 0, budget 64, ways 4, chunk 4096):
val 1.2769815154, test 1.1712950946, renamed 1.5450188284, train_seconds 67.46, train_seconds_with_allocation 73.05, peak_allocated_mib_train 298.6.

## Hard targets
1. val/test/renamed bit-identical to the numbers above (same config, seed 0).
2. peak_allocated_mib_train <= 151 MiB, and ideally <= 125. The cause is fused-read scratch across all 8 tables at once. Fix it by processing the fused gather/score/write in table groups or target sub-chunks with reused preallocated buffers, while keeping the arithmetic order identical so the result stays bit-exact.
3. train_seconds_with_allocation <= 67 s (the time target includes allocation). Options: preallocate and reuse all per-chunk buffers, remove host syncs (.item(), .cpu(), nonzero-size dependent shapes) from the hot loop, CUDA graphs on any fixed-shape segment, and profile `sum_rows` and the backoff recurrence (torch.profiler or cuda events) and attack the top 3 costs.
4. Keep the self-test (all determinism and causality checks) passing.

## Deliver
- `iter41.py`, self-test exit 0, and the smoke run exit 0.
- One full run with the iter40 config. Write the result JSON to results/ (same naming, iter41 prefix) and include the profile top-10 before and after.
- `briefs/iter41_codex_out.md`: what changed, the numbers, the profile, and anything that failed.
- Run python as `/c/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B` with CAP=0. GPU is a GTX 1650 (4 GB); run nothing else on it at the same time.
