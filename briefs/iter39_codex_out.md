# iter39: associative write batching

Created `iter39.py` as a copy of `iter38.py`. Existing files were not edited.
Read the README, iter38 implementation and briefs, and all four requested
compact full-run seed-0 JSONs, including the no-aging result (test 1.1712951,
renamed 1.5450188, train 193.3451 s, peak 116.6709 MiB).

## Changes

- Admission remains `none` by default in both `StorageOptions` and the CLI.
  Without admission, the write path skips the resident admission scan, Boolean
  admission mask/filter, and admission reduction. It increments write/admission
  counters together on device. The doorkeeper remains available for ablation;
  its resident scan now gathers all ways at once.
- Aging now defaults to **0 (off)** in both the dataclass and CLI. The flag is
  retained. Aging uses 524,288-slot slabs instead of 65,536-slot slabs and
  precomputes the 256 half-count rounding thresholds once per aging pass.
  Per-slot hash identity, write nonce, salt, and stochastic rounding are retained.
- Coalesced writes gather all ways together. Resident hits use one batched count
  update. Accepted hit and miss values are concatenated for exactly one
  `LogCount.encode` and one `uniform_hash` call per compact write, excluding a
  separately triggered aging pass. Misses beyond rank W-1 remain dropped.
- Miss buckets gather counts after hit updates and sort ways once by encoded
  count, stably breaking ties by way. Safe buckets use rank-indexed old ways.
  Buckets that might re-evict a new entry use a scripted local tensor recurrence.
  It performs no global table writes, encoding, hashing, or Python per-turn
  dispatch. An empty correction set returns immediately.
- A final rank election retains only the last writer to each physical victim.
  Miss tags/counts each scatter once with unique destinations. Eviction and
  dropped-write statistics stay on device; no `.item()` or scalar conversion
  was added to the write path. Sparse tables remain separate.
- Results use the `iter39_` prefix, preserving the complete iter38 schema,
  timing scope, allocation probe, diagnostics, memory accounting, and metrics.

## Exactness argument and ordering caveat

Simply assigning miss rank r to the r-th lowest old way is **not** equivalent
to iter38. In a full bucket with counts `[20,21,22,23]`, four one-count misses
all replace way 0 in iter38. Distinct old-victim assignment would replace all
four ways. Keeping exactness therefore requires sequential dependencies for
some buckets. The correction retains that recurrence in TorchScript; it is
not a claim to eliminate every dependent kernel on collision-heavy buckets.
Speed still requires measurement.

After the unchanged descending-identity coalescing and resident updates, let
ways be sorted by `(encoded_count, way)`. For a bucket with k accepted misses,
the fast path requires each of its first k-1 insertions, paired with its
predicted old victim way, to sort at or after old victim k-1. By induction, no
new entry can then precede any remaining old victim. Rank r consequently takes
sorted old way r. The code marks a bucket unsafe if this sufficient condition
fails. Unsafe buckets simulate iter38's argmin/replacement on local counts,
including ties by lowest way and replacement of earlier insertions.

Encoding can be moved before this recurrence because each hit value is fixed
by its old count and coalesced increment, each miss value is its multiplicity,
and the stateless random value depends only on identity/write number. Neither
victim choice nor turn enters encoding. The highest-way resident match is
preserved. Victim selection reads encoded counts after resident updates.

All inserted miss counts are positive. Thus the number of evictions in a bucket
is `k - min(k, initially_empty_ways)`, even when the same way is replaced
repeatedly. Clearing usefulness for the final set of victim slots has the same
effect as clearing each intermediate victim: there are no intervening reads.
Keeping only the final writer gives the same tags/count bytes without scatter
races. Aging's LUT uses exactly the original low level and rounding probability
for each of the 256 possible input codes; slab boundaries do not enter hashing.

## Verification and smoke status

- Python AST parsing passed. Source comparison confirms that only
  `StorageOptions`, `Buckets`, the storage-check report, and the run output
  filename changed among existing definitions. All original storage checks
  remain present. Data, scoring, mixer, allocation, evaluations, timing, memory,
  and diagnostic sampling functions are unchanged.
- A dependency-free scalar exercise passed **83,420** exhaustive/random cases
  across ways 1, 2, 3, 4, 8, and 16. It checked the sufficient fast-path condition,
  exact correction, final-writer election, and eviction-count formula, including
  49,880 cases with repeated victims. This validates the ordering argument;
  it does not substitute for running the Torch implementation.
- `--self-test` now compares actual tables, stats, usefulness, and reads against
  the unchanged `iter38.Buckets.write` oracle after every chunk: compact/int32,
  ways 1/2/4/8/16, aging off/on, empty chunks, duplicate writes, heavy collisions,
  evictions, log counts, and sampled reads. It also includes the explicit
  repeated-victim counterexample and all 256 count codes across multiple large
  aging slabs. Run on CPU and CUDA with the commands below.
- Both self-test commands and the requested 409,600-target smoke were attempted.
  The supplied Python 3.12 executable could not start: **Access is denied**.
  Accessible Conda Python 3.13 and 3.11 installations lack PyTorch. Consequently,
  Torch/JIT compilation, CPU/CUDA exactness, the inherited tests, smoke timing,
  and the <=151 MiB train peak remain unverified. No training ran and no results
  JSON was fabricated.
- `git diff --no-index --check` reported no whitespace errors.

| Same 409,600-target smoke | train_seconds |
|---|---:|
| iter38 (supplied synchronized baseline) | 18.9 s |
| Tagged (supplied baseline) | 7.6 s |
| iter39 | Unavailable: Python executable blocked |

The 64 MiB persistent budget/accounting is unchanged. Write scratch scales with
chunk size and associativity, not table capacity. Aging scratch is bounded by
the larger slab. Actual peak and speed targets are not claimed without CUDA
measurement.

## Exact verification commands

PowerShell, from the repository root:

```powershell
$py = 'C:/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe'
$env:CAP = '0'
& $py -X utf8 -B iter39.py --self-test
& $py -X utf8 -B iter39.py --self-test --check-device cuda
& $py -X utf8 -B iter39.py --mode tables --smoke 409600 --storage compact --budget-mib 64
```

Equivalent requested smoke command in Git Bash:

```bash
CAP=0 /c/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B iter39.py --mode tables --smoke 409600 --storage compact --budget-mib 64
```

Default smoke output:
`results/iter39_tables_compact_none_budget64_ways4_age0_door256_diag16_stale64_cap0_chunk4096_seed0_smoke.json`.

## Exact full-run commands (provided only; not executed)

```powershell
$py = 'C:/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe'
$env:CAP = '0'

# Main configuration: no admission, no aging.
& $py -X utf8 -B iter39.py --mode tables --storage compact --budget-mib 64 --admit none --ways 4 --age-chunks 0 --seed 0

# Admission and aging ablations, same 64 MiB budget.
& $py -X utf8 -B iter39.py --mode tables --storage compact --budget-mib 64 --admit doorkeeper --ways 4 --age-chunks 0 --seed 0
& $py -X utf8 -B iter39.py --mode tables --storage compact --budget-mib 64 --admit none --ways 4 --age-chunks 256 --seed 0
& $py -X utf8 -B iter39.py --mode tables --storage compact --budget-mib 64 --admit doorkeeper --ways 4 --age-chunks 256 --seed 0

# Original-size Tagged control, using the unchanged direct-map writer.
& $py -X utf8 -B iter39.py --mode tables --storage tagged --admit none --age-chunks 0 --seed 0
```

Main full-run output:
`results/iter39_tables_compact_none_budget64_ways4_age0_door256_diag16_stale64_cap0_chunk4096_seed0.json`.
