# iter38 storage implementation

Created `iter38.py` by copying `iter37.py`. Existing files were not edited. Read
`README.md`, `iter37.py`, and the four requested reference result files.

The new implementation provides:

- `--storage tagged`: original int32/int32 direct-map semantics, including the
  original maximum-fingerprint chunk election. Coalesced writes provide exact
  eviction counts. The original `Tagged` remains as an independent check oracle.
- `--storage int32`: int32/int32 associative storage, to ablate compression while
  keeping the bucket algorithm.
- `--storage compact`: uint8 count plus int16 fingerprint, exactly three entry
  bytes. Counts are exact through 32, then use base-1.06 decoded levels and
  stateless, hash-randomized rounding between neighboring levels. Batch increments
  are unbiased before saturation. The last decoded level exceeds the 6.2M-target
  training pass. The mixer receives decoded float counts through the existing
  `table.read()` interface.
- Associative tables default to four ways. Hits are updated first; distinct misses
  replace the lowest encoded count, with lowest-way tie breaking. Chunk writes
  coalesce duplicate identities and process one miss per bucket per round, in
  descending fingerprint order. At most `ways` distinct misses per bucket are
  processed in a chunk; excess collision writes are explicitly counted as dropped.
  This defines deterministic bounded chunk collision handling rather than racing
  scatter assignments. Aging halves counts every 256 chunks, in bounded slabs.
- `--admit doorkeeper`: two-probe, bit-packed Bloom filter, two bits per sparse
  slot. First occurrences set bits; repeat occurrences can insert. Exact duplicates
  later in the same chunk can insert. Other Bloom tests use pre-chunk bits, so
  different new pairs do not admit one another through within-chunk false positives.
  Resident entries always update, including after a Bloom reset. Filters reset
  every 256 chunks. All sparse raw orders (>=3), structural tables, and sparse
  learned-key tables use admission; raw1/raw2 and other dense tables stay exact.
- `--budget-mib B`: a total persistent budget, including dense tables, mixer,
  filters, sampled usefulness, diagnostic tensors, lookup LUTs, character indices,
  and direct-map shared scratch. CUDA transient allocations are measured separately
  using the baseline peak accounting. Compact/associative tables need no
  full-table write scratch.

The budget implementation uses the permitted **one-shot allocation alternative**.
An initially small probe processes the first ceil(10%) targets in the original
shuffled training order. In tables mode it probes storage only: mixer updates
cannot affect the contexts, so disposable mixer training is omitted. Keys mode
also trains the disposable keys/mixer to obtain their evolving addresses.
Each sparse table samples context reads and one pseudo-random physical slot in
each group of 64. A live sampled occupant is useful if it has matched a read at
least once; its usefulness resets on replacement. Allocation weights are these
estimated useful entry counts, with weight one for a zero estimate. The allocator
doubles power-of-two table sizes in proportion to these weights while accounting
for every persistent byte. Rounding slack and the probe's measurements/sizes are
reported. The probe is discarded, RNG state restored, and the original single
training pass starts empty. No retained full hashes, online growth, or fingerprint
rehash is claimed.

Only storage, storage verification, requested diagnostics/timing, CLI, and result
reporting were changed. Static source comparisons found these regions identical
to iter37: Adam/LocalMixer/KeysNet; Model contexts, score, logprobs, learn_write and
memory; epoch_cpu; semantic staleness calculation; FLOP estimate; original checks.
Data partitioning, stream order, score-before-write, smoothing, copy heads,
manual mixer rule, evaluation functions/windows, and ten validation milestones
remain the same. No autograd or optimizer was introduced.

## Flags

| Flag | Default | Purpose |
|---|---|---|
| `--storage` | `tagged` | `tagged`, `int32`, or `compact` |
| `--admit` | `none` | `none` or `doorkeeper` |
| `--budget-mib` | omitted | Total persistent budget; omitted retains CAP-based sizes |
| `--ways` | 4 | Associative ways: 1, 2, 4, 8, 16; tagged uses one |
| `--age-chunks` | 256 | Associative count-halving interval; 0 disables; tagged ignores |
| `--door-reset-chunks` | 256 | Bloom reset interval |
| `--diag-every` | 16 | Sample every Nth context read |
| `--stale-every` | 64 | Sample semantic staleness every Nth chunk, outside train timer |
| `--smoke [N]` | omitted | Prefix N targets; bare flag means 16384; 16 eval windows |
| `--self-test` | off | Storage-only checks, with no dataset loading/training |
| `--check-device` | `cpu` | `cpu` or `cuda` for standalone storage checks |

`--mode`, `--chunk-size`, `--seed`, and environment `CAP` remain available.
Without a budget, CAP changes the same raw/structural capacities as iter37.
The unchanged `keys` mode is retained, but the requested experiment is `tables`.

## Results and diagnostics

Outputs use mode `"w"`, at `results/iter38_{tag}.json` or
`results/iter38_{tag}_smoke.json`. The tag includes mode, storage, admission,
budget, ways, aging, reset/sampling intervals, CAP, chunk size, and seed.
Repeated identical configurations overwrite their own iter38 result. Smoke never
overwrites a full result. For the requested default smoke command, the filename is:

`results/iter38_tables_compact_doorkeeper_budget64_ways4_age256_door256_diag16_stale64_cap0_chunk4096_seed0_smoke.json`

JSON includes the original metrics and val_curve, the exact requested references,
per-table persistent bytes, sampled pair hit rate, exact evictions, fraction of
write events passing admission, final occupancy, aging removals, dropped chunk
collision writes, and sampled live useful-entry estimates. Occupancy scans run
after training. Read diagnostics are frozen during every evaluation and guarded
by the original state-stamp checks. Dense-table diagnostics are included too.

The old per-chunk semantic `unique()` diagnostic now runs on every 64th chunk
with synchronized timing boundaries. Storage coalescing and bit updates still
use sort/unique_consecutive as part of their actual write algorithms; these are
not diagnostic unique scans. Staleness fractions use sampled targets as their
denominator. Timing fields distinguish:

- `train_seconds`: original main-pass work, including sampled storage diagnostics,
  excluding evaluations, setup/checks, allocation probe, and semantic staleness.
- `train_seconds_with_stale_diagnostics`: main pass plus sampled semantic staleness.
- `allocation.warmup_seconds` and `allocation.allocation_seconds`: probe work and
  total allocation work respectively; allocation_seconds includes warmup_seconds.
- `train_seconds_with_allocation`: main pass plus total allocation work. The time
  pass/fail bar uses this conservative total, so the probe cannot hide its cost.
- `train_seconds_with_all_diagnostics_and_allocation`: also includes sampled
  semantic staleness and the final diagnostic scans.

Training peak includes allocation/probe peaks. Evaluation peak retains separate
CUDA max-allocated accounting. The full-run targets are test <=1.185, renamed
<=1.550, train peak <=151 MiB, and time including allocation <=66.8699086 s.
Smoke runs cannot pass the full-run bars. No achieved quality/time/memory claim
is made without a full run.

## Verification status and deviations

The supplied Python executable could not start in this sandbox: PowerShell
reported `Program 'python.exe' failed to run: Access is denied`. Per the request,
Python unit checks and smoke training were skipped. No full training was run;
no iter38 metrics JSON was fabricated. Source comparison of the unchanged regions
passed. `git diff --no-index --check` found no whitespace errors (its nonzero exit
status also reports that the two files differ).
A separate PowerShell calculation of the codec formula gave 2.908% maximum
nearest-level error over 1..10000 and a maximum decoded count of about 7.329M.
This checks the level design, not the unexecuted Python/Torch implementation.

Embedded checks cover count-level monotonicity and <=10% quantization error at
every integer 1..10000; actual sequential increments on 512 independent counters
with <=10% ensemble-mean error at every prefix; saturation; dict-reference streams
for all three stores with/without admission and without evictions; within-chunk
duplicate admission; packed bit updates; resident updates after resets; frequency
eviction; aging; exact direct-map equivalence to iter37; budget formulas; exact
storage/output equality across two runs; configured-model causality and frozen
evaluation. The original learner/causality checks are retained. Full model output
equality is checked, but has not been executed; the original GPU mixer operations
are preserved, including their floating-point reduction behavior.

The 10% count accuracy requirement needs a statistical interpretation for Morris
increments: an individual random counter path cannot guarantee <=10% error.
Deterministic nearest-level decoding and the ensemble expectation are checked;
individual counter paths have stochastic error. This is an explicit limitation,
not a claim that every sampled path satisfies a uniform error bound. The other
intentional deviations are the allowed one-shot probe/restart allocation instead
of online rehash, and the documented bounded miss election within a chunk.

## Commands to run outside this sandbox

PowerShell, from the repository root:

```powershell
$py = 'C:/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe'
$env:CAP = '0'

# Storage checks, with no training or data loading.
& $py -X utf8 -B iter38.py --self-test
& $py -X utf8 -B iter38.py --self-test --check-device cuda

# Requested smoke; bare --smoke means 16384 targets.
& $py -X utf8 -B iter38.py --mode tables --smoke --storage compact --admit doorkeeper --budget-mib 64
```

Full ablation commands below are provided only; **they were not executed**:

```powershell
$py = 'C:/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe'
$env:CAP = '0'

# Original direct-map behavior, original sizes, sampled/separately timed diagnostics.
& $py -X utf8 -B iter38.py --mode tables --storage tagged --admit none --age-chunks 0

# Equal-budget factorial: compression/associativity and admission.
foreach ($store in @('tagged', 'int32', 'compact')) {
    foreach ($gate in @('none', 'doorkeeper')) {
        & $py -X utf8 -B iter38.py --mode tables --storage $store --admit $gate --budget-mib 64 --ways 4 --age-chunks 256
    }
}

# Associativity, keeping compact counts and admission fixed (4-way is above).
foreach ($way in @(1, 2, 8)) {
    & $py -X utf8 -B iter38.py --mode tables --storage compact --admit doorkeeper --budget-mib 64 --ways $way --age-chunks 256
}

# Aging ablation, with and without admission.
foreach ($gate in @('none', 'doorkeeper')) {
    & $py -X utf8 -B iter38.py --mode tables --storage compact --admit $gate --budget-mib 64 --ways 4 --age-chunks 0
}

# Allocation/budget ablation: same CAP sizes without allocation, then other budgets.
& $py -X utf8 -B iter38.py --mode tables --storage compact --admit doorkeeper --ways 4 --age-chunks 256
foreach ($budget in @(32, 96)) {
    & $py -X utf8 -B iter38.py --mode tables --storage compact --admit doorkeeper --budget-mib $budget --ways 4 --age-chunks 256
}
```

Run the requested smoke twice to compare end-to-end scalar metrics and validation
values, excluding timings/peaks. Both calls also run the embedded exact two-run
configured-output and storage checks. The second invocation overwrites only its
own smoke result, as requested:

```powershell
@'
import iter38 as m
o = m.StorageOptions(storage="compact", admit="doorkeeper", budget_mib=64)
a = m.run("tables", 16384, options=o)
b = m.run("tables", 16384, options=o)
names = ("val", "test", "renamed", "test_nocopy", "diagnostics", "memory_bytes")
assert all(a[k] == b[k] for k in names)
assert [v["val_subset"] for v in a["val_curve"]] == [v["val_subset"] for v in b["val_curve"]]
print("two smoke runs match")
'@ | & $py -X utf8 -B -
```
