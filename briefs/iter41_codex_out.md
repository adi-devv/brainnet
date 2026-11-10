# iter41: bounded read workspaces; execution blocked

Implemented `iter41.py` as a copy of iter40. `iter40.py` was not edited.
The required Python executable cannot start in this sandbox: **Access is denied**.
Self-test/smoke exit 0, full-run equality, resource targets, and profile rankings
remain unverified. No training results or profile measurements were fabricated.

## Changes

- Fused reads process at most 512 targets at once. Hash/gather scratch scales
  with 512 targets rather than 4096; the returned counts cover the original
  chunk. Writes still occur once per original chunk.
- Fixed-size CUDA read blocks replay a read-only graph with reused inputs,
  outputs, and private-pool scratch. Short blocks use eager execution.
  Warmup/capture perform no table, diagnostic, or RNG writes. Caller-owned
  results survive subsequent replay. `ITER41_GRAPHS=0` enables the eager control.
- Diagnostics retain the original global `::diag_every` sampling positions,
  including strides that do not divide 512. Integer hit sums and usefulness
  marking run once per original read outside graph capture.
- Tables-mode scoring fills one final packed count buffer directly, avoiding
  individual outputs plus a second packed copy. Keys-mode packing is retained.
- Backoff overwrites disposable counts after each original row sum/candidate
  calculation, removing the separate proposal allocation. Expressions, row
  order, reductions, and separate multiply/add arithmetic are unchanged.
  Optimized TorchScript execution remains disabled for the recurrence.
- Both mixer row reductions replay together in a fixed-shape CUDA graph for
  4096-target updates. `sum_rows` is source-identical to iter40: stable sort,
  integer lengths, and deterministic segmented summation.
- Execution workspaces use external weak-key caches. They do not mutate frozen
  predictor state, and are released with the store/mixer, including the probe.
  Their GPU allocations ARE counted in CUDA peak allocated memory. They are
  not free memory and have not been demonstrated to fit the target.
- Character accounting uses validity counts from the existing CPU batch,
  handling the partial tail wherever it appears in the shuffled order. Fully
  valid chunks skip boolean packing of targets and codes. Masked chunks keep
  the original filtering/order. Probe and main-pass per-chunk scalar reads
  were removed. Dynamic-size operations inside fused writes remain.
- Fused writes, rounding identities/nonces, admission, aging, planner, and
  optimizer arithmetic are unchanged. The result prefix is iter41, the time
  bar is 67 seconds including allocation, and the exact full configuration
  gains a strict metric equality bar included in `all_targets_pass`.

Neither <=151 MiB nor <=125 MiB is claimed. Graph compatibility, actual private
pool footprint, speed, and bit-exact behavior require execution on the target.

## Verification added

All inherited checks remain. New `bounded_execution_checks` compares actual
iter40 and iter41 implementations at lengths 0/1/511/512/513/1023/1024/4096,
eight tables, and diagnostic strides 1/3/16/1000. It checks scores, read counts,
diagnostic/usefulness state, frozen reads, and nonaliased outputs.

It also compares three 4096-target training steps in tables and keys modes:
all scores, expert log probabilities, counts, tags, diagnostics, usefulness,
unigram counts, and nested optimizer parameters/moments. Full and masked writes
include an invalid tail in the middle of a batch. CUDA-only checks compare
CPU-derived target counts with device masks for chunk sizes 1/37/128/512/4096.
Graph-cache release is checked after owners are discarded. These Torch checks
could not execute; they are not claimed as passed.

## Numbers and failures

| Measurement | supplied iter40 | iter41 |
|---|---:|---|
| val | 1.2769815154075623 | unavailable |
| test | 1.1712950946092606 | unavailable |
| renamed | 1.5450188283920288 | unavailable |
| train seconds | 67.45867439999347 | unavailable |
| seconds including allocation | 73.04998769999293 | unavailable |
| peak allocated training MiB | 298.6455078125 | unavailable |

Six commands were attempted sequentially with Python312, `-X utf8 -B`, CAP=0:
before profile, CPU self-test, CUDA self-test, 409600-target smoke, after
profile, and one full-run command with the exact requested configuration.
All failed BEFORE startup with Access is denied. Exit codes are null because
Python never started. The orchestration command exited 1.

`results/iter41_execution_attempts.json` records exact arguments, timestamps,
errors, and individual log paths. The GTX 1650 was idle when inspected
(0 MiB used, 0% utilization). No concurrent GPU job was launched.

## Profile top-10 before and after

| Profile | Operator top-10 | Region top-10 |
|---|---|---|
| iter40 before | unavailable: executable launch denied | unavailable |
| iter41 after | unavailable: executable launch denied | unavailable |

`prof41.py` runs the ordinary allocation probe, trains 10 warmup chunks, and
profiles 100 real 4096-target chunks using CPU and CUDA activities. Both
modules use the same seed, data, configuration, sample length, and allocation
procedure. Output contains operator top-10 by self device/self CPU time and
region top-10 by total device/CPU time. Nested regions must not be added.
Ordinary full-run timing excludes profiler overhead.

Regions cover contexts, fused reads/writes, backoff, deterministic row sums
and replay, mixer forward/gradients, Adam, and copy heads. CUDA graph kernel
attribution depends on installed PyTorch/CUPTI; replay has explicit host
regions. Read scratch/launch overhead, row-reducer launches, and backoff
allocation are the three implementation candidates addressed. They have NOT
been established as the measured top three costs.

On success, profiles write `results/iter40_profile_top10.json` and `.txt`, and
`results/iter41_profile_top10.json` and `.txt`. None were produced here.

## Checks actually completed

`powershell -NoProfile -ExecutionPolicy Bypass -File briefs/iter41_static_check.ps1`
exited 0. See `results/iter41_static_checks.json`.

- Compared all 42 inherited top-level source definitions. Changes are confined
  to FusedBuckets, table_proposals, LocalMixer, Model, epoch_cpu, prepare_model,
  self_tests, and run. The complete fused write method is source-identical.
- Checked 14,365 scalar diagnostic-sampling cases against global sampling,
  including empty inputs, partial blocks, and large strides.
- Parsed the PowerShell runner successfully.
- `git diff --no-index --check -- iter40.py iter41.py` passed.

These are source/scalar checks, not Python parsing, Torch/JIT execution, CUDA
exactness, memory measurements, or speed measurements. Python syntax could
not be checked using the required blocked executable.

Unchanged iter40 SHA-256:
`311af3f0c61033bf441badbc739decbc57a83a24e48ecc850ec05cc7412e6ec0`.

## Run on the permitted installation

`run_iter41.ps1` sequentially runs both profiles, both self-tests, smoke, and
one full run, saves logs, then checks exact metric equality and the 151 MiB /
67 second bars. Capture/setup stays inside ordinary allocation/train timing.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_iter41.ps1
```

Exact full-run command in Git Bash form:

```bash
CAP=0 /c/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B iter41.py --mode tables --storage compact --budget-mib 64 --admit none --age-chunks 0 --ways 4 --chunk-size 4096 --seed 0
```

Expected full-run output, not produced here:
`results/iter41_tables_compact_none_budget64_ways4_age0_door256_diag16_stale64_cap0_chunk4096_seed0.json`.
