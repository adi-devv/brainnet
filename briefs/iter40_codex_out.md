# iter40: fused associative storage

`iter40.py` retains iter39's predictor and introduces a flat associative store
for compact and int32 tables. Tagged storage remains the control. No full
training ran. Runtime correctness, speed, quality and the 151 MiB peak remain
unverified because the PyTorch-enabled Python executable is blocked here.

## Changes

- Count/tag arrays occupy one flat allocation each, with disjoint per-table
  bucket ranges and the original slot counts. Per-table objects expose disjoint
  views for diagnostics and persistent-memory reporting. Allocation happens
  directly into the flat buffers; there is no second full-sized table copy.
- All sparse contexts hash together. One broadcast bucket/way gather per array
  reads all sparse tables. Broadcast indices avoid materializing the full
  `[tables, targets, vocabulary, ways]` physical-index tensor. Four-way compact
  reads gather uint8 counts and int16 tags, then decode the matched maximum.
  Larger-way ablations retain iter39's max-within-four and ordered group sums.
- One write coalesces the concatenated records, updates hits, ranks misses,
  encodes, corrects unsafe buckets with one `sequential_victims` call, elects
  final writers, and scatters. Statistics use device-side integer additions.
  Bloom admission and aging also retain separate logical table identities.
- The score path gathers all table counts, then dispatches the ordered backoff
  recurrence once through TorchScript. Optimized TorchScript execution is
  disabled around this floating-point recurrence so a pointwise fuser cannot
  contract the original separate multiply/add operations. No sparse-table
  read/write loop remains in the associative score/learn path. CUDA graphs and
  `torch.compile` were omitted.
- `sum_rows` still stably sorts contribution indices and uses segmented sums.
  Segment lengths now come from a fixed-size integer histogram. This removes
  `unique_consecutive`'s variable-sized CUDA output and the final unique-row
  assignment. Empty segments produce zero; nonempty segments retain the same
  ordered values. The learned floating-point state must match iter39 exactly.
- The first-10% allocation probe uses the same fused reads/writes. The greedy
  planner, replay/restart, data, splits, masked-target semantics, local learner,
  diagnostics sampling, metrics, timing boundaries and allocation accounting
  retain their original definitions. Results use the `iter40_` prefix.

## Exactness argument

Let table t start at global bucket offset O_t. Its global coalescing key is
`((local_bucket + O_t) << tagbits) | unsigned_tag`. Disjoint bucket intervals
prevent cross-table coalescing. Within one table this adds a constant to every
key, preserving iter39's descending bucket/fingerprint order and each bucket's
miss ranks. Subtracting `O_t << tagbits` **before** stochastic-rounding hashing
recovers the exact original identity. All tables receive one write per chunk,
including empty chunks, so their original write nonces equal the shared nonce.

Resident matching still selects the highest matching way. Counts are encoded
once from the same old counts/multiplicities and local identity/nonce. Victim
selection reads counts after hit updates. The sufficient distinct-victim test,
scripted sequential correction (including re-eviction), and last-writer election
are unchanged. Different table ranges cannot share a physical victim.

Read fingerprint comparisons and max-per-four/group-add order are unchanged.
Usefulness sampling and aging hashes subtract the table's physical slot offset.
Bloom probes keep each table's original mask and use disjoint packed-bit ranges;
duplicates are admitted only within that table. Device integer reductions
preserve exact statistics. The ordered expert recurrence, mixer, and optimizer
arithmetic retain iter39's operation order. Fixed histogram segment lengths
describe the same stably ordered learner contributions as the original unique
row list, with additional empty segments between them.

## Verification

Passed without PyTorch:

- AST parsing and bytecode compilation; no duplicate top-level definitions.
- AST comparison with iter39: inherited changes are confined to `sum_rows`,
  `Model`, CPU-compatible check RNG guards, `prepare_model`, row/storage checks,
  configured checks, and the result filename in `run`. `Buckets`, dense/tagged
  stores, codec, hashes, mixer, keys network, Adam, data sampler, stale metrics,
  FLOP accounting and greedy planner remain unchanged.
- A dependency-free scalar exercise checked 4,000 fused streams, 365,277
  accepted misses and 86,285 repeated-victim buckets across ways 1/2/4/8/16
  and both tag widths. It checks global grouping/local identity recovery,
  highest-way hits, safe/corrected victim ordering and final-writer election.
  This checks the ordering argument, not execution of Torch kernels.
- Replaying the planner with the supplied full-run allocation weights and the
  extra 448-byte layout preserves **both** its probe and final slot allocations.
  Persistent size becomes 67,018,301 bytes, leaving 90,563 bytes below 64 MiB.
  This is persistent-storage arithmetic, not a measured CUDA working-set peak.
- `git diff --no-index --check -- iter39.py iter40.py` found no whitespace
  errors (Git only warned about the repository's LF/CRLF conversion setting).

The dependency-free source/allocation/scalar checks are reproducible with
`python -B briefs/iter40_static_check.py` and do not import PyTorch.

The executable self-tests now compare actual fused tables against independent
`iter39.Buckets` after every random chunk: compact/int32, 1/2/4/8/16 ways,
admission on/off, aging on/off, unequal sizes, empty chunks, collisions, dropped
misses, Bloom resets, signed tags, duplicate residents and repeated victims.
They also check all 256 count codes across table-crossing aging slabs. Model
comparisons check exact expert probabilities, scores, counts, diagnostics and
nested optimizer parameters/moments for tables/keys and both associative stores.
Persistent-memory reporting is compared with unique underlying storage bytes.
All inherited storage checks and model checks remain; `--self-test` additionally
runs configured checks for both modes and tagged/compact/int32 storage.

CPU self-test, CUDA self-test and the exact requested 409,600-target smoke were
attempted with Python 3.12. Each was rejected before startup with **Access is
denied**. Accessible Conda Python 3.13 and its Python 3.11 environment have no
`torch`. No Torch/JIT, CPU/CUDA equality, configured-check, smoke timing or CUDA
peak pass is claimed, and no results JSON was fabricated.

## Expected speedup and remaining measurement

Eight sparse-table dispatches become one, including dynamic coalescing/ranking
and the unsafe recurrence. This can remove roughly seven eighths of the
per-table launch/synchronization overhead; it does not reduce the number of
queried slots or all sorting/data-transfer work. Global coalescing sorts a
larger record batch, and fused reads increase simultaneous scratch usage.
A several-fold improvement in the launch-bound sparse component is expected;
a 2-4x end-to-end improvement is a hypothesis, not a benchmark or target pass.
The <=67 s goal needs about 3.3x over the supplied 221.9 s reference.

The supplied profiler does not isolate `sum_rows`, so it cannot establish that
the deterministic reducer caused the iter38-to-39 slowdown. Source inspection
identifies its two dynamic unique-row outputs per mixer update as avoidable
synchronization points; the replacement removes those while preserving stable
sorting and exact segmented summation. A synchronized reducer/gradient profile
on the target installation is still needed to attribute the slowdown.

## Commands

Run from the repository root on the PyTorch-enabled installation:

```powershell
$py = 'C:/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe'
$env:CAP = '0'
& $py -X utf8 -B iter40.py --self-test
& $py -X utf8 -B iter40.py --self-test --check-device cuda
& $py -X utf8 -B iter40.py --mode tables --smoke 409600 --storage compact --budget-mib 64 --admit none --age-chunks 0
```

Exact full-run command, provided only and **not executed**:

```powershell
$py = 'C:/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe'
$env:CAP = '0'
& $py -X utf8 -B iter40.py --mode tables --storage compact --budget-mib 64 --admit none --ways 4 --age-chunks 0 --seed 0
```

Git Bash equivalent:

```bash
CAP=0 /c/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B iter40.py --mode tables --storage compact --budget-mib 64 --admit none --ways 4 --age-chunks 0 --seed 0
```

Full-run output:
`results/iter40_tables_compact_none_budget64_ways4_age0_door256_diag16_stale64_cap0_chunk4096_seed0.json`.
