# iter39 configured-check determinism

## Cause

`LocalMixer.gradients` accumulated its `char` and `pair` gradients with floating-point CUDA `index_add_`. Repeated selectors send several contributions to the same output element. CUDA atomic additions can arrive in different orders, and floating-point addition is order-sensitive. The rounding differences enter Adam's parameters and moments, then appear in the final evaluation scores. Seeding the RNG cannot fix atomic scheduling.

The supplied `../dbg39d.py` leaves a gap in the claimed mixer comparison. Its `tens(model.mixer)` inspects direct attributes, but `LocalMixer` holds only `E` and `opt`. All learned tensors are nested inside `model.mixer.opt.p`, `.m`, and `.v`. Consequently, the script captures **zero mixer tensors**. Equal table tensors and equal earlier scores are compatible with a differing final mixer update. Disabling collection freezes diagnostics, while scores continue to use the learned mixer.

The same floating-point accumulation exists in iter38. Its reported 0/6 outcomes cannot establish deterministic accumulation. iter39's different launch/allocation history can expose the inherited race. The connection to the observed flaky run is a source-based diagnosis. This sandbox could not execute PyTorch to confirm the first differing optimizer tensor dynamically.

## Fix

- Added `sum_rows`, which stably sorts selectors, reduces contiguous segments, then assigns each unique destination row once. Stable sorting preserves the original contribution order within each row. `torch.segment_reduce` performs the segment sums without competing floating-point atomic writers. Segment lengths come directly from the same sorted indices, so `unsafe=True` skips redundant synchronizing validation.
- Replaced the mixer's two floating-point row accumulations with this helper. The keys network's six `A_j` gradient accumulations use the same helper because they had the same repeated-index issue.
- Strengthened `configured_checks` to compare every nested parameter and both Adam moments after each of its four updates. Keys-mode checks include the keys optimizer. The final `torch.equal(*outputs)` assertion remains exact.
- Added self-test coverage for cancellation-sensitive float32/float64 contributions, empty and single updates, unused rows, and repeated 4096-by-96 floating-point updates. The numerical-reference check accompanies exact repeat checks.

`Buckets`, `LogCount`, hashes, admission, aging, reads, and the table write algorithm are byte-for-byte unchanged by this patch. In tables mode, mixer weights do not affect context construction or targets, so selecting a deterministic mixer accumulation order preserves the iter38 table stream and bytes. The existing independent iter38 table oracle remains in `--self-test` for CPU and CUDA.

The deterministic accumulation order can change learned weights at rounding precision relative to earlier runs. Keys-mode learned addresses can consequently change relative to an earlier run. Associative-table exactness is defined for the same context/target stream. No global deterministic setting is changed, and the scoring arithmetic and tolerances are retained.

## Write-path review

- **Resident updates** coalesce `(bucket, fingerprint)` records before writing. Different records cannot select the same resident slot. The highest-way match preserves iter38's behavior if duplicate resident tags already exist.
- **Miss updates** use unique `(group, rank)` scratch destinations. Stable count sorting breaks ties by way. The scripted recurrence preserves sequential re-eviction, and its padded turns occur after all real turns for a bucket. Integer maximum reduction chooses the final rank for each physical victim, leaving a single tag/count writer per victim.
- **Encoding and aging** use stateless hashes of identity, write number, and salt. Aging slabs have disjoint destinations. Combining encodes does not introduce a random-number-generator dependency.
- **Tagged tables** elect a maximum fingerprint using integer reduction. `MeasuredTagged` coalesces duplicate records and keeps one winner per cell. The baseline tagged scratch stores identical elected values at repeated indices.
- **Dense tables and unigram** use integer `index_add_`. These additions are order-independent within the documented one-pass count bound. Bloom updates first deduplicate bit positions and then integer-sum disjoint masks before one update per byte.
- **Diagnostics** use integer reductions. Repeated usefulness destinations all receive the same zero or one, and default-stream operation order separates clearing from recording. Device-side statistics introduce no competing floating-point writers.
- **Other learner operations** update each Adam element once. The keys CE `scatter_add_` has one target per row. Copy-head scatter additions sum exact zeros and ones with at most 127 contributors for the supported 128-character window. These do not have the learner-row accumulation problem.

No additional conflicting-value table write was found in this review. The reasoning assumes the existing serialized, single-stream model API. CPU/CUDA oracle execution is still required to validate the implementation on the target installation.

## Verification

- Python AST parsing and whitespace checks passed.
- AST comparison against the pre-edit file found changes only to `LocalMixer`, `KeysNet`, `storage_checks`, and `configured_checks`, plus the two new helper/check functions.
- The entire `Buckets` AST is identical, and the final exact score assertion is retained.
- Both CPU and CUDA self-test commands were attempted with accessible Conda Python. They stopped at import with `ModuleNotFoundError` for `torch`. The other accessible Conda environment also lacks PyTorch.
- The Python 3.12 executable named in the original brief exists, but execution is denied by the sandbox. Runtime determinism, CPU/CUDA table equality, performance, and memory have therefore not been measured here.

Run these commands from the repository on the PyTorch-enabled installation. The repeated reproduction uses fresh processes and retains the supplied sequence of `checks`, CPU storage checks, and configured checks.

```powershell
$py = 'C:/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe'
$env:CAP = '0'
& $py -X utf8 -B iter39.py --self-test
if ($LASTEXITCODE) { throw 'CPU self-test failed' }
& $py -X utf8 -B iter39.py --self-test --check-device cuda
if ($LASTEXITCODE) { throw 'CUDA self-test failed' }
1..20 | ForEach-Object {
    $output = & $py -X utf8 -B ../dbg39c.py iter39 2>&1
    $code = $LASTEXITCODE
    $output
    if ($code -ne 0 -or ($output -join "`n") -match '\bFAIL\b') {
        throw "Fresh-process reproduction failed on repeat $_"
    }
}
& $py -X utf8 -B -c "import json, iter39 as m; v=json.loads((m.I.ROOT/'data/meta.json').read_text())['vocab']; o=m.StorageOptions(storage='compact', admit='none', budget_mib=64., ways=4, age_chunks=0); print(m.checks(v, 'keys', 'cuda')); print(m.configured_checks(v, 'keys', 'cuda', o))"
if ($LASTEXITCODE) { throw 'Keys configured checks failed' }
```

## Diff

The diff below is against iter39 as it existed at the start of this task.

```diff
--- a/iter39.py
+++ b/iter39.py
@@ -546,6 +546,24 @@
                 for k, v in tensors.items()}
 
 
+def sum_rows(template, index, values):
+    """Sum learner contributions in a fixed order, with one writer per row.
+
+    CUDA float index_add_ uses unordered atomics for repeated indices. Stable
+    sorting retains input order within each row; segmented reduction owns each
+    output element, and the final assignment has unique row indices.
+    """
+    result = torch.zeros_like(template)
+    if index.numel() == 0:
+        return result
+    order = index.argsort(stable=True)
+    rows, lengths = index[order].unique_consecutive(return_counts=True)
+    # lengths comes from this exact permutation, so it is positive and covers
+    # all values. Skip segment_reduce's synchronizing input validation.
+    result[rows] = torch.segment_reduce(values[order], 'sum', lengths=lengths, unsafe=True)
+    return result
+
+
 class LocalMixer:
     """hybrid_mixer log-linear pool, with char/pair rows and confidence slopes.
 
@@ -573,8 +591,8 @@
         # Negative CE derivative wrt the independent log-linear weights.
         d = (expected-actual)*valid[:, None]/valid.sum().clamp_min(1)
         ch, pair = selectors
-        return dict(char=torch.zeros_like(self.opt.p['char']).index_add_(0, ch, d),
-                    pair=torch.zeros_like(self.opt.p['pair']).index_add_(0, pair, d),
+        return dict(char=sum_rows(self.opt.p['char'], ch, d),
+                    pair=sum_rows(self.opt.p['pair'], pair, d),
                     confidence=(d[..., None]*phi).sum(0))
 
 
@@ -629,7 +647,7 @@
         d = (e @ self.opt.p['R'].T)*(1-h.square())
         hf, ef, df = (a.flatten(0, 1) for a in (h, e, d))
         grads = dict(R=hf.T @ ef, c=ef.sum(0), b=df.sum(0))
-        grads.update({f'A{j}': torch.zeros_like(self.opt.p[f'A{j}']).index_add_(0, idx.flatten(), df)
+        grads.update({f'A{j}': sum_rows(self.opt.p[f'A{j}'], idx.flatten(), df)
                       for j, idx in enumerate(ids)})
         grads.update({f'P{k}': em[:, k].flatten(0, 1).T @ df for k in range(3)})
         return grads
@@ -1216,6 +1234,34 @@
                 oracle='iter38.Buckets.write', bit_exact=True)
 
 
+def row_sum_checks(device):
+    """Exercise repeated learner rows, cancellation, and empty updates."""
+    index = torch.tensor([2, 0, 2, 0, 2, 0, 5], device=device)
+    for dtype, large in ((torch.float32, 2**24), (torch.float64, 2**53)):
+        template = torch.zeros(6, 2, dtype=dtype, device=device)
+        values = torch.tensor([[large, 1], [4, 2], [-large, 3], [-4, 4],
+                               [1, 5], [.5, 6], [7, 8]], dtype=dtype, device=device)
+        want = torch.tensor([[.5, 12], [0, 0], [1, 9], [0, 0], [0, 0], [7, 8]],
+                            dtype=dtype, device=device)
+        assert torch.equal(sum_rows(template, index, values), want)
+        assert torch.equal(sum_rows(template, index[:0], values[:0]), template)
+        assert torch.equal(sum_rows(template, index[-1:], values[-1:]),
+                           torch.cat((template[:5], values[-1:])))
+
+    # Match the learner's large repeated-index updates, with unequal signed
+    # floats and missing rows. Require exact repeats, including every feature.
+    index = torch.arange(4096, device=device) % 7
+    values = torch.arange(4096*96, dtype=torch.float32, device=device).reshape(4096, 96).sin()
+    template = torch.zeros(9, 96, device=device)
+    want = sum_rows(template, index, values)
+    for repeat in range(8):
+        assert torch.equal(sum_rows(template, index, values), want), (device, repeat)
+    reference = torch.stack([values[index == row].double().sum(0) for row in range(9)])
+    torch.testing.assert_close(want.double(), reference, rtol=0, atol=1e-4)
+    return dict(stable_row_order=True, empty_and_single_updates=True,
+                repeated_float_rows_bit_exact=True)
+
+
 def storage_checks(device='cpu'):
     """Small storage checks, runnable without loading data or requiring CUDA."""
     codec = LogCount(device)
@@ -1372,7 +1418,8 @@
                 deterministic_storage_twice=True, frequency_eviction=True, count_aging=True,
                 packed_bloom_concurrent_bits=True, resident_updates_survive_bloom_reset=True,
                 tagged_matches_iter37=True, budget_includes_all_persistent_storage=True,
-                associative_matches_iter38=associative_exactness_checks(device))
+                associative_matches_iter38=associative_exactness_checks(device),
+                learner_row_sums=row_sum_checks(device))
 
 
 def result_tag(mode, options, chunk_size, seed):
@@ -1401,6 +1448,7 @@
 
 def configured_checks(vocab, mode, device, options):
     outputs = []
+    learned_states = []
     with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
         for repeat in range(2):
             torch.manual_seed(38)
@@ -1410,6 +1458,19 @@
             for step in range(4):
                 q, state = model.score(x)
                 model.learn_write(x, y, q, state, .001)
+                # Optimizer tensors are nested, not direct mixer attributes.
+                optimizers = dict(mixer=model.mixer.opt)
+                if model.net:
+                    optimizers['keys'] = model.net.opt
+                snapshot = {f'{module}.{kind}.{name}': tensor.clone()
+                            for module, opt in optimizers.items()
+                            for kind in ('p', 'm', 'v')
+                            for name, tensor in getattr(opt, kind).items()}
+                if repeat == 0:
+                    learned_states.append(snapshot)
+                else:
+                    for name, tensor in snapshot.items():
+                        assert torch.equal(tensor, learned_states[step][name]), (step, name)
             model.set_collect(False)
             before = I.stamp(model)
             a = model.score(x)
@@ -1427,7 +1488,8 @@
             del model, q, state, a, b
     assert torch.equal(*outputs), 'configured outputs differ between identical runs'
     return dict(configured_storage_causal=True, configured_eval_frozen=True,
-                configured_outputs_deterministic_twice=True)
+                configured_outputs_deterministic_twice=True,
+                configured_optimizer_deterministic_twice=True)
 
 
 def run(mode, smoke=None, chunk_size=4096, seed=0, options=None):
```
