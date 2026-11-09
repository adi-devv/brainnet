# iter19: count tables with merged predictive states

Create exactly one new file, `iter19.py`, in this folder. Do not edit any other file. Do not run git. Another job is building `iter18.py` at the same time; leave it alone.

Python: `/c/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B`. Read `iter12.py`, `iter10.py`, `iter8.py`, the `Counts` and `Model` classes in `iter5.py`, and `iter15.py` (`table_step`, `novel`) first. Match the repo's dense style, but keep it correct.

**Only run smoke tests (a few hundred thousand chars at most, under 3 minutes). Do not run the full experiments. I run those.** The GPU is shared with the other build's smoke tests.

## Idea
The tables only pool evidence between literally identical contexts. Merge contexts that predict the same futures into discrete states, then use state IDs to form new contexts, so that a context never seen literally can still hit counts when its pieces were merged. Everything is counts and discrete decisions: no gradients through any chain, no backward errors.

## Method
Subclass `iter12.Model` (variant `learned`). Training is the same as `iter12.run` (bs 976, one pass over `tr[:cut]`, cut = 95%, predict before write) and the existing tables and heads are unchanged.

**Replay checkpoints** at 25%, 50% and 75% of training chars. At each one, using only windows already trained on (keep a bounded buffer of them, at most 1.5M chars, by reservoir or evenly spaced retention):
1. Hold back a random 10% of buffered windows as replay-validation.
2. For suffix orders 4 and 8, collect the 65,536 most frequent contexts in the other 90%. For each, compute the next-char distribution at horizon 1 and the char-distribution at horizon 8 (the char 8 steps ahead), smoothed.
3. Cluster contexts per order into at most 2,048 states with k-means on the concatenated square-root distributions (Hellinger geometry), weighted by context frequency, fixed iterations, seeded, on GPU.
4. Accept test: on replay-validation, compare horizon-1 cross entropy of a predictor backed off through state counts against the same predictor with no merging, plus a storage penalty of 1e-6 nats per stored state per validation char. If merging does not improve it, keep every context as its own state for that order and log the rejection.
5. Rebuild: store the context-hash to state-ID map as sorted tensors (`searchsorted`). Unknown contexts get a reserved state ID. Clear and rebuild the new state tables below from the buffer, so counts match the new states.

**New predictors, added as extra mix inputs:**
- `state1`: counts keyed by the order-4 state ID at t.
- `state2`: counts keyed by the order-8 state ID at t.
- `chain`: counts keyed by the hash of order-4 state IDs at t, t-4, t-8, t-12, which is a discrete second layer, so a 16-char context can match through merged pieces.

Use tag-checked hash tables like `iter10.py` (2^21 cells each, int32 counts and tags), keyed by (state context, char), with +0.5 back-off smoothing onto the previous prediction, like the clock tables. Before the first checkpoint these inputs fall back to the previous prediction. The mix sees them through added inputs (widen the Neuron input count accordingly). State lookups must use only chars up to t.

Update the new tables with the same predict-before-write ordering as the rest.

## Variants
- `merge`: as above. Results `results/iter19_merge.json`.
- `nomerge`: identical code path and identical added tables and replay cost, but step 3 assigns each of the 65,536 contexts its own state, unmerged (hash-collapse any overflow into the reserved ID). This is the control: same replay, no merging. Results `results/iter19_nomerge.json`.

Open results with mode `'x'`, seed 0, memory cap as in `iter12.run` (2.7 GiB).

Report in JSON: `test_loss`, `renamed_loss`, `renamed_gap`, `val_loss`, novel first/repeat loss via `iter15.novel`, `train_seconds` (all training including buffer, clustering, rebuilds), `replay_seconds` separately, `chars`, `peak_allocated_mib`, per-checkpoint accept/reject and validation nats per order, number of live states, hit rate of each new predictor on test, settings. Also print a `pass` dict with these bars: test < 1.329, renamed < 1.667, train < 180 s, peak < 321 MB, and `merge` beating `nomerge` by >= 0.01 on renamed (this last one needs both result files, so compute it only if the other exists).

## Checks, run at the start
- `searchsorted` lookup equals a naive dict on a small case.
- Clustering plus rebuild leaves the state table sum equal to the number of buffered positions.
- Causality: `iter10.causality`, plus the state and chain contexts at positions <= t do not change when inputs after t change.
- `I.stamp` shows evaluation writes no state.

Add `--smoke N` to limit training chars, with checkpoints scaled to N, and write results to a temp path outside `results/`. Do not leave smoke files in the repo.

Finish with: what you ran, the smoke output, and any deviation from this brief.
