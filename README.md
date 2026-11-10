# brainnet

Notes and experiments on neural networks that learn **without gradient descent**.

Nothing here is proven. This is a working list of pointers and the approaches being tried, with honest status on each.

## Pointers

- The brain has no global backward pass. Synapses change from local signals, and a reward or surprise signal decides which changes stick.
- New experiences are stored fast (hippocampus-like), then replayed in sleep to slowly train long-term weights (cortex-like).
- Gradient descent needs a differentiable path from the result to the weights. A hard pass/fail reward has none, so search or local rules are needed.
- The price of going gradient-free is sample efficiency. Expect many more trials per useful update.
- Results so far (tiny character-level code model, GTX 1650): backprop-free count tables already beat a backprop transformer at the same data; see Findings.

## Approaches

| Approach | Idea | Status |
|---|---|---|
| Evolution strategies | Copy the weights with random noise, score each copy by reward, move toward the best | Tried on Sudoku with a 0/1 reward |
| Meta-evolution | Evolve the learner's architecture and hyperparameters across lives so each new life learns faster | Tried on Sudoku together with the above |
| Hebbian and three-factor rules | Local "fire together, wire together" updates, gated by a reward or surprise signal | Not started |
| Predictive coding, forward-forward | Layer-local error signals instead of one global backward pass | Not started |
| Closed-form memory writes | Write a fact straight into the weights with a rank-one solve (MEMIT/AlphaEdit style), no training loop | Idea only; still needs a few gradient steps to find the target value |

Current direction and pass/kill bars: [PLAN.md](PLAN.md).

## Findings (iterations 1-18, character-level code LM, test loss in nats/char, lower is better)

All backprop-free rows use hand-written updates with grad mode off. Seed 0 unless noted.

| Method | 1M chars | 6.2M chars | Renamed-identifier test | Time |
|---|---|---|---|---|
| Backprop GPT (baseline) | 2.330 | 1.329 / 1.335 (2 seeds) | 1.667 at 6.2M (gap 0.339); 1.530 at 16M (gap 0.406) | ~181 s at 6.2M, 321 MB |
| Backprop GPT, same one-pass windows as `local` (iter 18 control) | - | 1.341 | 1.686 at 6.2M (gap 0.346) | 181 s at 6.2M, 319 MB |
| DFA stacked learner (iter 14: fixed causal EMAs, 6 lag lookups, random feedback) | 1.992 | 1.549 | 1.862 at 6.2M (gap 0.313) | 69.5 s at 6.2M (2.6x faster) |
| Exact layer-local gradient (iter 15 `local`) | 1.909 | 1.492 | 1.819 at 6.2M (gap 0.327) | 66-73 s at 6.2M, 105 MB |
| `local` + suffix-match copy heads (iter 18) | - | 1.411 | 1.697 at 6.2M (gap 0.286) | 74.5 s at 6.2M, 103 MB |
| Count tables + learned pointer heads (iter 12) | 1.268 | **1.165** | 1.606 at 6.2M (gap 0.441) | 244 s at 6.2M, ~1.9 GB |
| Tables + local net (iter 15 `hybrid`) | 1.261 | 1.155 | 1.596 at 6.2M (gap 0.441) | 707 s at 6.2M, ~2 GB |

Backprop at 16M chars: 1.124 / 1.139.

What this says:
- At equal data (6.2M), backprop-free tables beat backprop (1.165 vs 1.329). They are one-shot memories, so they copy repeated new names well (loss 3.2 on a repeat vs 5.9 on first sight).
- The tables have a large renamed gap (0.44 at 6.2M), but their absolute renamed loss still beats backprop at equal data (1.606 vs 1.667). An earlier version of this note said they "do not generalize", which overstated it. They cost about 6x backprop's memory and more time.
- The DFA net is about 2.6x cheaper per step than backprop but loses on quality at 6.2M. It wins only in the first ~1M chars.
- Exact layer-local gradient beats random feedback at 1M (1.909 vs 1.992).
- Adding the local net to the tables gains little (0.010 test at 6.2M) and takes about 3x longer.
- `local` is about 2.5x faster than backprop per char but worse at equal data (1.492 vs 1.329). More passes in equal time do not close it (2.5 passes, 163 s: 1.353 test, 1.720 renamed).
- `local` does avoid backprop across layers, but it still uses the exact chain rule inside each layer. It also has no attention, so its speedup is partly from a smaller model.
- Copy heads (iter 18) add 0.08 test at 2.4x backprop's speed. Renamed loss nearly reaches backprop on the same sampling (1.697 vs 1.686), and repeated new names are far cheaper (1.66 vs 5.38 nats). Test loss still misses (1.411 vs 1.341).
- Open: see PLAN.md.

## Findings (iterations 20-25, same task, 6.2M chars, one pass, seed 0 unless noted)

Each layer learns from its own loss. There is no cross-layer backprop, but gradients inside a layer are still exact. Since iter 22, each layer updates during the forward sweep, before the next layer runs.

| Method | Test | Renamed | Val | Time | Memory |
|---|---|---|---|---|---|
| Iter 18 (`local` + copy heads) | 1.411 | 1.697 | - | 74.5 s | 103 MB |
| Iter 21: lagged lower-layer state into the final layer | 1.380 | 1.668 | 1.538 | 63.8 s | 110 MB |
| Iter 22: lags at both upper layers, updates during the forward sweep | 1.357 | 1.646 | 1.498 | 68.5 s | 139 MB |
| Iter 23: iter 22 + two-ahead auxiliary readouts (dropped) | 1.353 | 1.643 | 1.494 | 77.4 s | 150 MB |
| **Iter 24: iter 22 + LR held for 75% of steps, then cosine to 1e-4 (3-seed mean)** | **1.332** | **1.630** | **1.468** | **66.1 s** | **139 MB** |
| Iter 25: iter 24 with layers learning detached cumulative-logit residuals (dropped) | 1.380 | 1.701 | 1.528 | 62.3 s | 164 MB |

What this says:
- Iter 24 matches backprop's test loss (1.332 vs 1.332, backprop's 2-seed mean on its own sampler; 1.341 on the same windows) at about 2.7x the speed, 66 s vs 181 s, using 139 MB vs 321 MB.
- Its renamed-identifier loss is lower than backprop's single measured seed (1.630 vs 1.667).
- This is parity at higher speed, not a demonstrated quality win. The test set was reused to choose between iterations through iter 24, so it is now development data. Selection uses validation from here on.
- Backprop still wins with more data: 1.124 test at 16M chars. Iter 24 has not been run at that scale.
- Learning-rate decay was the largest single gain, 0.025 test for no added time.
- Iter 20 (associative retrieval) did not help: 1.645 test with retrieval, 1.573 without.
- Iter 25 made validation 0.061 worse, so redundant layer objectives were not the bottleneck in this setup. Its code and result stay in the repo as a record.

Code is in `iterN.py`, briefs in `briefs/`, raw numbers in `results/`.

## Findings (iterations 26-37, same task and splits, 6.2M chars, one pass, seed 0 unless noted)

Iter 30 adds a fair control: the same layered architecture trained with ordinary backprop.

| Method | Test | Renamed | Val | Time | Peak memory |
|---|---|---|---|---|---|
| Iter 26: EMA gates (3 seeds) | 1.321 | - | 1.459 | - | - |
| Iter 27: width 320 (3 seeds) | - | - | 1.458 | - | 136 MiB |
| Iter 28: soft tanh (dropped) | - | - | 1.480 | - | - |
| Iter 29: gates + width 320 | - | - | 1.449 | 107 s | - |
| Iter 30: local learning control | 1.323 | - | 1.456 | 69.8 s | - |
| Iter 30: local + attention | 1.291 | - | 1.410 | - | - |
| **Iter 30: backprop, same architecture (BP reference)** | **1.256** | **1.571** | **1.387** | **79.4 s** | **151 MiB** |
| Iter 33: top-down teaching (dropped) | - | - | 1.560 | - | - |
| Iter 34: auxiliary heads (dropped) | - | - | 1.492 | - | - |
| **Iter 35: one-shot count tables, no backprop** | **1.223** | 1.588 | **1.341** | **66.9 s** | **103 MiB** |
| Iter 35: tables + local learned keys | 1.228 | 1.594 | 1.342 | 85.8 s | 132 MiB |
| Iter 37: tables, 4x capacity | 1.189 | 1.555 | 1.290 | 68.3 s | 240 MiB |
| Iter 37: tables, 16x capacity | 1.177 | 1.544 | 1.262 | 72.7 s | 792 MiB |

What this says:
- Count tables (hashed context to next-char counts, mixed by a local log-linear pool) beat matched backprop on test and val with about 29k FLOPs/char, 248x fewer than backprop's 7.19M, and less time and memory. Renamed code is still slightly worse at 1x capacity.
- Iter 35 seemed to plateau late in training (val 1.293 to 1.269 over the last 60%). Iter 37 shows the cause is capacity: with 4x and 16x tables the curve keeps falling (16x: 1.236 to 1.178), and both beat backprop on test and renamed.
- The cost is memory: 16x needs 792 MiB against backprop's 151 MiB. Iter 38 (in progress) targets 16x quality within 151 MiB using compact 3-byte entries, 4-way buckets with frequency eviction, a doorkeeper admission filter and budgeted growth.
- Wall time is only 1.2x faster than backprop despite 248x fewer FLOPs. The time goes to hashing, gathers, sync and diagnostics, not arithmetic.
- Caveats: one seed, 6.2M chars, and the test set is development data. Count models are known to be strong at small data, and backprop may close the gap at scale (Astra's review, `briefs/iter37_astra_out.md`). A data-scaling screen against matched and larger backprop models is the decisive next test.
- Iter 36 (`roles.py`, identifiers keyed by causal role for the renamed gap) is built but not yet run.

## Findings (iterations 38-39: compact associative storage, 64 MiB, one pass, seed 0)

| Method | Test | Renamed | Val | Time | Peak memory |
|---|---|---|---|---|---|
| Iter 38: compact 4-way, doorkeeper admission | 1.200 | 1.559 | 1.325 | 209.6 s | 116 MiB |
| Iter 38: compact, no admission | 1.187 | 1.561 | 1.306 | 194.4 s | 117 MiB |
| **Iter 38: compact, no admission, no aging** | **1.171** | **1.545** | **1.277** | 193.3 s | **117 MiB** |
| Iter 39: vectorized write + deterministic mixer sums | 1.171 | 1.545 | 1.277 | 221.9 s | 117 MiB |

- Compact no-aging storage beats BP (1.256 / 1.571) and 16x tables (1.177 / 1.544) on test, at 117 MiB versus 151 MiB.
- Iter 39 is bit-exact with iter38. The write is 2x faster in a smoke run, but the full run is still 222 s. Per-table, launch-bound writes and reads dominate. Iter 39 also fixes a flaky determinism check: CUDA float `index_add_` atomics in the mixer gradients were replaced with a sorted segment reduce.
- Next: fuse all sparse tables into one flat store (one gather and one scatter per chunk) to reach <=67 s.
