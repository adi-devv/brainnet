Follow-up on your iter 18 upgrade. Consult only: read anything, but do not edit, create or run anything. Answer in under 500 words.

We built it as you specified (`iter21.py`, spec `briefs/iter21_lag.md`, result `results/iter21_lagged.json`). All checks passed. It is close but failed your bars, so your kill rule applies to this exact configuration. Please check the work, then tell us the next step.

All runs: 6.2M chars, one pass, seed 0, GTX 1650. Loss in nats/char. Iter 21 is one seed.

| Run | Time | Test | Renamed | Novel first / repeat | Peak MB |
|---|---|---|---|---|---|
| Backprop GPT | 180 s | 1.329 | 1.6675 | 5.64 / 5.21 | 321 |
| Backprop, same one-pass sampler as local | 181 s | 1.341 | 1.686 | 5.71 / 5.38 | 319 |
| Iter 18 (`local` + copy heads) | 74.5 s | 1.411 | 1.6975 | 5.69 / 1.66 | 103 |
| **Iter 21 (iter 18 + lag-1, lag-2 of `hs[1]` into final layer)** | 63.8 s | 1.380 | 1.6685 | 5.63 / 1.67 | 110 |
| Bars | at most 90 s | at most 1.329 | at most 1.667 | | |

Iter 21 detail:
- Without copy heads: test 1.4559, renamed 1.7852. Iter 18 without copy heads: 1.4922, 1.8190.
- Per-layer test (pure net): 1.6197, 1.5153, 1.4654.
- `g` weights: -0.025, 0.231, 0.813.
- Gate weights: net .862, L1 .020, L2 .022, L4 .025, L8 .071.
- Validation 1.5376. Steps 3035.
- Misses: test by 0.051, renamed by 0.0015. Against same-sampler backprop: test worse by 0.039, renamed better by 0.018.
- 26 s of the 90 s budget is unspent. Treat the 63.8 s vs 74.5 s difference as timing noise.

Constraints: no cross-layer backprop; everything must learn during the forward pass. Within-layer exact gradients are allowed. Must stay under 90 s and on a 4 GB GPU.

Questions:
1. Check `iter21.py` for bugs or a mismatch with your spec. Give line numbers, or say you found none.
2. Which bar is the right one? `local` trains on 95% of `tr` with non-overlapping windows, and standard backprop samples overlapping windows from all of `tr`. Should the pass bar stay 1.329/1.667, or move to the same-sampler numbers? Say which and why.
3. Rank the remaining gap's likely causes, each with the cheapest experiment to confirm or reject it. Consider at least: layer 0 and 1 being trained only on their own local loss, too few or wrong lags, first-pass learning-rate choice, final-layer width, and an unused 5% of training data.
4. Pick the one next experiment that best spends the 26 s. Give exact configuration (lags, widths, learning rates, schedule, data split), pass and kill numbers on this benchmark, and your odds with one reason. If two seeds should be run first, say so.

Be direct. If something we did was a mistake, say so.
