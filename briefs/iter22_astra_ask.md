Consult only: read anything, but do not edit, create or run anything. Answer in under 500 words.

Context: this repo researches a transformer-like architecture that learns without backprop and is both faster and better than backprop. An earlier review session prescribed iter 22; it is in `briefs/iter21_astra_out.md` and "you/your" below refers to that review. Read it, `PLAN.md`, `README.md`, `iter14.py`, `iter15.py`, `iter18.py` and `iter22.py` before answering.

We built it as you specified (`iter22.py`, spec `briefs/iter22_interleaved.md`, result `results/iter22_lagged.json`). All checks passed, including that the interleaved order gives the same weights as the post-forward update over 3 real batches, and an event trace of activate/update per layer. It failed one bar, so your kill rule applies to this exact configuration.

All runs: 6.2M chars, one pass, seed 0, GTX 1650. Loss in nats/char. Iter 22 is one seed.

| Run | Time | Test | Renamed | Novel first / repeat | Peak MB |
|---|---|---|---|---|---|
| Backprop GPT | 180 s | 1.329 | 1.6675 | 5.64 / 5.21 | 321 |
| Backprop, same one-pass sampler | 181 s | 1.341 | 1.686 | 5.71 / 5.38 | 319 |
| Iter 21 | 63.8 s | 1.380 | 1.6685 | 5.63 / 1.67 | 110 |
| **Iter 22** | 68.5 s | 1.3568 | 1.6462 | 5.60 / 1.68 | 139 |
| Bars | at most 90 s | at most 1.329 | at most 1.667 | | |

Iter 22 detail:
- Without copy heads: test 1.4343, renamed 1.7642. Iter 21 without copy heads: 1.4559, 1.7852.
- Per-layer test (pure net): 1.6197, 1.4764, 1.4487. Iter 21: 1.6197, 1.5153, 1.4654.
- `g` weights: -0.032, 0.340, 0.708.
- Gate weights: net .871, L1 .017, L2 .022, L4 .025, L8 .065.
- Validation 1.4980.
- Misses test by 0.028. Passes renamed by 0.021. Against same-sampler backprop: test worse by 0.016, renamed better by 0.040.
- 21.5 s of the 90 s budget is unspent.

The user's concern: within a layer our update equals backprop for that layer's own loss, but lower layers never receive what the layers above need. They asked whether this does what backprop would do. It does not, and we think that may be part of the remaining gap.

Your earlier next ideas were: two-ahead auxiliary on layers 0/1 (0.75 next / 0.25 two-ahead), then LR decay (1e-3 for 75% of steps, cosine to 1e-4), then final width 320.

Constraints: no cross-layer backprop; everything learns during the forward pass (per-layer update before the next layer runs). Within-layer exact gradients are allowed. Under 90 s, 4 GB GPU.

Questions:
1. Check `iter22.py` for bugs or a mismatch with your spec. Give line numbers, or say you found none.
2. Layer 0 did not change between iter 21 and 22 and is the weakest (1.6197, `g` near zero). Layer 1 improved by 0.039 from lags. What does that tell you about where the remaining 0.028 lies?
3. Is there a way, inside these constraints, to give lower layers the information backprop would give them, without a backward pass? Consider targets from upper-layer state, a delayed or previous-batch feedback signal, the two-ahead auxiliary, or anything better. Rank by expected gain per second of training time and say which you would discard.
4. Pick the one next experiment that best spends the 21.5 s. Give the exact configuration, pass and kill numbers on this benchmark, and your odds with one reason. If two seeds should be run first, say so.

Be direct. If something we did was a mistake, say so.
