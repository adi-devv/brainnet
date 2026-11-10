Consult only: do not edit, create or run anything. Quota is tight, so read only `briefs/iter24b_astra_out.md`, `iter22.py` lines 1-75 and `iter24.py` lines 1-25. Answer in under 300 words.

Context: this repo researches a transformer-like architecture that learns without cross-layer backprop and is both faster and better than backprop. Each layer learns from its own local loss, with updates during the forward sweep. Within-layer exact gradients are allowed. Dataset, one pass and the 90 s budget on a GTX 1650 are fixed. We select on validation, not test.

Base (iter 24, seeds 0/1/2): 66 s, val 1.4669/1.4693/1.4686, test mean 1.3320, renamed mean 1.6298. Per-layer test 1.6442, 1.4592, 1.4012. Learned `g` weights [-0.089, 0.356, 0.759].

Iter 25 (your first choice, detached cumulative-logit residual learning, seed 0): val 1.5276, 0.061 WORSE than iter 24. Test 1.3795, renamed 1.7009, 62 s, cumulative per-layer test 1.6442, 1.4674, 1.4517. Failed the kill rule, seeds 1-2 not run. Base stays iter 24.

Iter 26 plan (your second choice, built on iter 24): at each upper layer l, `pi = softmax(lower @ G + d)` with G (256x3) and d (3) zero-initialized, so it starts identical to iter 24. Input becomes `[lower, 3*pi_0*ema0, 3*pi_1*ema1, 3*pi_2*ema2, lag1, lag2]`, same W. Gates train on the layer's own local loss by exact gradient: `v_k = ema_k @ W_block_k`, `q_k = 3*(d*v_k).sum(-1)` where d is the layer's dL/da, `dL/dlogit_k = pi_k*(q_k - sum_j pi_j q_j)`, then G grad = `lower.T @` that, summed over batch and time. One softmax per position, not per unit. Everything else equals iter 24. Kill rule: seed 0 val <= 1.4569 and < 90 s, then three paired seeds average >= 0.010 improvement.

Questions:
1. Why do you think iter 25 lost 0.061? What does that say about your diagnosis of redundant objectives, and does it change what limits this learner?
2. Given that, is iter 26 still the best next test? Say yes or no. If no, name the one replacement and its exact configuration.
3. Any flaw in the iter 26 design or kill rule? For example: a single gate per position versus per-unit gates, gate LR, or the scale of 3*pi.
4. Your honest probability that iter 26 clears the kill rule.

Be direct.
