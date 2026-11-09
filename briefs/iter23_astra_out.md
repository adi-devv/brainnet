1. **The auxiliary failed; it did not diagnose the remaining gap.** It weakens the hypothesis that this particular two-ahead objective, weighting and training horizon supplies enough useful supervision to close it. It does **not** rule out inadequate downstream credit, insufficient capacity/context, optimization limits, or better local objectives. Finite differences establish correct implementation, not useful supervision. Backprop’s seed spread does not establish the variance of this paired difference.

   Also, the remaining gap is **0.0240 against the headline baseline but 0.0123 against matched sampling**. Neither difference can be attributed entirely to local learning.

2. **My next experiment: final width 320.** This is a conservative choice under the runtime constraint, not a prediction that capacity is the bottleneck.

   Exact configuration: three layers, widths **256/256/320**; iter22 lag construction, local next-character losses, copy heads, gates, EMAs, sampler and split; batch 16, T=128, 6.2M characters, one pass, seed 0. Resize final-layer weights, bias, readout and optimizer state consistently, using the existing initialization rule; preserve lower-layer initialization. Use iter24’s specified network schedule and fixed gate LR. No auxiliary.

   Compare directly with iter24. **Pass:** ≤90 s, ≤4 GB, test ≤1.329, renamed ≤1.667; then replicate seeds 1 and 2. Otherwise kill this configuration. Deeper/wider stacks threaten runtime; the cached predictor adds a moving-target learning problem without evidence it supplies useful credit.

3. **Drop iter23 from the active baseline; retain the result.** Stack a marginal addition only after paired-seed evidence supports its gain and the combination satisfies the runtime ceiling. Backprop’s 0.006 spread is not a universal significance threshold. A single 0.004 improvement does not justify spending nine seconds permanently.

4. **Subjective odds: about 15%** that modest changes within this family clear both quality bars under 90 seconds. My earlier 25% for the auxiliary was too optimistic. If widening fails, I would change the local objective toward **residual prediction conditioned on detached lower-layer logits**—training each layer to improve the accumulated prediction—before buying more capacity or another auxiliary.