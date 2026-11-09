1. **Audit: I found no training bug or spec mismatch.** [iter22.py:18](D:/Aadit/claudez/brainnet/iter22.py:18) preserves initialization; lines 29–31 construct causal lags; lines 47–55 calculate gradients before mutation; lines 66–73 interleave updates while retaining old activations/logits. This is static inspection; I ran no experiments. README’s “without gradient descent” description is incorrect: manual Adam still uses gradients.

2. **The remaining 0.028 cannot be assigned to layer 0.** Its unchanged result follows mechanically from unchanged inputs, initialization and local objective. Near-zero `g[0]` discounts its logits, not its features. Layer 1’s improvement shows that layer 0 already contained information the previous upper-layer input arrangement used poorly. The pure-net gain, 0.0216, explains most of the combined gain, 0.0231. Missing downstream credit remains plausible, but these measurements do not establish it as the bottleneck.

3. **Exact current-batch downstream credit is unavailable before the downstream computation occurs.** Delaying that information changes which weights/examples it describes. Forward-mode differentiation does not remove this scheduling dependency.

   My expected-gain-per-added-second ranking, necessarily speculative:

   - **Two-ahead auxiliary on layers 0/1.** Cheap additional readouts can encourage features useful beyond immediate prediction. This supplies richer supervision, not backprop’s information.
   - **Late LR decay.** Essentially free, but evidence is weak: iter17’s earlier cosine experiment underperformed constant LR, although its schedule/exposure differed.
   - **Final width 320.** Straightforward additional capacity; more expensive and does not address downstream credit.
   - **Learned prediction of upper-layer logits from lower states**, trained using cached, correctly paired examples. This can convey delayed upper-layer knowledge without differentiating through upper layers, but adds a moving prediction problem during only 3,035 steps. Prefer logits over arbitrary hidden-state coordinates.

   **Discard** applying previous-batch errors directly to current-batch activations: examples do not correspond. Defer replay-based targets and inverse-map target propagation: overhead and instability are unjustified here. Standard [synthetic-gradient training](https://proceedings.mlr.press/v70/jaderberg17a/jaderberg17a.pdf) uses downstream derivatives to construct training targets; it is not automatically compliant. Random feedback already lost to exact local learning here.

4. **Next: the two-ahead auxiliary alone.** Keep iter22’s widths, lags, EMAs, sampler, 95% training split, batch 16, T=128, seed 0, one pass, constant network Adam 1e-3 and gate Adam 1e-2.

   Add separate zero-initialized 256×V readouts and V biases to layers 0/1, preserving existing initialization. Their local objective becomes **0.75 next-character CE + 0.25 two-ahead CE**. Use `y[:,1:]` as two-ahead targets, mask the final position and unavailable targets, and normalize each CE over valid targets. Sum weighted hidden gradients before clipping; update once before advancing. Auxiliary logits never enter `g` or the copy gate. Layer 2 stays unchanged.

   **Pass:** ≤90 s, test ≤1.329, renamed ≤1.667, within 4 GB. **Kill this configuration if any fails.** Headroom is a ceiling, not guaranteed runtime.

   No two-seed rerun first; replicate a pass with seeds 1 and 2. **Subjective odds: 25%**—cheap richer supervision is plausible, but recovering 0.028 remains substantial.