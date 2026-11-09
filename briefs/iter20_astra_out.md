**Keep improving iter18. Iter20’s failure is primarily architectural, and my proposed simplification deserves the blame.** I reviewed the requested files statically; I did not execute experiments or modify files.

1. **Implementation audit.** I found no indexing, causality, gradient-sign, or update-order bug explaining the results. [iter20.py:29](D:/Aadit/claudez/brainnet/iter20.py:29) computes the stated local gradients; activations are emitted before updates.

   One mismatch with my original proposal: [line 50](D:/Aadit/claudez/brainnet/iter20.py:50) implements dense random \(U\), whereas I specified a sparse signed scatter. It matches your brief, however, and mainly increases computation.

   [Line 70](D:/Aadit/claudez/brainnet/iter20.py:70) averages current and retrieved states; line 72 carries that averaged state forward. This is a poor design choice, not a coding error. Biases are fine. The intentionally omitted retrieved-route gradients in `learned_input` match its spec.

2. **Ranked explanations and cheapest discriminating tests:**

   - **Missing temporal computation on learned features.** `local` concatenates current hidden activity with three hidden-state EMAs at every upper layer ([iter14.py:45](D:/Aadit/claudez/brainnet/iter14.py:45)). Iter20 only mixes time in its initial features, except retrieval. Test `noretrieve` with one additional .8 EMA input at each upper block; initialize its new weight rows to zero.
   - **Much smaller learned representation.** Iter20 trains approximately **136k parameters versus local’s 776k**, freezes input representations, and halves hidden width. Test H=256 alone. Separately compare fixed versus learned inputs with retrieval disabled and scale held at 1. Existing `learned_input` changes two factors and retains harmful averaging.
   - **Retrieval destroys context identity.** Three matches reduce the current-state coefficient to ¼; nested suffixes can retrieve the same successor repeatedly. Test an identity-preserving retrieval branch with independently learned coefficients initialized to zero. Repeat improvements are compatible with this damage: repeats constitute only **1.28% of scored renamed characters**.
   - **Scale/saturation and changing residual targets.** Scale 16 may saturate tanh; cumulative predictors also chase changing earlier logits. First measure saturation and compare scale 4 against 16. Then test independent readouts with learned mixing, holding everything else fixed.

3. **If repairing iter20:** use **D=256, H=256, three blocks**, learned inputs, scale 1, Adam **1e-3**. Set
   \[
   a=s+\sum_L\alpha_{\ell,L}m_Lr_L,\qquad s_{\rm next}=s+hU,
   \]
   with scalar \(\alpha\) initialized to zero and updated from the block’s local loss. Preserve the unmodified residual stream. These are experimental settings, not evidence that it will meet 90 seconds or recover 0.316 loss. I would not add copy heads before establishing a competitive underlying model.

4. **Iter18’s next missing capability is ordered combinations of learned features.** Its upper layers receive current features and temporal averages, but no explicit previous hidden states. Copy already handles repeated spelling well. Improving the language model is more promising than duplicating copy machinery. Closing the strict bars requires **0.0822 test and 0.0305 renamed**, not merely the 0.0705 same-sampler gap.

5. **One next experiment:** keep iter18’s three width-256 layers; add **lag-1 and lag-2 second-layer hidden states only to the final layer’s input**, expanding its matrix from **1024×256 to 1536×256**. Zero-initialize added rows. Keep batch 16, T=128, seed 0, network LR **1e-3**, gate LR **1e-2**, and existing copy heads.

   This adds 131,072 weights and approximately 262,144 forward/update MACs per character; timing remains uncertain. **Pass:** 6.215526M characters, ≤90 seconds, test ≤1.329, renamed ≤1.667. **Kill this configuration if any fails.** My odds: **15%**—it restores a specific missing computation, but the required loss reduction remains substantial.