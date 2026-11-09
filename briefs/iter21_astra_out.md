**Keep iter21’s useful change, but retire this exact configuration. Next, add the same ordered-state inputs to layer 1.**

1. **Audit:** I found no bug or mismatch with the iter21 brief. [Lines 17–18](D:/Aadit/claudez/brainnet/iter21.py:17) correctly extend weights and optimizer state; [lines 28–34](D:/Aadit/claudez/brainnet/iter21.py:28) correctly construct causal, zero-padded lags. The inherited update handles the enlarged matrix.

   One qualification I previously missed: training is **not literally interleaved with forward computation**. [iter14.py:55](D:/Aadit/claudez/brainnet/iter14.py:55) completes the forward pass, then computes local updates. There is no cross-layer backprop. To meet your literal constraint, emit each layer’s old-weight activations/logits, then update that layer before advancing; update `g` and the copy gate at the end. Preserve old-weight values throughout each local gradient calculation.

2. **Keep 1.329/1.667 as the success bars.** They express the original practical goal; changing them after failure would weaken that claim. Use **1.340673/1.686344 as the controlled-comparison reference**, because it matches training exposure and sampling. Report both. The withheld 5% makes the original comparison imperfect, but does not invalidate the practical target.

3. **Ranked hypotheses; each test should change one factor:**

   - **Insufficient ordered processing in layer 1.** Layer 2 benefited from explicit lags; layer 1 still receives temporal averages. Cheapest test: add lag-1/2 of `hs[0]` to layer 1.
   - **Local objectives discard information useful downstream.** Low `g[0]` does **not** establish this: layer 0 still supplies features. Test a next-two-characters auxiliary readout on layers 0/1, with loss weights 0.75 next-character and 0.25 two-ahead; mask unavailable targets. Improvement would support richer local supervision, not prove downstream optimality.
   - **Constant learning rate leaves late optimization noise.** Test unchanged architecture with network LR 1e-3 for 75% of steps, then cosine decay to 1e-4; keep gate LR fixed.
   - **Final-layer capacity.** Test final width **320**, holding lower widths and lags fixed. The current measurements cannot distinguish insufficient capacity from poor input features.
   - **Withheld data.** It is validation data, not accidentally unused training data. Test all `tr` only as a separately labelled experiment, reporting the additional characters and surrendering that validation split.

4. **One next experiment:** three layers, widths **256/256/256**; raw-character lags **0–5**; hidden-state lags **1,2 at both upper layers**; retain all three EMAs. Extend `W1` from **1024×256 to 1536×256**, zero-initializing added rows. Keep `W2` unchanged. Network Adam **1e-3 constant**, gate Adam **1e-2 constant**; batch 16, T=128, seed 0, one pass over the same 95% split. Interleave updates as above.

   This adds 131,072 weights. Treat the claimed 26 seconds as uncertain headroom.

   **Pass:** ≤90 seconds, test ≤1.329, renamed ≤1.667. **Kill this configuration if any fails.** No two-seed rerun first; replicate a passing candidate with seeds 1 and 2. **Odds: 20%**, because ordered inputs already helped, but another 0.051 improvement is substantial.