1. **Iter 25 weakens my redundant-objectives diagnosis substantially.** I ranked it too confidently. The experiment also removed learned logit mixing and zero-initialized upper readouts, so it did not isolate objective redundancy. Zero readouts initially block gradients into upper representations; cumulative training then asks those representations to correct a changing lower predictor. In one pass, that adaptation cost could matter. Iter 24’s negative first-layer mixing weight also suggests useful calibration that fixed summation sacrifices. These are plausible explanations, not established causes. The evidence now favors preserving independent predictive supervision; it does not identify the dominant bottleneck.

2. **Yes—iter 26 remains my best next test**, with modest confidence. It preserves the successful objective and initial predictions while testing content-dependent context selection. Iter 25’s failure provides no positive evidence for gating; it mainly lowers confidence in my previous ranking.

3. **The gradient is correct**, provided all terms use pre-update parameters, lower states remain detached, and gate-bias gradients are summed too. Distinguish bias `d` from activation gradient `d`.

   - One gate per position is a sensible cheap first test. Per-unit gates add flexibility and substantially more parameters.
   - `3*pi` exactly preserves initialization, but later permits threefold branch amplification and competition between timescales. Watch tanh saturation.
   - Specify gate LR explicitly. I recommend **0.1× the network schedule**: peak `1e-4`, floor `1e-5`, same optimizer. This is a conservative heuristic.
   - Runtime includes three branch projections for gate gradients, not merely softmax.
   - The kill rule is coherent but can reject seed-sensitive improvements. Require **each run <90 s**, and mean paired validation reduction ≥0.010. Keep test out of selection.

4. **15%** probability of clearing both stages with that gate LR—subjective, not statistically estimated.