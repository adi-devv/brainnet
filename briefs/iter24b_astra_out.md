1. **My leading suspect is redundant local objectives:** every layer learns a complete next-character predictor rather than learning what earlier layers missed. Loss falls 1.6442 → 1.4592 → 1.4012, showing useful depth but diminishing gains. The final 1.3320 includes mixing and copying, so that improvement cannot establish effective representation learning. These numbers support the hypothesis; they do **not** isolate the dominant bottleneck. Fixed, content-independent context mixing is the strongest competing explanation.

2. Ranked by **expected validation gain per added second**:

   1. **Detached cumulative-logit residual learning.** Replace independent prediction objectives with successive corrections. Almost no additional computation; directly addresses redundant learning. I retain this above widening.
   2. **Content-dependent gates over the existing EMA branches.** At each upper layer, predict three softmax weights from the current lower-layer state and use them to weight its three EMA summaries. Train gates through that layer’s local loss. Cheap adaptive context selection, although it cannot recover information already lost through averaging.
   3. **Width 320.** Straightforward capacity increase, but dominant dense operations grow roughly 1.56×. With only 24 seconds available, its validation gain per second looks worse and runtime could exceed budget.

3. **First experiment:** retain three layers, width 256, existing context features, copy head, initialization except as below, and the iter24 schedule. Keep the same data/order, batching and single pass.

   Define \(s_0=r_0\), \(s_l=\operatorname{stopgrad}(s_{l-1})+r_l\). Train layer \(l\) with CE on \(s_l\), differentiating only its parameters. Zero-initialize upper-layer readout weights/biases; retain pre-update states/logits during the sweep. Replace learned `g` mixing with \(s_2\); train the existing copy gate against its probabilities.

   **Projected time: 65–70 seconds**, unmeasured. Screen seed 0: validation improvement ≥0.010 against matched iter24, runtime <90 seconds. Then require ≥0.010 mean improvement across three paired seeds. Otherwise kill this configuration.

4. **Locally trainable attention: yes. More promising within budget than #1: no.** A small attention branch could fit, but exact local attention gradients add substantial computation, and failed associative retrieval lowers my confidence. Residual learning tests a clearer objective defect more cheaply.