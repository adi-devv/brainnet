1. **My leading explanation is biased supervision feeding back through the representation hierarchy.** Your logit gradient is
\[
p_l-\big[(1-a)y+a\,p_{\rm top}\big].
\]
At \(a=0.5\), half the target becomes an immature, co-adapting teacher’s distribution. Lower layers learn its mistakes; the teacher then consumes their altered features. This changes gradient direction, not merely learning rate.

“Still greedy” explains why it may not close the BP gap; replacing reliable labels explains the additional damage. Top-only test loss worsens **1.390→1.489**, so mixing alone cannot explain it. However, one run cannot establish the dominant cause, and positive KL does **not** establish gradient conflict.

The risk was predictable; **+0.103 was not**. Iter32’s improvement-only distillation forecast was overconfident.

2. **Reasoning can reject mechanisms; it cannot reliably simulate the training trajectory.** Proposed screen, without running anything now:

- Reuse full-width local checkpoints plus Adam states at initialization and 10–20% progress. Future runs should save these. On four fixed batches, measure teacher CE/entropy and CE–KL gradient cosine.
- On backbone parameters, compare the **actual clipped Adam displacement** with negative BP gradients of top CE and, separately, iter30’s summed CE. Also measure displacement norm and directional derivative; cosine alone misses magnitude. BP is diagnostic only.
- Apply disposable candidate updates and measure top/mixed loss on separate batches. Then branch local versus candidate for 64 identical batches, evaluating 32 held-out windows. Preserve the original schedule; `--smoke` compresses it.
- Budget **roughly 5–15 seconds warm**—an estimate, excluding setup. Without checkpoints, budget their creation separately.

These screens detect immediate interference reasonably well, but miss delayed alignment and representation adaptation. Linear probes are especially misleading here: BP already worsens layer-0 linear prediction while improving the top. Tiny architectures also change the problem. Prediction accuracy is **uncalibrated** until retrospectively tested across prior ideas; survivors still need full runs and replication.

3. **Next experiment: rank 1.** Forecasts below are **subjective ranges, not confidence intervals**, versus local **1.4564**.

| Rank | Experiment | Predicted val; first proxy |
|---|---|---|
| 1 | Lower-layer training-only nonlinear heads, e.g. 256→64→96. Their CE trains each local representation; existing inference readouts train on detached features. Top unchanged. | **1.42–1.49**; paired 64-step held-out top-loss change after head warmup. |
| 2 | Local CE plus modest, norm-controlled direct random projection of detached top error into lower hidden updates. | **1.43–1.55**; downstream directional derivative after adaptation. |

[Local auxiliary learning](https://proceedings.mlr.press/v119/belilovsky20a.html) and [direct feedback alignment](https://arxiv.org/abs/1609.01596) provide precedents, not large-language-model guarantees. DTP adds inverse-learning cost and [noninvertibility problems](https://arxiv.org/abs/2006.14331); perturbation methods face variance/sample-cost pressure. Neither ranks ahead here. Require measured time-to-target gains before scaling.

No files changed or training run.