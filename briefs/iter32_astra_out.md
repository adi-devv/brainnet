**1. Credible research goal; weak basis for expecting victory.** Matching backprop without cross-layer derivatives is plausible. Faster *and* better large-scale language modelling remains unestablished by the cited evidence.

- [Forward-forward](https://arxiv.org/abs/2212.13345) and [PEPITA](https://arxiv.org/abs/2201.11665): small-benchmark demonstrations; extra forward passes consume potential savings.
- [FA/DFA scaling studies](https://arxiv.org/abs/1807.04587): substantial CIFAR/ImageNet deficits. [Signal propagation](https://arxiv.org/html/2204.01723v2): CIFAR-100 error 34.30% versus BP’s 26.20%; reported parallel timings are per-layer.
- [Greedy/local-loss DGL](https://proceedings.mlr.press/v119/belilovsky20a/belilovsky20a.pdf): genuine ImageNet evidence, but per-layer VGG-13 scores 64.4% versus 66.6%; stronger blockwise results retain within-block backprop.
- [Predictive coding](https://proceedings.mlr.press/v306/qi26f.html): recent deep-ResNet parity, but iterative relaxation remains a cost. [NoProp](https://arxiv.org/html/2503.24322v2): promising through Tiny ImageNet, with architecture differences and repeated denoising.
- [Synthetic gradients](https://arxiv.org/html/1608.05343v2): decouple updates, but gradient-target training can retain backward dependencies; not automatically compliant.

Your local method is only **1.14× faster** here. Interestingly, BP worsens the first layer’s own test loss, **1.644→1.757**, while improving the final layer—evidence consistent with greedy objectives sacrificing useful downstream representations.

**2. Ranked experiments.** All ranges below are **subjective forecasts**, not confidence intervals: validation improvement versus 1.4564; runtime relative to local. Regressions remain possible. None reliably promises most of 0.069.

| Rank | Single change | Gain; cost |
|---|---|---|
| 1 | Add small nonlinear, training-only auxiliary predictors at lower layers; retain existing inference readouts. Let representations support nonlinear prediction rather than forcing immediate linear separability. | 0.010–0.040; 1.10–1.30× |
| 2 | After warmup, add KL distillation from detached top-layer predictions into lower readouts. Generate targets with an extra forward pass, then stream local updates; no BP-trained teacher. | 0–0.025; 1.30–1.60× |
| 3 | Add copy-aware local CE: mix each layer’s prediction with frozen copy distributions/gate weights. Retain original CE. | 0–0.015; 1.00–1.10× |

First/third preserve streaming local updates; second trades compute for bounded activation storage. Avoid repeating iter25’s failed residual objective.

For **each**, keep backbone, sampler, schedule and initialization matched; compare against iter30 BP **and BP with the identical auxiliary objective/head**. Add attention+BP first: remove `_layer`’s hard-coded detach. Attention’s 0.046 gain is architectural, not recovered credit assignment. Reset memory peaks after startup checks; current peaks include checks/evaluation.

**3. Best reframe:** architecture efficiency with backprop versus an optimized standard transformer, equal data/tokenization/tuning budget. Offer both copy mechanisms; measure loss versus FLOPs and elapsed time, plus time-to-target. This retains “faster AND better,” but explicitly relinquishes “without cross-layer backprop.”

**4. Minimum screening evidence:** 25M then 100M **distinct**, deduplicated characters; three versus eight layers; context 512. Run seed-zero paired screens sequentially on the shared GPU; replicate only the finalist across three seeds. Use fresh repository-disjoint holdouts, resumable jobs, and matched compute curves.

Suggested commitment gate: ≤0.01 local/BP loss gap without widening with depth/data, plus ≥20% lower time-to-target. That warrants further research—not a large-scale victory claim.