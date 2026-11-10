**Keep iter 27 as the base; try iter 26’s gates on it next.** Its exact paired mean gain is **0.010044**: a narrow threshold pass, with consistent improvement across seeds. All odds below are subjective research estimates.

1. **Leaky tanh: useful negative evidence, but not a diagnosis.**

   Iter 28 changed **iter 24**, not iter 27. The derivative implementation passes independent autograd checks.

   My leading explanation is that it changed representation scale and learning dynamics substantially despite the “5%” coefficient. Recorded upper-layer maxima reach **1.89/1.99**, versus tanh’s bound of 1; top-layer mean absolute activation exceeds 1. Layer 1’s larger outputs also enter every top-layer context branch.

   The intervention simultaneously removed boundedness and introduced a derivative floor. It therefore cannot distinguish useful saturation from scale control or an optimizer interaction. Adam also means a larger derivative does not directly imply proportionally larger updates.

   **About 70% confidence** that bounded feature scale and/or saturation-induced stability contributes here. Much weaker evidence that saturation itself is beneficial. The defensible conclusion: **gradient starvation was not fixed profitably by this intervention.** Keep tanh; don’t spend another run rescuing it now.

2. **Next experiments, ranked.**

   Gains below are *additional validation reductions versus iter 27*. Odds mean achieving **≥0.005 mean gain within a controlled 90-second budget**.

   | Rank | Configuration | Plausible gain | Odds |
   |---|---|---:|---:|
   | 1 | Iter 27 + EMA gates at layers 1 and 2 | 0.004–0.010 | 60% |
   | 2 | Widths **256/256/384** | 0.003–0.009 | 45% |
   | 3 | Iter 27 + EMA gate at layer 1 only | 0.002–0.006 | 30% |
   | 4 | Widths **256/320/320** | 0.000–0.012 | 20% |

   **Concrete next spec:** retain iter 27 initialization, widths, tanh, schedule, copy heads and local losses. Add iter 26’s zero-initialized `G1,G2: 256×3` and `d1,d2: 3`; scale EMA branches by `3*softmax(lower@G+d)`; use exact local gate gradients and **0.1× network LR**.

   Why gates first: their gain replicated, and they improve layer 1’s own readout by roughly **0.014–0.016**, while final-only widening leaves that layer unchanged. That supports complementarity, although gains may overlap. My central forecast is **0.006**, with roughly **80% odds of any improvement**.

   Treat the reported **~4 s** overhead as provisional: gate training includes extra branch projections, whose top-layer cost grows with width. Middle-layer widening increases both its own projection and the following layer’s input width, making it less attractive.

   **Kill rule:** stop after seed 0 if paired validation gain is ≤0 or controlled timing clearly exceeds budget. Otherwise complete seeds 1–2. Promote under the prospective rule below; no test-based rescue.

3. **Change the timing protocol; revise the incremental quality bar prospectively.**

   Keep **90 seconds under specified operating conditions**. Battery-state outliers should invalidate timing, rather than kill an architecture.

   Use AC power, a fixed performance mode, comparable temperature and no competing GPU workload. Compare candidate and iter 27 **in the same session**, reversing order across seeds. Report synchronized whole-training wall time, paired runtime ratios and spread. Warm-up and CUDA synchronization matter; the current runners already synchronize around training. [PyTorch benchmarking guidance](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)

   GPU-event/block timings are useful diagnostics, but cannot replace wall time or remove throttling. Host work and synchronization belong in the actual cost.

   I would change future promotion to **mean paired gain ≥0.005, improvement in all three seeds, and controlled median time ≤90 s**, retaining **0.010 as a major-improvement milestone**. This explicitly revises my earlier advice: requiring another 0.010 at every step can reject worthwhile cumulative progress. Iter 27’s gain SD is **0.000834**; its 0.000044 clearance is bookkeeping precision, not evidence of a meaningful distinction from 0.0099.

4. **What you can claim—and the next evidence to buy.**

   Justified: **“On this fixed one-pass benchmark, the locally trained model achieves lower recorded test and renamed losses than the evaluated backprop baselines, with roughly 2–2.5× lower recorded training time.”** Iter 27 means are **1.32150 / 1.62109**.

   It uses **exact within-layer gradients and Adam, without cross-layer backprop**. The speed advantage includes architecture differences; this does not establish that removing cross-layer backprop causes the win.

   Prioritize remaining evidence:

   - **Matched validation and tuning:** identical training partition, exposure, evaluator and validation exclusion; give backprop a bounded LR/schedule search and three seeds. Schedule tuning was your largest gain.
   - **Matched architecture:** train the same EMA/lag/copy architecture with end-to-end gradients to isolate the learning-rule contribution.
   - **Time to common validation quality:** compare learning curves within the same one-pass ceiling, alongside final quality and controlled runtime.

   The repeatedly consulted test set remains development evidence. With the dataset fixed, use any genuinely untouched partition if one exists; otherwise acknowledge that fresh confirmatory generalization evidence is unavailable.

Read-only review completed; no files changed or training run.