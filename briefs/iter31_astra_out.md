**Run iter29 seeds 1–2 unchanged. Then prioritize the backprop control over another architectural tweak.** The largest correction: the existing “16M” result is repeated sampling, not one-pass data scaling.

All odds and forecast ranges below are subjective planning estimates.

1. **The +12 s is acceptable; don’t optimize before replication.**

   You retain approximately **2.06× training speed versus the recorded backprop baseline**, with a reproducible seed-0 validation improvement of **0.00779 nats/char**. Two additional seeds cost about **3 training minutes**.

   Estimate: **70–80% probability** of passing the three-seed quality rule—mean paired improvement ≥0.005 and every seed positive. Expected mean improvement: **0.005–0.009**. Compare against each seed’s actual iter27 result.

   The timing margin is fragile: the adjacent pair differences are **+21.45 s and +2.85 s**. The two-run median passes the 90 s rule, but **2.35 s headroom is not reliable operating margin**.

   For the exact optimization bundle, my planning estimate is **3–7 s saved**, giving roughly **81–85 s under these conditions**, with substantial uncertainty and possible negligible savings. Consolidating projections preserves leading FLOPs; cached `pi` and fewer synchronizations save overhead. About **70% odds of a measurable improvement**, not a guaranteed runtime. Optimize after quality replication; apply shared optimizer improvements to the control too, and recheck the final implementation’s quality.

2. **Next three steps, ranked by value per GPU minute:**

   | Rank | Experiment | Cost and expected value |
   |---|---|---|
   | **1** | Complete iter29 seeds 1–2 | ~3 training minutes; resolves whether the current gain generalizes across seeds. Highest immediate value. |
   | **2** | Matched-sampler backprop: existing schedule versus hold+cosine; log validation curves for both and the local winner | ~6 training minutes for two backprop seed-0 runs, plus ~1.5 minutes for a local curve run. Highest value for establishing a credible quality win. |
   | **3** | One capacity probe: **256/256/384 on iter27**, initially without EMA gates | Roughly **80–88 s** projected from the controlled iter27 timing. Estimate **50–60% odds** of ≥0.003 validation improvement; plausible gain **0.003–0.010**, with regressions possible. Tests whether width buys more quality per second than gates. |

   For **rank 2**, backprop already has warmup plus cosine. Test **warmup → hold until 75% → cosine**, retaining its existing peak LR, floor, optimizer and batch size. Match schedule progress by fraction of training characters; copying local’s absolute LR would confound the test. Retain whichever schedule wins validation.

   I assign roughly **50–60% odds** that this improves backprop by ≥0.005 validation loss; **0–0.02** is a plausible benefit, and worsening is possible. That is enough to threaten the current lead: the recorded matched-sampler backprop validation is **1.46396**, only **0.01535** behind iter29 seed 0.

   Log **fixed validation windows at common character counts**, with synchronized training time excluding evaluation and evaluation time reported separately. This establishes sample-efficiency trajectories; intermediate checkpoints do not establish optimal performance for separately scheduled shorter budgets.

   Among learning-rule ideas, I would next investigate **local losses that account for the copy mixture**, after these controls. That directly targets the mismatch between training network-only predictions and evaluating a mixture. It is more speculative than widening: roughly **25–35% odds of ≥0.005 improvement**, perhaps **0.01–0.02 if successful**. Keep every derivative within its layer. The evidence does not justify prioritizing another residual-objective or saturation fix.

3. **What the evidence currently overstates:**

   - **16M is outside the stated experiment.** [baseline.py](C:/Users/aadit/AppData/Local/Temp/claude/2e162b2224a69697/71efcaa5-cf65-4cbc-90fa-3206306ba725/scratchpad/bn/baseline.py:35) samples windows with replacement from **6.54M characters**. A 16M exposure run increases budget and repeats data. Defer it under your fixed-data, one-pass constraint; genuine 16M distinct-data scaling requires a new dataset protocol.
   - **The old baseline uses your local validation suffix during training.** Use iter18’s restricted partition and sampler for the new comparison. Its historical test score remains informative, but evaluating its model on local validation would not produce a held-out result.
   - **The quality advantage is currently a development-set result.** README correctly acknowledges reused test data. Renamed loss is another transformation of that same test, and its backprop reference has only one measured seed.
   - **This compares complete systems, not learning rules alone.** Copy heads, architecture, batch size and optimizer differ. Offering backprop the same copy mechanism is a valuable subsequent control; isolating local learning itself requires a matched-architecture comparison.
   - **Iter29’s improvement survives disabling copy:** seed-0 network-only test loss improves by about **0.007**. That supports a network improvement rather than merely a changed copy mixture.

No files changed or training run.