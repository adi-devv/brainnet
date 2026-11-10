**Run the unchanged same-session timing control first. Keep iter 29 alive; optimize if the slowdown reproduces.** Its exact seed-0 validation gain is **0.0075933**. Odds below are subjective estimates.

1. **The overhead is real, but +36.4 s is probably inflated.**

   Ranked suspects:

   | Rank | Code | Added cost |
   |---|---|---|
   | 1 | [iter29.py:43](/C:/Users/aadit/AppData/Local/Temp/claude/2e162b2224a69697/71efcaa5-cf65-4cbc-90fa-3206306ba725/scratchpad/bn/iter29.py:43) | Three branch projections per upper layer, followed by elementwise products and reductions. |
   | 2 | [iter22.py:60](/C:/Users/aadit/AppData/Local/Temp/claude/2e162b2224a69697/71efcaa5-cf65-4cbc-90fa-3206306ba725/scratchpad/bn/iter22.py:60), reached through iter29 lines 48–54 | Four extra parameter updates, each with a CUDA-dependent Python assertion: **four extra host synchronization points per batch**, plus small Adam kernels. |
   | 3 | iter29 lines 34, 44–46 | Gate scaling/materialization, recomputing `pi`, gate gradients. The large `cat` already exists in iter27; stacking three scalar scores is small. |

   The projections add approximately
   \[
   2(2048)(3)(256)(256+320)=1.812\text{ GFLOP/batch},
   \]
   or **5.50 TFLOP per pass**. That is **25% extra relative to the upper layers’ combined forward-W and W-gradient multiplication work**, less relative to the whole model. Synchronization can also materially affect runtime. [PyTorch guidance](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization)

   Crucially, those added projection FLOPs are only **12.5% above iter26’s**, not several times larger. The observed difference is **12 ms/batch**: possible through inefficient kernels/launches/synchronization, but not explained by width scaling alone.

   **Working estimate: +5–18 s under comparable conditions; about 70% odds the controlled increment is below +20 s.** This is a forecast, not a measurement. Recorded iter27 times already span **70.9–88.7 s**; iter26’s saved references imply **+12.3–25.9 s**, excluding its explicitly slow run. Your reported back-to-back +4 s is stronger timing evidence than those unmatched references. Post-run idle clocks don’t establish clocks during training. Startup checks and evaluation are outside the training timer.

2. **First optimization: consolidate the gate-gradient projections; retain the existing forward and W gradient.**

   Replace line 43 with this algebra, flattening batch/time:

   ```python
   U = u.flatten(0, 1)          # [N, 1536]
   D = delta.flatten(0, 1)      # [N, H]
   v = D @ w[256:1024].T        # [N, 768], one GEMM
   s = (
       U[:, 256:1024].reshape(-1, 3, 256)
       * v.reshape(-1, 3, 256)
   ).sum(-1)
   logit = s - pi.reshape(-1, 3) * s.sum(-1, keepdim=True)
   ```

   This uses
   \[
   \langle\delta,u_kW_k\rangle=\langle u_k,\delta W_k^\top\rangle.
   \]
   It replaces **six projections with two larger GEMMs**, retaining the same leading FLOP count. `v` is strictly local; never propagate it into the lower layer.

   Also cache the forward `pi` in an explicit per-call training context, and combine finite checks into **one check per layer before updates**, retaining separate clipping norms and gate LR. Apply shared optimizer changes to the iter27 timing control too.

   **Do not try to recover position-specific scores from `dW = U.T @ D`: that sum has discarded position information.** It supplies only aggregated contractions, insufficient for content-dependent gate gradients.

   If projections still dominate, the next exact implementation is **forward branch reuse**: compute six block projections with a batched GEMM, sum them for the preactivation, and retain the three EMA contributions for `s = sum(delta * contribution)`. This removes the redundant gradient projections, but adds intermediate traffic and may reduce GEMM efficiency. Benchmark before adopting; avoiding gated-block materialization is a later, more invasive change.

   **Verification:** extend the existing float64 local-autograd check at iter29 lines 107–145 to compare original/optimized gradients, clipped gradients, Adam moments and updates, using active gates, both widths and masked targets. Retain `atol=1e-12, rtol=1e-10` for float64 gradients; separately compare float32 behavior with fixed precision settings. These are algebraically identical transformations, **not guaranteed bitwise-identical floating-point computations**. [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.15/notes/numerical_accuracy.html)

   **About 70% odds the first optimization bundle measurably helps; no basis yet to promise a 17-second saving.**

3. **Decision order: control → optimize if needed → seeds 1–2 → promote or stop.**

   Use an **AB/BA same-session comparison** of unchanged iter27/29, with fresh models, matched warm-up, AC/performance mode and comparable temperatures. Record synchronized whole-training time plus clocks/power during training. Keep diagnostic runs separate from quality evidence.

   If controlled timing fits, complete seeds 1–2. If it clearly exceeds 90 s, allow one bounded implementation optimization attempt before stopping this candidate.

   Promote only when **mean paired validation gain ≥0.005, all three seeds improve, and controlled median candidate time ≤90 s**. Use each seed’s actual iter27 JSON: iter29’s hard-coded references and `pass` field do **not** implement this rule. If optimization changes floating-point execution, confirm quality with that final implementation.

   My current odds: **~75% of passing the quality rule; ~55% of passing quality and timing together**, allowing the optimization attempt.

No files changed or training run.