**Run seeds 1 and 2. Then test final-layer width 320.** The gates look useful; saturation is a plausible limitation, not yet a diagnosis. All odds below are subjective research estimates.

**1. Iter 26: replicate, don’t kill yet.**

- Iter 24’s validation sample SD is **0.00126**; the paired seed-0 gain is **0.00853**. That strongly suggests a real improvement, although baseline variance does **not** establish variance of the paired gain.
- My odds: **90%** that mean validation improves; **20–30%** that it clears the full ≥0.010 improvement and <90 s rule.
- Seeds 1 and 2 must average **≥0.01074 improvement** to clear the three-seed bar. Seed 0 alone has not failed a rule defined on the three-seed mean.
- Keep the threshold unchanged. If the mean misses, retain the finding but don’t promote this configuration.

The same-seed reruns differ by only **0.000072 validation**, supporting numerical repeatability, not seed robustness. Also, the supplied JSON records **91.44 s**, not 82.5 s; I accept your newer back-to-back measurement, but the saved artifact does not document it.

**2. Saturation: worth one cheap experiment, not an emergency fix.**

Iter 24 uses the same unnormalized tanh upper layers, but **doesn’t record saturation or save trained weights** in this runner. Its trained saturation cannot be recovered from code alone.

At `|h| > .99`, tanh’s derivative is below **0.0199**. That attenuates local representation updates, but readouts still learn, and these derivatives aren’t multiplied across layers. Endpoint saturation also doesn’t tell us whether units were saturated throughout training.

The gates amplify the fastest EMA by about **2.02× / 1.82× on average**, which could contribute; learned projection weights can compensate. Don’t infer causation from those means.

Cheap intervention: upper layers only,

`h = 0.95*tanh(a) + 0.05*a`

with exact derivative `0.95*(1-tanh(a)^2) + 0.05`. Keep the original tanh value for this derivative. This preserves slope 1 near zero and supplies a gradient floor, but introduces unbounded activations. **About 30% odds of ≥0.005 validation improvement** over its unchanged parent. Measure validation saturation and derivative statistics during already planned runs; reduced saturation alone is not success.

**3. Ranked next experiments after replication**

Forecasts below are validation reductions against iter 24, with unchanged data and one pass.

| Rank | Configuration | Plausible gain | Odds of ≥0.010 gain and <90 s |
|---|---|---:|---:|
| **1** | **Iter 24, widths 256/256/320** | 0.005–0.015 | **35%** |
| 2 | Iter 24, upper-layer activation above | −0.010 to +0.015 | **20%** |
| 3 | Iter 24, all layers width 320 | 0.005–0.025 | **10%**, primarily runtime risk |

These are uncertain forecasts, not confidence intervals.

**Concrete next spec:** start from iter 24; widen only layer 2. `W2` becomes **1536×320**, `b2` becomes **320**, and `R2` becomes **320×V**, with matching optimizer states. Preserve lower-layer initialization. Use existing initialization rules, including zero-initialized lag rows and readout scaling by `1/sqrt(320)`. Keep independent local losses, learned logit mixing, copy heads, schedule, sampler, batch size and context unchanged.

My earlier **1.56× compute argument was wrong for final-only widening**: its main projection grows **1.25×**, while lower layers stay unchanged. That makes this a better budget bet than my previous ranking suggested. Rough runtime expectation: **83–89 s** from your 78.2 s base, unmeasured.

**Kill rule:** reject seed 0 if validation does not improve or runtime reaches 90 s. Otherwise complete seeds 1–2; promote only if every run is <90 s and mean paired validation improvement is ≥0.010.

Iter 26 already shows a promising numerical advantage over the reported backprop results. Establishing “better” still needs matched validation comparisons; the repeatedly consulted test set should remain descriptive.