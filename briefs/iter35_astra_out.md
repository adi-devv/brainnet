**1. Stop iterating on local targets; retain local learning as an architectural constraint.** My earlier auxiliary-head recommendation was too optimistic.

These runs establish failed interventions, not a 1.45 ceiling: local attention already reaches **1.4104**, although its recorded 86.7 seconds exceeds BP’s 79.4. The matched local/BP speed advantage is only **1.14×**. BP worsening layer-0 prediction while improving downstream performance supports a representation-coordination problem; better individual predictors needn’t produce better features.

Depth/width scaling remains unmeasured. **My hypothesis:** increasing depth tends to enlarge the greedy-learning penalty; width may relieve bottlenecks, but needn’t close the gap against equally widened BP. [DGL’s ImageNet results](https://proceedings.mlr.press/v119/belilovsky20a.html) show locality is viable, not a universal ceiling or a large-language-model victory.

**2. Prioritize architecture and sparse memory; give DFA one bounded trial.** My best bet is increasing independently learnable capacity while limiting sequential depth and active computation. Your table/copy results support this direction, although their memory and runtime costs remain unresolved.

Defer [DTP’s inverse-learning difficulties](https://arxiv.org/abs/2006.14331), [forward-gradient variance](https://mengyeren.com/research/2023/scaling-forward-gradient-with-local-losses/), and [equilibrium](https://arxiv.org/abs/1602.05179)/[predictive-coding relaxation](https://proceedings.mlr.press/v306/qi26f.html). They offer no demonstrated speed-and-quality advantage here. Even [billion-parameter optical DFA](https://arxiv.org/html/2409.12965v1) used only 1.2M tokens and context 24; optical training took approximately 20 hours versus BP’s two. Parameter count alone isn’t the required scaling evidence.

**3. Ranked experiments.** Forecasts are **subjective plausible ranges**, not confidence intervals; thresholds are proposed decisions. Current references: local **1.4564**, BP **1.3869**.

| Rank | Mechanism | Predicted val | First proxy | Kill criterion |
|---|---|---|---|---|
| 1 | Two-layer local learner: allocate saved layer compute to width and top-layer causal attention. | 1.39–1.48 | `mixed` at matched compute; architecture-specific runs from initialization | No loss/time Pareto improvement over existing local and attention controls. |
| 2 | Bounded hashed context→next-token memory, causal writes after scoring, locally learned mixture gate. | 1.37–1.49 | `mixed`; inspect novel-repeat versus first-occurrence losses | Less than 0.01 full-run gain, or lookup/write costs erase time-to-target benefit. |
| 3 | Local CE plus small, norm-controlled fixed random projection of top error into lower updates. | 1.43–1.54 | `mixed` after adaptation; actual Adam displacement diagnostic | Less than 0.01 replicated gain after one bounded coefficient sweep. |

None reliably promises beating BP. Advance survivors to larger distinct datasets and matched BP controls; BP remains the comparator.

**4. Screening is useful triage, not calibrated selection.** Two correct rejection signs establish little sensitivity to eventual winners. Keep 64 steps for obvious damage; use **256 steps at 15% and 60%**, preserving schedules, for survivors. Evaluate **256 windows distributed across validation**, using paired document/block uncertainty.

Promote on `mixed`; retain `mixed_nocopy` and top loss as diagnostics. Replace raw-gradient cosine with clipped Adam displacement measurements. Checkpoint switching tests adaptation, so DFA and architectural changes also need initialization-based runs. Replicate finalists across three seeds and reserve fresh repository-disjoint evaluation data.

No files changed or training run.