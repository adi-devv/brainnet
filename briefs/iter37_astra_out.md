**1. A real small-scale result, not yet a large-scale win.** Tables improves test by 0.0337 nats/char against iter30’s 1.14M-parameter, attention-free BP control. One seed and repeatedly consulted test data limit confidence. Repeat loss remains worse than BP: **2.61 versus 1.82**. Keys’ early 1.337 is **validation-subset** loss; tables already reaches 1.339 there.

Literature supports memory’s durability, but not BP-free superiority: [billion-word results](https://arxiv.org/abs/1312.3005) favor neural models over Kneser–Ney; [infini-gram](https://arxiv.org/html/2401.17377v2) gains increase with datastore size but generally shrink with neural-model size. [kNN-LM](https://arxiv.org/abs/1911.00172) uses BP-trained representations. [Product keys](https://arxiv.org/abs/1907.05242), [memory layers](https://arxiv.org/abs/2412.09764), and [Engram](https://arxiv.org/html/2601.07372v1) demonstrate scalable memory **inside BP-trained systems**. None establishes your intended result.

**2. Runtime is dominated by work the FLOP estimate omits.** JSON confirms **28,980 FLOPs/char: 248× below BP**. Likely costs: vocabulary-wide hash/gathers, collision handling, atomics, intermediate tensors, Python/kernel launches, transfers, and synchronization. Crucially, per-table `unique` staleness diagnostics run every chunk inside training timing. Evaluations are excluded; including recorded evaluations reduces the advantage to **1.04×**.

Profile first; sample diagnostics, remove per-chunk host synchronization, fuse lookup/backoff/mixing, and pipeline transfers. Memory bandwidth will still prevent FLOP-proportional speedups. **Subjective forecast:** 2–5× faster training than current tables after substantial optimization. No measured ceiling exists; unchanged evaluations alone cap train-plus-eval speedup versus recorded BP at approximately **4.7×**.

**3. Cheapest decisive scaling screen:** use existing 10/30/100% checkpoints for triage, then independently train on nested **10M/30M/100M characters** from [The Stack’s deduplicated Python subset](https://huggingface.co/datasets/bigcode/the-stack-dedup). Use fresh repository-disjoint, cross-split-deduplicated holdouts, frozen evaluation counts, identical character encoding/context, matched iter30 BP, and approximately **10M-parameter transformer BP**. Tune schedules per budget; compare equal-data and equal-time results. Replicate endpoints across three seeds. This can reject a scaling hypothesis; surviving warrants 1B characters, not a large-scale victory claim.

**4. Scale storage with distinct useful context–target pairs**, measuring eviction, hit rate, and retained probability mass. Sweep 1×/4×/16× capacity; fixed direct-mapped caches eventually forget through replacement. Role-based prediction plus name binding offers structural generalization. Online prototype clustering or locally learned similarity could extend this, but drifting keys require stable/versioned addresses or reindexing. Raw-character CE on anonymized inputs is poorly aligned with relational similarity. The 0.9998 staleness figure mostly reflects the shared unigram—not catastrophic long-context staleness.

**5. Ranked experiments.** Forecasts are subjective; thresholds are proposed decisions.

| Rank | Experiment | Prediction | Kill criterion |
|---|---|---|---|
| 1 | Scaling screen plus capacity sweep | BP closes the gap | At 100M, expanded memory still loses quality with a flatter improvement curve |
| 2 | iter36 roles versus literal control | Renamed gain 0.01–0.04 | Gain <0.01, ordinary loss worsens >0.005, or added time erases advantage |
| 3 | Profile and fuse tables | Training speedup 2–5× | Stop optimization branch below 1.5× after measured bottlenecks are addressed |

No files changed or training run.
