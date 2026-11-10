# Astra: how to close the local-vs-backprop gap (read-only advice)

Repo: this folder. Read `iter30.py`, `iter27.py`, `results/iter30_{local,backprop,attn}.json`, `briefs/iter31_astra_out.md`, and the README.

## Goal (unchanged, but stricter)
A transformer-like architecture that learns WITHOUT cross-layer backprop and is faster AND better than backprop. The target is LARGE SCALE (much more data, longer context, bigger models), not 6.2M chars. Reject ideas expected to fail at scale (hand-built fixed structure that replaces learning, memory-hungry count tables). Prefer mechanisms that scale like transformers and backprop.

## New decisive results (seed 0, same iter27 architecture, same sampler/steps/schedule)
| mode | val | test | renamed | train s | peak MiB |
|---|---|---|---|---|---|
| local (iter27) | 1.4564 | 1.3231 | 1.6234 | 70 | 166 |
| backprop (same summed per-layer losses, gradients flow across layers) | 1.3869 | 1.2562 | 1.5714 | 79 | 151 |
| local + 1 causal attention block (4x64, RoPE) feeding layer 2, local learning | 1.4104 | 1.2913 | 1.5970 | 87 | 153 |

Val-subset curves (every 10% of training):
- local: 1.860 1.720 1.622 1.561 1.512 1.510 1.434 1.417 1.385 1.361
- backprop: 1.811 1.632 1.518 1.459 1.419 1.397 1.358 1.354 1.314 1.299
- attn: 1.845 1.678 1.575 1.513 1.466 1.460 1.399 1.381 1.335 1.313

Takeaways:
- Our earlier "beat backprop GPT" came from the architecture, not the learning rule.
- On the same architecture, local learning costs 0.069 val and is never more sample-efficient.
- Local attention is a real gain of 0.046.

## Questions (be concise; numbers are subjective estimates, label them so)
1. Is the goal still credible? Specifically: can any learning rule without cross-layer backprop plausibly match backprop at scale? Cite literature (forward-forward, DFA/feedback alignment, local-loss / greedy layerwise, PEPITA, signal propagation, predictive coding, NoProp, decoupled greedy learning, synthetic gradients, etc.) and their known scaling gaps.
2. Rank the top 3 single-iteration experiments for closing most of the 0.069 gap without full cross-layer backprop. Prefer ideas that keep layerwise parallelism or memory or speed advantages that grow with depth or scale. For each, give expected val gain, cost, and how to keep it honest: same architecture, compare against the backprop mode in iter30. Also consider adding attention to the backprop mode as the new reference.
3. If the honest answer is that local learning cannot win, what is the best reframe that still meets "faster AND better than backprop at scale"? For example, our architecture trained with backprop versus a standard transformer at equal compute.
4. What is the minimum evidence at larger scale (more distinct data, a deeper or wider model) needed before committing? Keep it cheap: one GPU, shared.

Do not change files or run training. Answer in under 500 words.
