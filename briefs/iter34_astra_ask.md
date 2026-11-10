# Astra: top-down teaching failed — why, what next, and can we predict instead of run? (read-only advice)

Repo: this folder. Read `iter33.py`, `iter30.py`, `briefs/iter33_topdown_brief.md`, `briefs/iter32_astra_out.md`, `results/iter33_topdown.json`, `results/iter30_{local,backprop}.json`.

## Goal (unchanged)
A transformer-like architecture that learns WITHOUT cross-layer backprop (the brain doesn't do backprop) and is faster AND better than backprop at LARGE scale. Do not reframe to "our arch trained with BP".

## Result (seed 0, same iter27 architecture)
iter33 top-down teaching: lower layers 0,1 train on (1-a)*CE + a*KL(p_top_detached || p_l), a ramped 0->0.5 over 5%-20% of steps, same-step, one forward pass. All correctness checks pass; alpha=0 reproduces iter30 local exactly.

| mode | val | test | train s |
|---|---|---|---|
| local | 1.4564 | 1.3231 | 70 |
| topdown a=0.5 | 1.5596 | 1.4109 | 71 |
| backprop | 1.3869 | 1.2562 | 79 |

Curve (val subset every 10%): 1.885 1.836 1.758 1.687 1.641 1.612 1.557 1.545 1.475 1.452 (local: 1.860 1.720 1.622 1.561 1.512 1.510 1.434 1.417 1.385 1.361). Final mean KL, last 10%: layer0 0.258, layer1 0.103.

Our hypothesis: KL to the top's *output* is still a greedy target. It tells layer l to mimic the answer, not to build features layer 2 needs, and it fights the CE term.

## Questions (concise, <450 words; label subjective numbers)
1. Why did it hurt this much (+0.10)? Was this predictable before running? Which mechanism dominates?
2. User's question: do we really need to run experiments, or can outcomes like this be known by mentally simulating / reasoning, or by cheap proxies (tiny-scale runs, gradient-alignment measurements such as cosine between a local update and the true BP gradient on a few batches, linear probes)? Give a concrete cheap protocol to screen ideas in seconds before a full 70 s run, and say how reliable such predictions are.
3. Best next single experiment toward matching BP without cross-layer derivatives, scalable. Candidates: target propagation / difference target propagation with learned feedback (inverse) weights; feedback alignment of the top error to lower layers; nonlinear training-only auxiliary heads (your iter32 rank 1); forward-gradient / activity-perturbation; others. Rank top 2 with predicted val vs 1.4564 and which cheap proxy (from Q2) to check first.

Do not change files or run training.
