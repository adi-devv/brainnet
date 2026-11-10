# Astra: how do we beat backprop at scale without backprop? (advice only, write no code)

Read README.md, PLAN.md, briefs/iter37_astra_out.md and iter40.py in this repo.

Goal: a learning system with NO backprop (local or one-shot learning only, no global backward pass) that beats backprop-trained transformers on loss, wall time and compute AT LARGE SCALE: as data and model size grow, its loss-vs-compute curve has to stay below backprop's. Small-data wins alone do not count.

Where we are (char-level code LM, 6.2M chars, one pass, GTX 1650):
- BP reference (same layered arch with attention): test 1.256, renamed-identifier 1.571, 79 s, 151 MiB, 7.2M FLOPs/char. BP at 16M chars: 1.124.
- Best backprop-free system, count tables (hashed contexts, one-shot count writes, backoff, small local mixer): test 1.171, renamed 1.545, 67 s, about 29k FLOPs/char. Bigger tables help (1x 1.223 -> 16x 1.177). Generalization to renamed code is weak.
- Local layer-wise learning (no cross-layer backprop) + attention reached 1.291; DFA, top-down teaching and auxiliary heads lost.

Compute available: local GTX 1650 4 GB, plus Kaggle T4/P100 (about 30 GPU-h/week, 9 h per session).

Questions (give your own view, not a review of ours):
1. Count tables are known to lose to neural nets as data grows. What backprop-free approach do you believe can actually match backprop's scaling, and why (mechanism, evidence from the literature)? Rank the 2-3 best candidates.
2. For the top candidate: the concrete next 3 experiments, each with the hypothesis, the setup, the pass/kill bar against a compute-matched BP baseline, and whether it runs on Kaggle or locally.
3. How to measure "beats BP at scale" honestly: data/model sizes, compute matching, and fitting the scaling curves.
4. What in our current line of work (tables, local layers) to keep, and what to drop.
Under 1000 words.
