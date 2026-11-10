# Astra: review the current approach and say what's next (read-only advice)

Repo: this folder. Read `README.md`, `iter27.py`, `iter30.py`, `iter33.py`, `iter34.py`, `screen34.py`, `briefs/iter3{0,1,2,4}_astra_out.md`, `results/iter30_{local,backprop,attn}.json`, `results/iter33_topdown.json`, `results/iter34_auxhead.json`, `results/screen34_{auxhead,topdown}.json`.

## Goal (unchanged)
A transformer-like architecture that learns WITHOUT backprop (the brain doesn't do it) and is faster AND better than backprop at LARGE scale. Do not reframe to "our arch trained with BP".

## Where we are (seed 0, iter27 arch: 3 layers, widths 256/256/320, layer-local CE, inputs detached, EMAs, copy heads, g logit mixing)
| mode | val | test |
|---|---|---|
| backprop | 1.3869 | 1.2562 |
| attn-local | 1.4104 | - |
| local | 1.4564 | 1.3231 |
| iter34 aux nonlinear heads (train-only) | 1.4924 | 1.3559 |
| iter33 top-down KL teaching | 1.5596 | 1.4109 |

iter34 per-layer test: 1.800/1.527/1.431 vs local 1.644/1.459/1.390 (all layers worse, including the top).
Fast screen (screen34, branch from 15% ckpt, 64 steps, ~8 s): mixed-loss delta vs local got the sign right twice (topdown +0.088, auxhead +0.015). The top-only delta misled for auxhead (-0.010).

Pattern: every "better local target" we tried has made results worse. The gap to BP stays about 0.07.

## Questions (concise, <500 words; label subjective numbers)
1. An honest verdict on the whole layer-local-loss line. Is this local-CE family capped near 1.45 at this scale? Does the BP gap grow or shrink with depth and width? What evidence from the literature and from our runs says so?
2. Should we keep pushing on credit assignment (DFA, which was your #2; DTP; forward-gradient; equilibrium or predictive coding), or change direction? Possible directions: an architecture where local learning is naturally enough (e.g. wider or shallower, mixture or fast-weight memory, more lookup-style capacity), or a different learning rule? Which direction has the best chance of being better than BP at LARGE scale, rather than just matching it at 3 layers?
3. Give the top 3 next experiments, ranked. For each: one-line mechanism, predicted val vs local 1.4564 and BP 1.3869, which screen34 proxy to check first, and its kill criterion.
4. Is our screening protocol good enough, or what should change (more steps, a ckpt at a later %, more windows)?

Do not change files or run training.
