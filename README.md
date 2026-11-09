# brainnet

Notes and experiments on neural networks that learn **without gradient descent**.

Nothing here is proven. This is a working list of pointers and the approaches being tried, with honest status on each.

## Pointers

- The brain has no global backward pass. Synapses change from local signals, and a reward or surprise signal decides which changes stick.
- New experiences are stored fast (hippocampus-like), then replayed in sleep to slowly train long-term weights (cortex-like).
- Gradient descent needs a differentiable path from the result to the weights. A hard pass/fail reward has none, so search or local rules are needed.
- The price of going gradient-free is sample efficiency. Expect many more trials per useful update.
- Results so far (tiny character-level code model, GTX 1650): backprop-free count tables already beat a backprop transformer at the same data; see Findings.

## Approaches

| Approach | Idea | Status |
|---|---|---|
| Evolution strategies | Copy the weights with random noise, score each copy by reward, move toward the best | Tried on Sudoku with a 0/1 reward |
| Meta-evolution | Evolve the learner's architecture and hyperparameters across lives so each new life learns faster | Tried on Sudoku together with the above |
| Hebbian and three-factor rules | Local "fire together, wire together" updates, gated by a reward or surprise signal | Not started |
| Predictive coding, forward-forward | Layer-local error signals instead of one global backward pass | Not started |
| Closed-form memory writes | Write a fact straight into the weights with a rank-one solve (MEMIT/AlphaEdit style), no training loop | Idea only; still needs a few gradient steps to find the target value |

## Findings (iterations 1-15, character-level code LM, test loss in nats/char, lower is better)

All backprop-free rows use hand-written updates with grad mode off. Seed 0 unless noted.

| Method | 1M chars | 6.2M chars | Renamed-identifier test | Time |
|---|---|---|---|---|
| Backprop GPT (baseline) | 2.330 | 1.329 / 1.352 (2 seeds) | 1.530 at 16M (gap 0.406) | ~181 s at 6.2M |
| DFA stacked learner (iter 14: fixed causal EMAs, 6 lag lookups, random feedback) | 1.992 | 1.549 | 1.862 at 6.2M (gap 0.313) | 69.5 s at 6.2M (2.6x faster) |
| Exact local gradient (iter 15 `local`) | 1.909 | not run | 2.154 at 1M (gap 0.245) | 8.5 s at 1M |
| Count tables + learned pointer heads (iter 12) | 1.268 | **1.165** | 1.606 at 6.2M (gap 0.441) | 244 s at 6.2M |
| Tables + local net (iter 15 `hybrid`) | 1.261 | not run | 1.620 at 1M (gap 0.359) | 101 s at 1M |

Backprop at 16M chars: 1.124 / 1.139.

What this says:
- At equal data (6.2M), backprop-free tables beat backprop (1.165 vs 1.329). They are one-shot memories, so they copy repeated new names well (loss 3.2 on a repeat vs 5.9 on first sight).
- The tables do not generalize across names: renamed gap 0.37-0.44, versus 0.31 for DFA and 0.41 for backprop at 16M.
- The DFA net is about 2.6x cheaper per step than backprop but loses on quality at 6.2M. It wins only in the first ~1M chars.
- Exact layer-local gradient beats random feedback at 1M (1.909 vs 1.992).
- Adding the local net to the tables gains little at 1M (0.007 test, 0.015 renamed), below the pre-set bar for a full run.
- Open: a rule that learns what tables cannot (cross-name generalization), noise-based local rules, prospective configuration.

Code is in `iterN.py`, briefs in `briefs/`, raw numbers in `results/`.
