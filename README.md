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

Current direction and pass/kill bars: [PLAN.md](PLAN.md).

## Findings (iterations 1-18, character-level code LM, test loss in nats/char, lower is better)

All backprop-free rows use hand-written updates with grad mode off. Seed 0 unless noted.

| Method | 1M chars | 6.2M chars | Renamed-identifier test | Time |
|---|---|---|---|---|
| Backprop GPT (baseline) | 2.330 | 1.329 / 1.335 (2 seeds) | 1.667 at 6.2M (gap 0.339); 1.530 at 16M (gap 0.406) | ~181 s at 6.2M, 321 MB |
| Backprop GPT, same one-pass windows as `local` (iter 18 control) | - | 1.341 | 1.686 at 6.2M (gap 0.346) | 181 s at 6.2M, 319 MB |
| DFA stacked learner (iter 14: fixed causal EMAs, 6 lag lookups, random feedback) | 1.992 | 1.549 | 1.862 at 6.2M (gap 0.313) | 69.5 s at 6.2M (2.6x faster) |
| Exact layer-local gradient (iter 15 `local`) | 1.909 | 1.492 | 1.819 at 6.2M (gap 0.327) | 66-73 s at 6.2M, 105 MB |
| `local` + suffix-match copy heads (iter 18) | - | 1.411 | 1.697 at 6.2M (gap 0.286) | 74.5 s at 6.2M, 103 MB |
| Count tables + learned pointer heads (iter 12) | 1.268 | **1.165** | 1.606 at 6.2M (gap 0.441) | 244 s at 6.2M, ~1.9 GB |
| Tables + local net (iter 15 `hybrid`) | 1.261 | 1.155 | 1.596 at 6.2M (gap 0.441) | 707 s at 6.2M, ~2 GB |

Backprop at 16M chars: 1.124 / 1.139.

What this says:
- At equal data (6.2M), backprop-free tables beat backprop (1.165 vs 1.329). They are one-shot memories, so they copy repeated new names well (loss 3.2 on a repeat vs 5.9 on first sight).
- The tables have a large renamed gap (0.44 at 6.2M), but their absolute renamed loss still beats backprop at equal data (1.606 vs 1.667). An earlier version of this note said they "do not generalize", which overstated it. They cost about 6x backprop's memory and more time.
- The DFA net is about 2.6x cheaper per step than backprop but loses on quality at 6.2M. It wins only in the first ~1M chars.
- Exact layer-local gradient beats random feedback at 1M (1.909 vs 1.992).
- Adding the local net to the tables gains little (0.010 test at 6.2M) and takes about 3x longer.
- `local` is about 2.5x faster than backprop per char but worse at equal data (1.492 vs 1.329). More passes in equal time do not close it (2.5 passes, 163 s: 1.353 test, 1.720 renamed).
- `local` does avoid backprop across layers, but it still uses the exact chain rule inside each layer. It also has no attention, so its speedup is partly from a smaller model.
- Copy heads (iter 18) add 0.08 test at 2.4x backprop's speed. Renamed loss nearly reaches backprop on the same sampling (1.697 vs 1.686), and repeated new names are far cheaper (1.66 vs 5.38 nats). Test loss still misses (1.411 vs 1.341).
- Open: see PLAN.md.

Code is in `iterN.py`, briefs in `briefs/`, raw numbers in `results/`.
