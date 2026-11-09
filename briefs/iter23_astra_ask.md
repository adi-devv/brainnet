Consult only: do not edit, create or run anything. Quota is tight, so read only `briefs/iter22_astra_out.md`, `iter22.py` lines 1-75 and `iter23.py` lines 1-45. Answer in under 350 words.

Context: this repo researches a transformer-like architecture that learns without cross-layer backprop and is both faster and better than backprop. Each layer learns from its own local loss, with updates during the forward sweep. Within-layer exact gradients are allowed. Budget: under 90 s on a GTX 1650, 4 GB. The review in `briefs/iter22_astra_out.md` prescribed iter 23; "you" means that review.

We built it as specified. All checks passed, including finite differences on the combined objective, and reduction to iter 22 with weights (1, 0). It failed the test bar, so your kill rule applies.

All runs: 6.2M chars, one pass, seed 0. Loss in nats/char. One seed each; backprop's seed spread on test is about 0.006.

| Run | Time | Test | Renamed | Test, no copy heads | Per-layer test | `g` |
|---|---|---|---|---|---|---|
| Backprop GPT | 180 s | 1.329 | 1.6675 | | | |
| Backprop, same one-pass sampler | 181 s | 1.3407 | 1.6863 | | | |
| Iter 22 | 68.5 s | 1.3568 | 1.6462 | 1.4343 | 1.6197, 1.4764, 1.4487 | -.032, .340, .708 |
| Iter 23 (0.75 next / 0.25 two-ahead on layers 0, 1) | 77.4 s | 1.3530 | 1.6430 | 1.4289 | 1.6166, 1.4695, 1.4419 | -.042, .337, .720 |
| Bars | at most 90 s | at most 1.329 | at most 1.667 | | | |

Iter 23 gained 0.004 on test for 9 s, about seed-noise size. Validation 1.4937 (iter 22 1.4980).

Running now, separately: iter 24 is iter 22 without the auxiliary, plus network LR 1e-3 for 75% of steps, then cosine to 1e-4; gate LR fixed. Do not wait for it.

Questions:
1. What does the auxiliary's near-zero effect tell you about the remaining 0.024? Be specific about what it rules out and what it does not.
2. Assuming LR decay does not by itself reach 1.329, name the one next experiment with the best chance. Consider final width 320, deeper or wider stacks, your cached upper-layer logit predictor, or something none of us has listed. Give the exact configuration and a pass or kill rule.
3. Should a marginal gain like 0.004 for 9 s be kept and stacked later, or dropped? Give a rule.
4. Is beating the 1.329 test bar inside this budget realistic for this family? Give your odds. If it is not, say what you would change about the approach.

Be direct. If something we did was a mistake, say so.
