Consult only: do not edit, create or run anything. Quota is tight, so read only `briefs/iter24_astra_out.md`, `iter22.py` lines 1-75 and `iter24.py` lines 1-30. Answer in under 350 words.

Context: this repo researches a transformer-like architecture that learns without cross-layer backprop and is both faster and better than backprop. Each layer learns from its own local loss, with updates during the forward sweep. Within-layer exact gradients are allowed. Budget: under 90 s on a GTX 1650, 4 GB. We are researching, not looking for proven recipes.

Your last review proposed more data. We are not doing that: the corpus has only 6.2M unique training characters, and data tests scaling but does not improve the method. Treat the dataset, one pass and the budget as fixed. Use validation, not test, to select from now on.

Current model (iter 24, three seeds): 66 s, test 1.3320, renamed 1.6298, val about 1.468. Three layers of width 256, no attention: fixed causal EMAs plus six lag lookups, lagged lower-layer state as input to upper layers, per-layer next-character readouts mixed by learned `g`, suffix-match copy heads with a learned gate, LR decay. Per-layer test: 1.6442, 1.4592, 1.4012. Backprop GPT with attention: 181 s, test mean 1.3320, renamed 1.6675 on its one measured seed.

Already tried:
- DFA: test 1.549.
- Count tables: test 1.165 but renamed gap 0.441, 244 s, 1.9 GB.
- Two-ahead auxiliary readouts on lower layers: gained 0.004 for 9 s.
- Associative retrieval (iter 20) gave no useful gain.

Questions:
1. What most limits this learner now? Name the one mechanism you believe costs the most loss, and the evidence in these numbers.
2. Name your top three changes to the architecture or learning rule, ranked by expected validation gain per second. Include ideas nobody has listed here if you think they are better. Your earlier ideas were width 320 and residual prediction on detached lower-layer logits; rerank them honestly against anything else.
3. For your first choice only: exact configuration, expected time, and a pass or kill rule on validation.
4. Could learned attention be trained locally inside this budget, and would it help more than your first choice? Answer yes or no, with the reason.

Be direct. If the whole direction has a better alternative, say so.
