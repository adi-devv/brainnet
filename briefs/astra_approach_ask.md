Follow-up. Still consult only: do not edit, create or run anything. Answer in under 500 words.

Decision: a 2-3x speedup from forward-pass learning is good enough for us. Your last answer gave ceilings and hints. We now want buildable designs.

Facts to design against:
- Backprop GPT: 6.2M chars, 180 s, test 1.329, renamed 1.667, 321 MB.
- Our `local` net: 3 layers, width 256, no attention: fixed causal EMAs plus 6 lag lookups. Each layer has its own linear readout and updates from its own cross entropy during the forward pass. 6.2M chars, about 70 s, test 1.492, renamed 1.819, 105 MB. It uses the exact chain rule inside a layer and only avoids crossing layers. Dropping attention confounds its speed.
- Count tables + learned pointer heads: 6.2M chars, 244 s, test 1.165, renamed 1.606, about 1.9 GB.
- GTX 1650 4 GB. Each experiment should take minutes.

Give one design for each:

**A. Forward-pass learning: at least 2x faster than backprop per character, at equal or better loss than backprop on the same data.** It must have attention or something that does attention's job. Specify:
1. Exactly what each block computes.
2. What updates each parameter, from which signal, and at what point in the forward pass.
3. Where any within-layer or within-block derivatives are used. Say plainly how much of the chain is still crossed.
4. Why it recovers the loss that layer-local learning has lost so far (0.16 test loss at 6.2M). Name the specific missing thing it restores.
5. Where the speedup comes from, as an operation count against the backprop GPT above.

**B. Sample efficiency: equal loss from 3-5x fewer chars.** Turn binding-aware copying into a precise algorithm. Specify:
1. How a binding is created from the observed prefix, what its state is, and when it ends.
2. How the model decides copy versus generate, and how it scores first spellings.
3. What gets keyed on binding IDs, and how evidence is shared across spellings.
4. How it fits in under 321 MB, or say honestly that it cannot.
5. What fraction of the current test loss you expect this to remove. Give a number and the reasoning.

Then:
- Can A and B be one model? If so, describe the combination in three sentences. If they should stay separate, say why.
- Which should we build first, and the smallest experiment with exact pass and kill numbers?
- Updated odds, each with one reason: A meets its target; B meets its target; both together are faster and better than backprop on our setup.

Do not re-explain the ceilings. If either target is not reachable with any design you can defend, say so and give the closest version you can.
