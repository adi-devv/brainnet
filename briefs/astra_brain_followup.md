Follow-up on your predictive-memory answer. Still consult only: do not edit, create or run anything. Answer in under 350 words.

We want a big speedup, not 1.5x. Two routes, and I want your view on each:

**A. Replace backprop with something that happens during the forward pass itself.** No separate backward step: each parameter updates as the signal passes through it.

**B. Sample efficiency.** Reach the same loss from far fewer characters.

Measured so far on our setup, char-level code model, one seed unless noted:
- Backprop GPT: 6.2M chars, 180 s, test 1.329, renamed-identifier test 1.667. 16M chars, 465 s, test 1.124, renamed 1.530.
- Count tables + learned pointer heads, no backprop: 6.2M chars, 244 s, test 1.165, renamed 1.606, about 1.9 GB.
- A net with layer-local updates during the forward pass, no attention: 6.2M chars, about 70 s, test 1.492, renamed 1.819.

So dropping the backward pass alone has not made anything both faster and better. Our no-backprop tables are slower per character than backprop, because of lookups and count updates. They look about 1.6x faster to reach equal test loss, by rough interpolation.

Questions:
1. What is the realistic ceiling on A, in time per character, and what would have to change to pass it?
2. What is the realistic ceiling on B, in characters needed for equal loss on held-out and renamed data? What exact mechanism delivers it?
3. Does your predictive memory deliver A, B, both or neither? If a different design is better for a big speedup, name it and change your pick.
4. Which one mechanism should we build first to get the largest speedup, and what numeric result on the setup above would prove the speedup is real? Include equal-loss comparisons, not just time per step.
5. Update your odds if they changed, and say why.
