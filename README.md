# brainnet

Notes and experiments on neural networks that learn **without gradient descent**.

Nothing here is proven. This is a working list of pointers and the approaches being tried, with honest status on each.

## Pointers

- The brain has no global backward pass. Synapses change from local signals, and a reward or surprise signal decides which changes stick.
- New experiences are stored fast (hippocampus-like), then replayed in sleep to slowly train long-term weights (cortex-like).
- Gradient descent needs a differentiable path from the result to the weights. A hard pass/fail reward has none, so search or local rules are needed.
- The price of going gradient-free is sample efficiency. Expect many more trials per useful update.
- Gradient-free methods have not matched backprop at language-model scale. Small, continual-learning settings are the realistic place to look first.

## Approaches

| Approach | Idea | Status |
|---|---|---|
| Evolution strategies | Copy the weights with random noise, score each copy by reward, move toward the best | Tried on Sudoku with a 0/1 reward |
| Meta-evolution | Evolve the learner's architecture and hyperparameters across lives so each new life learns faster | Tried on Sudoku together with the above |
| Hebbian and three-factor rules | Local "fire together, wire together" updates, gated by a reward or surprise signal | Not started |
| Predictive coding, forward-forward | Layer-local error signals instead of one global backward pass | Not started |
| Closed-form memory writes | Write a fact straight into the weights with a rank-one solve (MEMIT/AlphaEdit style), no training loop | Idea only; still needs a few gradient steps to find the target value |
