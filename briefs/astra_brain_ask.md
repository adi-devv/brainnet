# Consult only: what would beat backprop, using what the brain does better?

Do not edit, create, run, or read anything in this repo. Ignore any existing code here. Answer in under 450 words.

## Goal
A new transformer-like architecture that learns without backprop and is both faster and better than backprop. "Better" means lower loss on the same data, stronger generalization to new situations, and higher sample efficiency. It should hold up at scale, not only on toy models.

This is research. We want the approach most worth trying, not a proven recipe and not a safe incremental one. A long shot is fine if the mechanism is real. Evolution strategies, RL², and meta-learning over "lifetimes" are out of scope.

Assume nothing has been tried yet.

## The brain is the existence proof
Humans learn far more from far less data than current nets, and generalize much further. Ilya Sutskever has named this as the central gap: models need vastly more examples and still generalize worse than people. The brain does this with slow, noisy parts on about 20 W. What is broadly known, with the caveat that much of the neuroscience is unsettled:

- **Sample efficiency:** people pick up a new word, tool or rule from a handful of examples, and apply it in very different settings.
- **Local learning:** a synapse changes from signals available at that synapse, such as pre- and postsynaptic activity plus broadcast modulators like dopamine. There is no known biological reverse pass that carries exact error gradients.
- **Sparse activity:** only a small share of neurons fire at any moment, and most of the network is not computing on a given input.
- **Two memory systems:** the hippocampus stores new episodes fast, in one shot, and the cortex learns slowly. Replay during sleep moves knowledge from one to the other.
- **Prediction:** cortex may be constantly predicting its input, with learning and signalling focused on what was surprising.
- **Value signals:** emotion and reward circuits steer what is worth learning, and may be part of why learning is so efficient.
- **Attention:** a person spends more processing on what is new than on what is familiar.

Do not copy biology for its own sake. Decide which of these actually causes the advantage, and which is just a side effect of the hardware.

## Test setup available
Character-level code language model, about 6-16M characters of training data, compared against a small backprop GPT (1.2M parameters). One GTX 1650 with 4 GB, 12 CPU cores, 15 GB RAM. Each experiment should take minutes.

## Question
Pick exactly one approach. Give:
1. Which brain property you think matters most for the advantage, and why.
2. The approach, concretely: what the network computes, what updates each parameter, and from what signal.
3. Why it could beat backprop on both speed and quality. Name the specific inefficiency of backprop it removes, and why that removal does not cost quality.
4. Why it should still work at large scale.
5. How it is truly not backprop in disguise. Say plainly if any part still computes gradients through a chain of operations.
6. The smallest experiment that would kill the idea, with what result would count as failure.
7. Why not your two closest alternatives.
8. Your honest odds it beats backprop, and the most likely way it fails.

Say plainly if you think the goal as stated is unreachable, and what the closest reachable version is.
