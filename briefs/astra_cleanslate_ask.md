# Consult only: what would beat backprop?

Do not edit, create, run, or read anything in this repo. Ignore any existing code here. Think from first principles and from what you know of the literature and of the brain. Answer in under 400 words.

## Goal
A new transformer-like architecture that learns without backprop and is both faster and better than backprop. "Better" means lower loss on the same data, stronger generalization to new situations, and higher sample efficiency. It should hold up at scale, not only on toy models.

This is research. We want the approach most worth trying, not a proven recipe and not a safe incremental one. A long shot is fine if the mechanism is real. Evolution strategies, RL², and meta-learning over "lifetimes" are out of scope.

Assume nothing has been tried yet.

## Test setup available
Character-level code language model, about 6-16M characters of training data, compared against a small backprop GPT (1.2M parameters). One GTX 1650 with 4 GB, 12 CPU cores, 15 GB RAM. Each experiment should take minutes.

## Question
Pick exactly one approach. Give:
1. The approach, concretely: what the network computes, what updates each parameter, and from what signal.
2. Why it could beat backprop on both speed and quality. Name the specific inefficiency of backprop it removes, and why that removal does not cost quality.
3. Why it should still work at large scale.
4. How it is truly not backprop in disguise. Say plainly if any part still computes gradients through a chain of operations.
5. The smallest experiment that would kill the idea, with what result would count as failure.
6. Why not your two closest alternatives.
7. Your honest odds it beats backprop, and the most likely way it fails.

Say plainly if you think the goal as stated is unreachable, and what the closest reachable version is.
