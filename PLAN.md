# Plan

**Goal:** a transformer-like architecture that learns without a backward pass and is both faster and better than backprop.

**Target for now:** 2-3x faster than backprop at equal or better loss. Removing the backward pass cannot give much more than that, because the forward and update work remains.

**Benchmark to beat (backprop GPT, 6.2M chars, GTX 1650):** 180 s, test 1.329, renamed-identifier test 1.667, 321 MB.

Both designs below came from a design consultation with GPT-6 Astra (`briefs/astra_approach_out.md`). Neither is proven. The odds are its own estimates.

## A. Forward-pass learning with fixed associative attention (build first, iter 20)

`local` runs in about 70 s but loses 0.163 test loss at 6.2M. It has no content-dependent access, and each layer solves its own prediction task. A restores both.

Three width-256 blocks. Each block:

1. **Retrieves** previous-layer states following the most recent matching character suffixes of lengths 2, 4 and 8. Only positions already observed.
2. Averages those states into its input `x`.
3. Computes `h = tanh(W x)`, `z = z_previous + R h`, `x_next = x + U h`. `h` has width 128. `U` is a fixed signed scatter. Initial embeddings are fixed.
4. **Updates during the forward pass**, from `e = softmax(z) - y`:
   - `R <- R - lr * e h^T`
   - `W <- W - lr * [(R^T e) * (1 - h^2)] x^T`

Weights stay fixed through each minibatch and update after the block emits its activations. No derivatives flow into previous logits, input states or retrieval.

This still crosses two learned maps inside each block. **It is local backprop, not gradient-free learning.**

- **Pass at 6.2M chars:** at most 90 s, test at most 1.329, renamed at most 1.667.
- **Kill** this configuration if any threshold fails. Replicate passes across 3 seeds.
- **Odds: 15%.** Fixed retrieval may underfit.
- **Control:** same model with retrieval off.

## B. Causal lexical binding and copying (after A)

File-local spelling identities, not semantic bindings.

- A streaming lexer creates an ID when an identifier finishes, and stores spelling, occurrence count and last position. IDs expire at file end, with bounded-memory eviction.
- At each possible identifier start, mix a character generator with a distribution over stored spellings. Update mixture weights and copy-rank counts from posterior responsibilities after observing characters.
- Filter candidates as the prefix grows, and include identifier termination. First spellings get generator probability.
- Key structural counts by token classes and locally renumbered identifier IDs, so alpha-renamed contexts share counts.
- Cap tables at 128 MB and strings/indexes at 32 MB. With the 105 MB `local` net that is about 265 MB.

- **Expected gain:** 5-10% loss reduction. This does not establish 3-5x sample efficiency.
- **Pass:** the same losses as A's bars using at most 2.07M chars. At most 1.24M shows 5x.
- **Odds: 10%.** Naming explains a limited share of the loss.

## Combining them

A supplies structural predictions, B supplies spelling probabilities, and A trains against the mixture responsibilities. **Odds of beating backprop on both speed and loss: 10%.** Lookup overhead threatens the speedup.

## Dropped

- Discrete state merging over count tables (iter 19): stopped before any code. It was too indirect a route to the goal.
