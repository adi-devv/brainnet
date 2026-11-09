# Iteration 2: Hebbian features + sparse written memory (no backprop)

Work in this folder only. No network, no pip install, no git commands, and do not write outside this folder.
Do not modify `benchmark.py`, `baseline.py`, `LOCK.json`, or anything in `data/`.

## Goal
Write `hybrid.py` (under about 300 lines) and run the variants below on CPU. The model predicts the next character and learns during its forward pass only.

## Hard rules
- No backprop anywhere: no `.backward()`, `torch.autograd`, `torch.func.grad`, or any `torch.optim`. Run under `torch.set_grad_enabled(False)`.
- No end-to-end error signal chained through layers. The only feedback allowed is one scalar per prediction (surprise, below), or a per-component score used for mixing.
- No learning during evaluation. The model is frozen while `B.evaluate` runs.

## Data and budget (must match the backprop baseline)
- Import `benchmark as B`, and use `B.load()`.
- Train on exactly 1,000,000 characters in total, sampled like the baseline: batches of 64 random windows of 128 characters from the train set, seeded by the run seed, using all windows until the budget is used (122 batches = 999,424 chars is the same count the baseline uses).
- Evaluate with `B.evaluate(model_fn, device="cpu")`. `model_fn(x)` takes a long tensor [B,128] and returns logits [B,128,V] (log-probabilities are fine). Position t must use only characters at positions <= t. Early positions have short context, so pad.

## Model
For each of several context lengths n (default 2, 4, 8, 16, 32 preceding chars):
1. **Input code:** encode the last n characters as a fixed vector (for example, random fixed character vectors combined by position, normalised). Fixed, not learned.
2. **Hebbian feature layer:** K neurons (default 2048), each with a unit-norm weight vector. The top-k neurons by similarity to the input are active (default k=8). Only winners update, moving toward the input and renormalising (competitive Hebbian/Oja). Fresh neurons should be claimed by unexplained inputs so none stay dead.
3. **Sparse written memory:** a table K x V of counts. Writing adds the true next character to the rows of the active neurons. Reading sums those rows into a next-character distribution, with smoothing that backs off to the shorter-context prediction.
4. **Three-factor signal:** surprise = -log p(true next char) from the final mix, one scalar per position. It scales the Hebbian learning rate and the write strength.
5. **Mixing:** combine context lengths with multiplicative weights (Hedge-style, updated from each length's own loss). No gradient-based mixer.

## Variants (run all in parallel, one process each, `torch.set_num_threads(1)` or 2, seed 0)
| Name | Setup |
|---|---|
| `count_ngram` | Control: exact context hash table per length (no neurons, no Hebbian), same mixing and backoff |
| `random_keys` | Control: neurons stay fixed at random, memory only, no Hebbian |
| `hebb_memory` | Hebbian + memory, surprise off |
| `full` | Everything on |
| `full_k32` | `full` with k=32 |
| `full_K8192` | `full` with K=8192 |

## Sanity tests first (`test_hybrid.py`), and they must pass before the variants run
1. A short repeating pattern is learned: loss falls near 0.
2. Uniform random characters: loss stays near log(V).
3. Causality: changing characters after position t leaves the output at positions <= t unchanged.
4. Frozen eval: evaluating twice gives the identical number and leaves the model unchanged.
5. Source check: the banned calls above do not appear in `hybrid.py`.

## Limits
Each variant must finish in about 5 minutes; reduce K only if it cannot, and say so. Total run at most 15 minutes. If a step fails twice after a fix, stop and report it.

## Output
- Save `results/iter2_<name>.json` for each variant: test loss, train seconds, eval seconds, chars per second, and the settings used.
- Save `results/iter2_summary.json` with all of them together.
- End with a plain summary: the sanity test results, a table of test loss and train seconds per variant, anything changed or failed, and any way the hybrid might be getting help it should not (leakage, peeking at test data, etc). Report real numbers only.
