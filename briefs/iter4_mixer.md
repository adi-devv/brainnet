# Iteration 4: gated linear mixer instead of Hedge

Work in `D:\Aadit\claudez\brainnet` only. Read `hybrid.py` and `benchmark.py`, then create one new file, `hybrid_mixer.py`. Do not create, change, or delete anything else except `results/iter4_*.json`. No network, no pip install, no git commands. Run Python only as `"C:\Users\aadit\AppData\Local\Programs\Python\Python312\python.exe" -X utf8 ...`. Another experiment is using the CPU, so use at most 2 threads and only do a smoke check on a very small budget. Do not run the full variants. Reply with at most 10 lines of notes: what you wrote, and the smoke check result.

## Why
In iteration 2, exact context counts (`count_ngram`) scored 1.6727 nats/char, and Hedge weights collapsed onto one context length. Hedge can only pick one expert. This tests whether local, context-selected logistic mixing gets more out of the same five predictions.

## Rules
- No backprop: no `.backward()`, `torch.autograd`, `torch.func.grad`, or `torch.optim`. Run under `torch.set_grad_enabled(False)`.
- Every weight update is a hand-written local delta rule. Each mixer neuron learns from its own log-loss against the true next character. No error passes between layers.
- No learning during evaluation.
- Import from `hybrid.py` and do not change it. Do not change `benchmark.py`, `baseline.py`, `LOCK.json`, or `data/`.
- Same data and budget: `B.load()`, 122 batches of 64 windows of 128 chars, seeded sampling, `B.evaluate(model.logprobs, device="cpu")`.

## Model
Use `Hybrid(V, mode="ngram", surprise=False)` from `hybrid.py` to produce the five per-length distributions `ps` (lengths 2, 4, 8, 16, 32) before each write, so training inputs are out-of-sample. Clamp probabilities to at least 1e-6 before taking logs.

A mixer neuron holds one weight per input and has a selector table. For selector value c it computes `logits = sum_i w[c, i] * log p_i`, softmaxes them over V, and updates only the selected row: `w[c, i] += lr * (log p_i[y] - sum_v q[v] * log p_i[v])`, where q is its own output distribution. Initialise weights to `1 / n_inputs`. Learning rate default 0.02.

## Variants
| Name | Setup |
|---|---|
| `hedge_recheck` | Original Hedge output. Must match 1.6727 within 0.002 |
| `mix_global` | One neuron, a single selector value |
| `mix_char` | One neuron, selector = last char |
| `mix_two_layer` | Layer 1: three neurons selected by nothing, last char, and hash of last two chars into 4096 values. Layer 2: one neuron, selector = last char, whose inputs are layer 1's log-probs. Each neuron learns from its own loss |

## Script
`python -X utf8 hybrid_mixer.py <variant>` trains seed 0, evaluates, asserts the model state is unchanged by evaluation, prints JSON, and saves `results/iter4_<variant>.json` with test_loss, train_seconds, chars_per_sec, and settings. Causality must hold: position t uses only chars at positions <= t. Handle the selector for position 0 with the pad value. Keep it under 150 lines.

## Kill rule
If `mix_two_layer` does not beat `hedge_recheck` by at least 0.1, the idea is dropped.
