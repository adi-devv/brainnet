# Iteration 5: does the mixer survive real tests?

Work in `D:\Aadit\claudez\brainnet` only. Read `hybrid.py`, `hybrid_gpu.py`, `hybrid_mixer.py`, `benchmark.py`, `baseline.py`, then create `iter5.py` (at most 260 lines). Create nothing else except `results/iter5_*`. Do not change any existing file. No network, no pip install, no git. Run Python only as `"C:\Users\aadit\AppData\Local\Programs\Python\Python312\python.exe" -X utf8 ...`. Only smoke check on tiny budgets; do not run the full commands. Reply with at most 10 lines: what you wrote and the smoke result.

## Rules
No backprop in the mixer or count code: no `.backward()`, `torch.autograd`, `torch.optim`; use `torch.set_grad_enabled(False)`. Local delta rules only, as in `hybrid_mixer.py` (`Neuron`; deltas summed over the batch, so lr means per-item). No learning during evaluation. Predictions before writes, so mixer inputs stay out-of-sample. Backprop is allowed only inside the baseline comparison, by importing `baseline.py` unchanged. Machine: GTX 1650 with 4 GB, 15 GB RAM, so total GPU memory stays under 3 GB.

## Hashed count tables on GPU
Replace the Python dicts of `ngram` mode with one flat int32 tensor per context length, `2**25` entries, indexed by `hash(context, char) mod 2**25`, on the device. Also keep a per-context total table of the same size (`hash(context)`). To read a position, read all V=96 chars for its context (`N x V` lookups). Writes use `index_add_`. Same backoff recipe as `Hybrid._predict` (alpha=0.5, uniform-count root, lens 2,4,8,16,32). Context hash: reuse the rolling `h*1000003+tok+1` from `Hybrid._codes`, then mix it well before the mod (the low bits are weak).

## Sampling
- `val` = last 5% of `train.bin`, never used for training in any iter5 mixer or PPM run.
- `1M` budget: seeded random windows as now, 999,424 chars.
- `one_pass`: the rest of train exactly once, as non-overlapping windows of 128 in seeded shuffled order. No char repeats, so inputs are never in-sample.
- Batch 976 windows. Device cuda.

## Commands (`iter5.py <name> [--lr X] [--budget N]`), each saving `results/iter5_<name>.json` with test_loss, val_loss, train_seconds, chars and settings
| Name | What |
|---|---|
| `check` | Hashed tables, `mix_two_layer`, bs=64, lr 0.0003125, 1M chars, full train as in iter4. Must score 1.3803 within 0.01, else exit with an error |
| `lr_grid` | `mix_two_layer` at 1M on train minus val, bs=976, lr in {3e-5, 1e-4, 3e-4, 1e-3}. Save val loss per lr and pick the best |
| `mix_ngram` | `mix_two_layer` at 1M, bs=976, best lr from `lr_grid` |
| `mix_ngram_hebb` | Same, but layer 1 neurons also take the 5 `hebb_memory` per-length distributions as inputs (10 inputs total). Use the Hebbian `Hybrid` from `hybrid_gpu.py` with `mode="hebb", surprise=False`, same windows, predictions before writes |
| `mix_full` | `mix_two_layer`, `one_pass`, best lr |
| `ppm_full` | Control: PPM-C style blending, orders 0 to 6, hashed tables as above, `one_pass`, escape = number of distinct chars seen in the context, no exclusions, frozen at test. No mixer. Also save per-order count of unseen contexts |
| `baseline_full` | Import `baseline.py`, train seed 0 at 16M chars with its locked settings, evaluate on the normal and renamed test |

## Renamed-identifier test
Decode `test.bin` with `data/meta.json` vocab. Rename every Python NAME token that is not a keyword or builtin to a random lowercase name of the same length, using the same mapping for the whole test text, seed 0. Strings and comments stay. Same length means the same fixed eval windows as `benchmark.eval_windows`. If `tokenize` fails on the concatenated text, run it line by line and keep what parses, and report the fraction of chars renamed. Evaluate every `mix_*`, `ppm_full` and `baseline_full` on both tests, saving `test_loss` and `renamed_loss`. Do not change `benchmark.py`; copy its evaluation logic for the renamed windows.

## Decision rules (save in settings, do not act on them)
- `mix_ngram_hebb` must beat `mix_ngram` by at least 0.03 on test, or Hebbian is dropped as a predictor.
- `mix_full` must beat `ppm_full` on test, or there is nothing beyond a compressor.
- The gap between `renamed_loss` and `test_loss` shows how much depends on memorised names.
