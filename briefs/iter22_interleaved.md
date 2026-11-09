# iter22: iter21 plus lags into layer 1, with updates interleaved in the forward sweep

Create exactly one new file, `iter22.py`, in this folder. Do not edit any other file. Do not run git.

Python: `/c/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B`. Read `iter14.py`, `iter15.py`, `iter17.py`, `iter18.py`, `iter21.py` first. Match the repo's dense style, but keep it correct. Grad mode stays off; no autograd.

**Only run smoke tests (a few hundred thousand chars at most, under 2 minutes). Do not run the full experiment. I run it.**

## Change 1: layer 1 input
Everything is `iter21.py` except layer 1's input and the training order.

Subclass `iter15.Model` with `hybrid=False`.

- `W1` and `W2` both become 1536x256: the first 1024 rows are the original tensors, unchanged; the 512 new rows are zeros. Resize both Adam state tensors to match, all zeros. Draw no extra random numbers, so all other initial weights equal iter18's at seed 0.
- For l in {1, 2}: `u = cat(hs[l-1], 3 EMAs of hs[l-1], lag1, lag2)`, where `lag1[:, t] = hs[l-1][:, t-1]`, `lag2[:, t] = hs[l-1][:, t-2]`, and positions with no such lag are zeros.
- Layer 0, the widths, EMAs and raw-character lags 0-5 are unchanged. `forward` returns the same tuple as `iter14.Model.forward`.

## Change 2: interleaved training
Override `train(self, x, y, gate)` so learning happens inside one bottom-to-top sweep:

For l = 0, 1, 2, in order:
1. Compute layer l's activation `h`, logits `z`, and input `u` using layer l's not-yet-updated weights and the lower layer's stored old-weight `h`.
2. Compute all of layer l's local gradients exactly as `iter15.Model.train(x, y, 0)` does (own CE, `R{l}.T` as feedback, the same `A`, `E`, `P`, `W`, `R`, `c`, `b` gradients).
3. Apply manual Adam to layer l's parameters now, before layer l+1's activation is computed. Same per-parameter clip 1, betas, eps, lr.
4. Pass the old-weight `h` upward. No layer may see a value computed with its own or a lower layer's updated weights during this sweep.

After the sweep: build `final = sum(g*z)` from the old-weight `zs` and old `g`, train `gate` on `final.softmax(-1)` and `y`, and update `g`. `self.step` increments once per `train` call, before any Adam update (Adam bias correction must match iter15). No cross-layer derivative anywhere. Share the layer computation between `forward` and `train` so they cannot drift.

Return the same final CE as iter15.

## Settings
As iter21: batch 16, T 128, seed 0, network lr 1e-3 constant, gate lr 1e-2 constant, `tr[:cut]`, cut 95%, one pass, `iter17.epoch(train, 0, len(train), V)`, same `train_seconds` bounds with `torch.cuda.synchronize()`.

## Checks, run at the start, before training
1. `iter18.checks(net, gate)`.
2. At init, on a real 128-char window, each `zs[l]` equals `Q.Model(V, False)` built from the same seed, to atol 1e-6. RNG restored as in iter21.
3. Both `us[0]` and `us[1]`: first 1024 columns equal the old input; next 256 equal the lower state shifted one with zeros at position 0; last 256 shifted two with zeros at positions 0 and 1.
4. Finite-difference check on float64 of the local CE against `W1` and `W2` at entries in each of the three column blocks (old rows, lag1 rows, lag2 rows), matching the gradient `train` uses. Freeze the lower layers.
5. **Order equals batch.** On deep copies, run 3 real batches through (a) the interleaved `train`, and (b) the same model class using the post-forward path: gate trained on the old `forward`, then `Q.Model.train(ref, x, y, 0)`. Assert every parameter, Adam state and gate tensor match, atol 1e-6.
6. **Order is really interleaved.** On a deep copy, record events and assert the sequence is exactly activate 0, update 0, activate 1, update 1, activate 2, update 2, then gate and g. Also assert that immediately after update l, layer l's parameters have changed and layer l+1's have not.
7. Confirm checks leave `net` and `gate` state untouched.

## Report
`results/iter22_lagged.json`, opened with mode `'x'`. CLI: `iter22.py [--smoke N]`. `--smoke N` limits training chars, evaluates 16 windows, and writes to a temp dir outside `results/`. Leave no smoke files in the repo.

JSON: same fields as `iter21.py`, with `settings` updated (layer inputs, lags, `update_order='per-layer during forward sweep; gate and g after'`), plus:
- `iter21_reference`: test 1.3799, renamed 1.6685.
- `same_sampler_backprop_reference`: test 1.340673, renamed 1.686344.
- `pass`: `train_seconds` <= 90, test <= 1.329, renamed <= 1.667.

Finish with: what you ran, the smoke output, and any deviation from this brief.
