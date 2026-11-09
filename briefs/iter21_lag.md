# iter21: iter18 plus lagged second-layer states into the final layer

Create exactly one new file, `iter21.py`, in this folder. Do not edit any other file. Do not run git.

Python: `/c/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B`. Read `iter14.py`, `iter15.py`, `iter17.py`, `iter18.py` first. Match the repo's dense style, but keep it correct. Grad mode stays off; no autograd.

**Only run smoke tests (a few hundred thousand chars at most, under 2 minutes). Do not run the full experiment. I run it.**

## Change
Everything is iter18 `copy` mode (`local` plus copy heads and `iter18.Gate`) except the final layer's input.

Subclass `iter15.Model` with `hybrid=False`. Reuse by import rather than copying: `iter15.Model.train`, `iter18.Gate`, `iter18.checks`, `iter17.epoch`.

- After `super().__init__`, replace `p['W2']` (1024x256) with a 1536x256 matrix: first 1024 rows are the original tensor, unchanged; the 512 new rows are zeros. Resize its two Adam state tensors to match, all zeros. Do not draw extra random numbers, so every other initial weight equals iter18's at seed 0.
- Override `forward` only for layer 2. Layer 2 input is `u = cat(existing_u, lag1, lag2)`, where `existing_u` is the existing 1024-wide `cat(hs[1], 3 EMAs of hs[1])`, `lag1[:, t] = hs[1][:, t-1]`, `lag2[:, t] = hs[1][:, t-2]`, and positions with no such lag are zeros. Layer 1's input is unchanged. Return the same tuple as `iter14.Model.forward`, with the wider `u` in `us[1]`.
- `iter15.Model.train` then produces the `W2` gradient through `readout_delta(us[1], d)`. Do not add any cross-layer gradient.

Settings as iter18: batch 16, T 128, seed 0, network lr 1e-3, gate lr 1e-2, `tr[:cut]`, cut 95%, one pass, `iter17.epoch(train, 0, len(train), V)`, same gate-before-network order as iter18, same `train_seconds` bounds with `torch.cuda.synchronize()`.

## Checks, run at the start, before training
1. `iter18.checks(net, gate)`.
2. At init, on a real 128-char window, each `zs[l]` from the new model equals iter18's `Q.Model(V, False)` built from the same seed, to atol 1e-6.
3. `us[1]` columns: first 1024 equal the old input; next 256 equal `hs[1]` shifted one position with zeros at position 0; last 256 shifted two with zeros at positions 0 and 1.
4. Finite-difference check on float64 of the layer-2 local CE against `W2` at entries in each of the three column blocks (old rows, lag1 rows, lag2 rows), matching what `train` computes.

## Report
`results/iter21_lagged.json`, opened with mode `'x'`. CLI: `iter21.py [--smoke N]`. `--smoke N` limits training chars, evaluates 16 windows, and writes to a temp dir outside `results/`. Leave no smoke files in the repo.

JSON: same fields as iter18 copy mode (`test_loss`, `renamed_loss`, `renamed_gap`, `val_loss`, novel first/repeat, nocopy test and renamed, `mean_gate_weight_test`, `g`, `train_seconds`, `chars`, `steps`, `peak_allocated_mib`, `settings`), plus `per_layer_test_loss` (pure net, each layer), `checks`, `iter18_copy_reference` with test 1.4112 and renamed 1.6975, and `pass` with these bars: `train_seconds` <= 90, test <= 1.329, renamed <= 1.667.

Finish with: what you ran, the smoke output, and any deviation from this brief.
