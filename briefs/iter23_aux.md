# iter23: iter22 plus a two-ahead auxiliary objective on layers 0 and 1

Create exactly one new file, `iter23.py`, in this folder. Do not edit any other file. Do not run git.

Python: `/c/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B`. Read `iter14.py`, `iter15.py`, `iter17.py`, `iter18.py`, `iter22.py` first. Match the repo's dense style, but keep it correct. Grad mode stays off; no autograd.

**Only run smoke tests (a few hundred thousand chars at most, under 2 minutes). Do not run the full experiment. I run it.**

## Change
Everything is `iter22.py` except the local objective of layers 0 and 1. Subclass `iter22.Model`; override only what you must. `forward`, `_layer`, `_update` and `train` should normally be inherited unchanged.

**New parameters.** For l in {0, 1}: `Ra{l}` (256xV, zeros) and `ca{l}` (V, zeros), with zeroed Adam state in `self.state`. Draw no random numbers, so all shared initial weights equal iter22's at seed 0. Layer 2 gets none. Store `self.aux_weights = (.75, .25)`; the code must read the weights from there.

**Two-ahead targets.** `y2 = cat(y[:, 1:], full -1 column)`, then also set `y2[y < 0] = -1`. Position t of `y2` is the character two ahead of `x[:, t]`. The final position and every unavailable target are masked.

**Objective for l in {0, 1}:** `0.75 * CE(z, y) + 0.25 * CE(za, y2)`, where `za = h @ Ra{l} + ca{l}` uses the not-yet-updated weights, and each CE is a mean over its own valid targets (`iter14.error` normalises this way). If `y2` has no valid target at all, the two-ahead error must be exact zeros, not NaN. `error` as written divides by zero there, so guard it.

**Gradients for l in {0, 1}**, all of them the exact gradient of the objective above:
- `e = error(z, y)`, `e2 = error(za, y2)`.
- Hidden delta: `d = (.75 * (e @ R{l}.T) + .25 * (e2 @ Ra{l}.T)) * (1 - h.square())`. It is summed before any clipping.
- `R{l}` gets `.75 * readout_delta(h, e)`; `c{l}` gets `.75 * e.sum((0, 1))`.
- `Ra{l}` gets `.25 * readout_delta(h, e2)`; `ca{l}` gets `.25 * e2.sum((0, 1))`.
- `b{l}`, `W1`, `A*`, `P*`, `E` are computed as in `iter22.Model._local_grads`, using the combined `d`.
- All of these go through one `_update` call, once per layer, before the next layer activates, with the existing per-parameter clip.

**Layer 2 is unchanged:** call `iter22.Model._local_grads` for it.

**Auxiliary logits never enter `g`, `final`, the copy gate, evaluation or the returned loss.** `forward` returns the same tuple as iter22.

## Settings
As iter22: batch 16, T 128, seed 0, network lr 1e-3 constant, gate lr 1e-2 constant, `tr[:cut]`, cut 95%, one pass, `iter17.epoch(train, 0, len(train), V)`, same `train_seconds` bounds with `torch.cuda.synchronize()`.

## Checks, run at the start, before training
1. `iter18.checks(net, gate)`.
2. At init, every parameter shared with `iter22.Model` built from the same seed is identical, atol 0, with RNG restored as in iter22. `Ra*`, `ca*` and their Adam state are exact zeros with the right shapes, and there are no aux keys for layer 2.
3. At init, on a real 128-char window, `forward` matches iter22's `zs`, atol 1e-6.
4. `y2` correctness on a real batch and on a batch with masked tails of different lengths: final position masked, every other position equals `y[:, t+1]` or -1 as specified, and the all-masked case gives zeros with no NaN.
5. Finite-difference check on float64 of the combined objective, freezing lower layers, with masked targets, at **non-zero** `Ra` and `ca` (fill the probe's with a local `torch.Generator`, never the global RNG). Cover entries of `W1` in all three row blocks, `A0`, `P0`, `E`, `b0`, `R0`, `c0`, `Ra0`, `ca0`, `R1`, `c1`, `Ra1`, `ca1`. Compare against the gradients `train` uses. Report the worst absolute error per key.
6. On the same probe, `train` applied the clipped analytic gradients: Adam first moment equals `.1 * clipped`, for `W1`, `Ra0` and `Ra1`.
7. Layer 2's gradients on identical inputs equal iter22's exactly.
8. **Reduces to iter22.** On deep copies with `aux_weights = (1., 0.)`, 3 real batches including a masked tail, every shared parameter, Adam state and gate tensor equals an `iter22.Model` trained the same way, atol 1e-6. Aux parameters stay zero.
9. **Aux logits are isolated.** On a deep copy, changing `Ra0, ca0, Ra1, ca1` to non-zero values leaves `forward`'s `zs`, the `final` passed to the gate and the gradient for `g` unchanged.
10. **Order.** On a deep copy with the real weights, record events as iter22 does and assert exactly activate 0, update 0, activate 1, update 1, activate 2, update 2, gate, g. Layer groups include `Ra{l}`, `ca{l}` for l in {0, 1}. Immediately after update l, layer l's parameters have changed and layer l+1's have not.
11. Confirm checks leave `net` and `gate` state untouched.

## Report
`results/iter23_aux.json`, opened with mode `'x'`. CLI: `iter23.py [--smoke N]`. `--smoke N` limits training chars, evaluates 16 windows, and writes to a temp dir outside `results/`. Leave no smoke files in the repo.

JSON: same fields as `iter22.py`, with `settings` updated (`aux_weights=[.75, .25]`, `aux_layers=[0, 1]`, `aux_target='two ahead, y[:, 1:]; final position masked'`, `aux_init='zero; no RNG draws'`, `aux_in_g_or_gate=False`), plus:
- `iter22_reference`: test 1.3568, renamed 1.6462.
- `same_sampler_backprop_reference`: test 1.340673, renamed 1.686344.
- `pass`: `train_seconds` <= 90, test <= 1.329, renamed <= 1.667.

Finish with: what you ran, the smoke output, and any deviation from this brief.
