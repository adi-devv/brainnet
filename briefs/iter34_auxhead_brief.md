# Build iter34.py (training-only nonlinear aux heads) + screen34.py (fast screen)

Repo: this folder. Read `iter30.py`, `iter33.py`, `iter27.py` (and imports), `briefs/iter34_astra_out.md`. Do NOT edit existing files. Do NOT run full training (GPU shared); only the smoke commands below.

## Idea (Astra rank 1)
In iter27 local learning, each lower layer l in {0,1} learns only from its own LINEAR readout CE, so its features are pushed to be linearly next-char-separable right away. Instead, add a small training-only nonlinear head per lower layer: h_l -> Linear(256->64) -> GELU -> Linear(64->96 vocab logits). The lower layer's backbone params learn ONLY from this aux head's CE (local: gradients stay inside layer l, inputs detached as iter27). The existing inference readout of layer l (used in evals, g mixing and copy) still trains on its own CE, but on DETACHED h_l, so it no longer shapes the backbone. Layer 2 (top) is unchanged.

## Spec
- Build on iter30/iter33 structure; same CLI shape: `--mode {auxhead,auxhead_bp}`, `--hid` (default 64), `--seed`, `--smoke N`, `--save-ckpt` (see screen).
- Everything else exactly iter27 seed 0: widths, init, sampler, batch, T, steps, LR schedule, Adam, clipping, copy heads, g mixing, evals, curve logging. Aux-head params: own generator (so every iter27 param stays identical at init), same Adam/LR. Second layer of each head zero-init (use a sensible init; say which).
- `auxhead_bp`: iter30 backprop mode plus the same aux heads (aux CE added to the summed loss; inference readouts as in iter30 backprop). Fair control.
- Aux heads are training-only: not used in any eval; exclude from inference param count but report them.
- Memory: reset CUDA peak after startup checks; report train vs eval peak. Report train_seconds excluding evals.

## Checks (startup)
- At init every iter27 param equals a fresh iter27 seed model.
- No gradient crosses layers (float64 tiny case); lower backbone gets gradient only from its aux head; inference readouts get no gradient into the backbone.
- Aux-head gradients match float64 autograd on a tiny case.

## screen34.py — fast screening tool (Astra Q2)
1. `--make-ckpt`: run iter30 `local` seed 0 to 15% of steps (exact schedule, not compressed), save model + Adam states + sampler RNG state to `ckpt/local_s0_p15.pt`. (One-time cost.)
2. `--candidate {local,auxhead,topdown}`: load ckpt, branch training for 64 identical batches with the original schedule continuing from 15%, then evaluate top-layer and final mixed loss on 32 fixed held-out val windows. Always also run `local` from the same ckpt for the paired delta. For candidates that add params (aux heads), init them fresh at branch time; do a 32-step head-only warmup (backbone frozen) before the 64 steps, and give `local` the same number of extra idle batches so the data matches.
3. Also print, on 4 fixed batches at the ckpt, cosine and norm ratio between each lower layer's candidate local update direction (Adam-free raw gradient) and the true BP gradient of the summed iter30 loss (diagnostic only).
4. Output `results/screen34_{candidate}.json` with deltas and wall time. Aim for <20 s warm.

## Output
`results/iter34_{mode}.json` (seed>0: `_seed{seed}`). Keep iter30 fields; add `hid`, `aux_params`, `aux_loss_last10` (per lower layer), `train_peak_mib`, `references=dict(local_val=1.45627, bp_val=1.38687, topdown_val=1.55959)`.

## Verify
Smoke both modes: `/c/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B iter34.py --mode MODE --smoke 204800`. If your sandbox can't run Python, write both files completely and skip. Report in `briefs/iter34_codex_out.md`: summary, smoke outputs, any deviation.
