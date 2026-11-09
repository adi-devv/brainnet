# iter20: forward-pass learning with fixed associative attention

Create exactly one new file, `iter20.py`, in this folder. Do not edit any other file. Do not run git.

Python: `/c/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B`. Read `iter14.py`, `iter15.py`, `iter17.py`, `iter18.py` and `iter5.py` first. Match the repo's dense style, but keep it correct. Grad mode stays off everywhere; no autograd.

**Only run smoke tests (a few hundred thousand chars at most, under 2 minutes). Do not run the full experiments. I run those.**

## Model
A new class, not a subclass of the `iter14` network. Constants: `D=256`, `H=128`, three blocks, `LENS=(2,4,8)`, seed 0 by default.

**Input state `s[0]`, width D.** Same form as `iter14.Model.forward` without the bias: six lag lookups `A_j[ids_j]` plus three causal EMAs of `E[x]` projected through `P_k`, with the same `M`, the same pad id `V` and the same shapes. Initialize with the same stds as `iter14`, then multiply by a module constant `INPUT_SCALE`: 16 in modes `assoc` and `noretrieve`, 1 in mode `learned_input`.

**Retrieval indices,** computed once per batch from `x` only. For each L in LENS and each position t, find the most recent j with L-1 <= j < t and `x[j-L+1..j] == x[t-L+1..t]`, inside the window. Result: an index `j+1` and a matched mask. Memory is window-local. `j+1 <= t`, so nothing unobserved is read.

**Block l (0, 1, 2):**
- `x_l` is the average of `s[l][t]` and the states `s[l][j_L+1]` of the matched lengths: divide the sum by `1 + number of matched lengths`. With no match, `x_l = s[l]`. Mode `noretrieve`: always `x_l = s[l]`, but still compute the retrieval indices so the cost stays comparable.
- `h = tanh(x_l @ W_l + b_l)`, with W_l of shape (D, H), std `1/sqrt(D)`, and b_l zeros.
- `z_l = z_(l-1) + h @ R_l + c_l`, with `z_(-1) = 0`, R_l of shape (H, V), std `1/sqrt(H)`, and c_l zeros.
- `s[l+1] = x_l + h @ U_l`, with U_l fixed, of shape (H, D), entries uniform ±`1/sqrt(H)`, never updated.

The prediction is `z_2`. There are no layer-mixing weights.

**Update, inside the block loop, immediately after the block emits `z_l` and `s[l+1]`.** There is no second pass over the network.
- `e = iter14.error(z_l, y)`. Treat `z_(l-1)`, `x_l` and retrieval as constants.
- `d = (e @ R_l.T) * (1 - h.square())`.
- Gradients: R_l from `readout_delta(h, e)`, c_l from `e.sum((0,1))`, b_l from `d.sum((0,1))`, W_l from `readout_delta(x_l, d)`.
- Apply the same per-parameter clip (norm at most 1) and manual Adam as `iter14.Model.train`: lr 1e-3, betas .9/.999, eps 1e-8, one shared step count per `train` call.
- Compute `z_l` and `s[l+1]` from the old weights before applying that block's update.

**Mode `learned_input`** is `assoc` plus training the input tables in block 0 only. Let `d0 = d @ W_0.T`, the error at `x_0`. Update `A_j`, `P_k` and `E` from `d0` the same way `iter15.Model.train` does from its `d`, treating retrieved states as constants, so only the `s[0][t]` route counts. Divide `d0` by `1 + number of matched lengths` at each position, because that is the weight of `s[0][t]` in `x_0`. This crosses three maps; label it so in `settings`. The other modes never touch these tables.

## Training
Same data as `iter17.py`: `tr[:cut]`, cut = 95%, one pass, `iter17.epoch(train, seed, len(train), V)`, bs 16, T 128, constant lr. `train_seconds` covers all training including retrieval, with `torch.cuda.synchronize()` at both ends.

## Report
Results in `results/iter20_{mode}.json`, or `results/iter20_{mode}_seed{n}.json` when seed is not 0, opened with mode `'x'`. CLI: `iter20.py assoc|noretrieve|learned_input [--seed N] [--smoke N]`. `--smoke N` limits training chars and writes outside `results/`. Leave no smoke files in the repo.

JSON: `test_loss`, `renamed_loss`, `renamed_gap`, `val_loss`, `per_block_test_loss`, novel first/repeat via `iter15.novel`, retrieval match rate per L on test, `train_seconds`, `chars`, `steps`, `peak_allocated_mib`, `settings`, and `pass` with these bars: train <= 90 s, test <= 1.329, renamed <= 1.667.

## Checks, run at the start
- Vectorized retrieval equals a naive Python loop on a small hand-made tensor, for every L.
- Finite-difference check of the R, W and b gradients of one block on a small float64 case, with `z_prev` fixed and nonzero.
- Causality: changing input chars at and after position t+1 leaves outputs at positions <= t unchanged.
- `I.stamp` shows evaluation writes no state.

Finish with: what you ran, the smoke output, and any deviation from this brief.
