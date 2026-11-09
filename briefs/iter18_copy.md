# iter18: suffix-match copy heads on `local`, plus a fair backprop control

Create exactly one new file, `iter18.py`, in this folder. Do not edit any other file. Do not run git. Another job is building `iter19.py` at the same time; leave it alone.

Python: `/c/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B`. Read `iter15.py`, `iter17.py`, `iter16.py`, `baseline.py`, `iter5.py` first. Follow `iter17.py` as the template. Match the repo's dense style, but keep it correct.

**Only run smoke tests (a few hundred thousand chars at most, under 2 minutes). Do not run the full experiments. I run those.** The GPU is shared with the other build's smoke tests.

## Mode `copy`
Train `iter15.Model(V, False)` exactly as `iter17.py` does for one pass with constant lr: seed 0, bs 16, T 128, `tr[:cut]`, cut = 95%, same `epoch` sampler (seed 0). Do not change the network, its settings or its updates in any way.

Add a copy layer on top:
- Four heads, suffix lengths L = 1, 2, 4, 8. At position t of a window, head L finds earlier positions j < t with `x[j-L+1..j] == x[t-L+1..t]` (positions must exist inside the window) and proposes the char `x[j+1]`. `x` is the input row, so `x[j+1]`, with j+1 <= t, is already observed. Never look at `y` or at any input after t. Head distribution = normalized counts of proposed chars over matches. No match means the head is masked (logit = -inf).
- Memory is window-local: nothing persists between windows.
- Gate: softmax over 5 components (net, 4 heads), logits = A @ phi, A a matrix of 5 x F. Features phi are detached and fixed: bias, per-head matched flag, per-head log2(1 + match count) / 7, per-head max proposal probability, net max probability, net entropy / log V.
- Mixture q = sum over k of pi_k p_k. Net component p_0 is the softmax of the existing `predict` combination (`sum(w*z for w, z in zip(net.p['g'], zs))`).
- Train A from mixture cross entropy with the analytic gradient, not autograd: for each k, d(log q(y))/d(logit_k) = pi_k (p_k(y)/q(y) - 1), multiplied by phi. Use a manual Adam with lr 1e-2. Floor probabilities at 1e-6. No gate quantity ever feeds back into the network.
- Train the gate at every position, in the same forward pass as the net update. The net update comes first or last, but the gate must use the net's prediction from before that batch's net update.

The prediction at eval is log q. Add a switch to evaluate with copy disabled (pure net).

Report, in one JSON result: `test_loss`, `renamed_loss`, `renamed_gap`, `val_loss`, `test_loss_nocopy`, `renamed_loss_nocopy`, novel first/repeat loss from `Q.novel` for the copy prediction, mean gate weight per component on test, `train_seconds` (training only, including the copy layer), `chars`, `peak_allocated_mib`, settings. Print a `pass` dict with these bars: test <= 1.30, renamed <= 1.64, train <= 180 s, peak <= 321 MB, repeat-name loss <= 4.0. `test_loss_nocopy` should be close to the known one-pass `local` result of 1.4922; print the difference.

Results go to `results/iter18_copy.json`, opened with mode `'x'` so nothing is overwritten.

## Mode `backprop_same`
A control: the `baseline.GPT` model, optimizer, AMP, clipping and cosine schedule exactly as in `baseline.train`, but batches come from the same partitioned sampler as `local`: one pass over `tr[:cut]`, non-overlapping 128-char windows, shuffled with seed 0, batch 64. Use `ignore_index=-1` for short final windows. GPT has no pad embedding, so drop any window whose start is 0 instead of feeding id V. Steps and the schedule come from the true number of batches. Training needs `torch.enable_grad()`, as `iter16.py` does. Evaluate test, renamed and `Q.novel` the same way as `copy`. Results go to `results/iter18_backprop_same.json`, mode `'x'`.

## Checks, run at the start of `copy`
- Vectorized match code equals a naive Python loop on a small hand-made tensor, for all L.
- Finite-difference check of the gate gradient on a small case.
- Causality: changing input chars after position t leaves outputs at positions <= t unchanged.

Add `--smoke N` to limit training chars and write results to a temp path outside `results/`. Do not leave smoke files in the repo.

Finish with: what you ran, the smoke output, and any deviation from this brief.
