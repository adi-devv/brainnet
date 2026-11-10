# Build iter33.py: top-down teaching (no cross-layer derivatives)

Repo: this folder. Read `iter30.py`, `iter27.py` (and what they import), and `results/iter30_*.json`. Do NOT edit existing files. Do NOT run full training (GPU shared); only the smoke commands below.

## Idea
Iter 27 trains each layer on its own next-char CE only, so lower layers are greedy. Full backprop on the same architecture is 0.069 val better (1.3869 vs 1.4564). Under backprop, layer 0's own test loss gets worse (1.644 -> 1.757) while the top improves.

The brain-like fix is to send ACTIVITIES down, not derivatives. Each step:
1. Run the full forward pass as now.
2. The top layer's own readout distribution p_top (softmax of layer 2's readout logits, network-only, no copy mixing) is DETACHED and sent down as a soft target.
3. Each lower layer l in {0, 1} trains on `(1-a)*CE(y, p_l) + a*KL(p_top || p_l)`, where p_l is its own readout. This is a local loss: gradients stay inside layer l (its inputs are detached as in iter 27), and nothing differentiates through p_top.
4. The top layer's loss is unchanged.

## Spec
- Build on iter30: same CLI shape, `--mode {topdown,topdown_bp}`, `--alpha` (default 0.5), `--seed`, `--smoke N`. Everything else stays exactly iter27 seed 0: widths, init, sampler, batch, T, steps, LR schedule, Adam, clipping, copy heads, g mixing, evals, and the curve logging from iter30.
- Ramp `a` linearly from 0 at 5% of steps to alpha at 20% of steps, then hold.
- Same-step teaching: use p_top from the SAME batch and step. Lower layers must keep what they need until layer 2's forward is done; restructure the step order if needed. No extra forward pass, and no backward sweep across layers.
- `topdown_bp`: the iter30 backprop mode plus the same KL term on layers 0 and 1, with p_top detached. This is the fair control (backprop given the same extra objective).
- Memory: reset the CUDA peak counter AFTER startup checks, and report the training-loop peak separately from the eval peak.
- Log the mean KL per lower layer over the last 10% of steps.

## Checks (startup, fast)
- With alpha=0, a full smoke run's train/val numbers equal `iter30.py --mode local` with the same smoke N, within GPU nondeterminism. If restructuring the step order makes exact equality impossible, explain why in the report.
- No gradient flows from any layer's loss into another layer's params (float64 tiny case; the KL target is detached).
- The KL gradient for a lower layer matches float64 autograd on a tiny case.

## Output
`results/iter33_{mode}.json` (with seed>0: `iter33_{mode}_seed{seed}.json`). Keep iter30's fields and add `alpha`, `mean_kl_last10`, `train_peak_mib`, and `references=dict(local_val=1.45627, bp_val=1.38687, attn_local_val=1.41043)`.

## Verify
Smoke both modes with `/c/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B iter33.py --mode MODE --smoke 204800`, plus the alpha=0 check. If your sandbox can't run Python, still write iter33.py completely and skip the smoke. Report in `briefs/iter33_codex_out.md`: a summary, smoke outputs, and any deviation.
