# iter34 implementation report

Created `iter34.py` and `screen34.py`. Existing files were not edited. No training, checkpoint creation, or screening ran successfully.

## Implementation

- `iter34.py`: `--mode {auxhead,auxhead_bp}`, `--hid` (64), `--seed`, `--smoke N`, and `--save-ckpt PATH`. Normal JSON names follow the requested mode/seed convention; smoke JSON retains iter30's temporary-file convention. Existing results/checkpoints are never overwritten.
- Local mode uses manual exact-GELU gradients for each lower layer's training-only nonlinear head. Only its aux CE shapes that layer's backbone. Inference readouts learn their own CE from detached features. Layer inputs are detached; the top layer, copy gate, g mixing, clipping and Adam retain iter27's rules and stored pre-update activations.
- BP control sums all three original inference-readout CEs plus both aux CEs, with flowing backbone gradients. The mixed loss continues to train only g; the suffix copy gate retains its original independent update.
- Head initialization uses a separate CUDA generator seeded by `seed ^ 0x34A0C`: first weights are normal with standard deviation `1/sqrt(256)`, first biases zero, second weights and biases zero. This gives uniform initial aux predictions and exactly zero first-step backbone gradients from the heads. At vocabulary size 96 and hidden width 64, the two heads add 45,376 training parameters.
- Aux heads are absent from all evaluation paths and inference parameter totals. Their count and tensor counts are reported separately, including a combined training parameter total. Results retain iter30 metrics/curve fields and add the requested aux loss, hidden size, reference values, and separate train/eval peaks. `aux_loss_last10` means the final 10% of training steps, explicitly recorded in settings, following iter33's convention.
- Startup checks are implemented for seed-matched iter27 initialization/global RNG equivalence, float64 local isolation, detached inference-readout gradients, nonzero-head aux/backbone derivatives against independent autograd, summed BP derivatives, actual clipped Adam moments, zero-output initialization, frozen-backbone warmup, and resumable sampler equivalence to iter17. These checks could not execute in this sandbox.
- CUDA peaks reset after startup checks. Training time excludes curve/final evaluations and checkpoint serialization. Allocated/reserved maxima are tracked separately for train and eval.
- Checkpoints include all model tensors, manual Adam moments and clocks, copy gate and its optimizer, total schedule length, CPU/CUDA RNG states, training-data fingerprint, sampler RNG, permutation, cursor and remaining targets. Saving the permutation/cursor is necessary because iter17 draws its shuffle up front.
- `screen34.py --make-ckpt` implements iter30 local seed 0 through `ceil(0.15 * full_total_steps)`, with the full original schedule, saving `ckpt/local_s0_p15.pt`.
- `screen34.py --candidate {local,auxhead,topdown}` loads independent copies, runs the paired local branch and candidate on the same 64 batches, and evaluates top CE, final mixed CE including copy, and mixed CE without copy. It uses the first 32 original fixed held-out validation windows, without respacing. It writes `results/screen34_{candidate}.json`, including candidate-minus-local deltas and synchronized wall time. Topdown uses iter33's original teaching ramp and default alpha 0.5.
- Diagnostics compare negative raw lower-backbone gradients against true fully flowing iter30 summed-readout BP gradients on four fixed batches. They exclude aux parameters, inference readouts, Adam and clipping. Fresh zero-output heads have zero norm and undefined cosine, reported as JSON null; an additional diagnostic repeats after warmup with the backbone still at its checkpoint.

## Screening choices

Aux heads receive 32 extra head-only warmup batches; local consumes the identical idle batches. Warmup leaves backbone/gate parameters, moments and clocks fixed, holds the next scheduled LR, and uses a fresh independent head Adam clock. Both subsequent 64-step branches start at `checkpoint.step + 1` on the original schedule. This resolves the unspecified warmup-clock detail without compressing or advancing the backbone schedule during idle batches. This choice is documented in code and output settings.

`--save-ckpt PATH` saves iter34's final training state before final evaluation. The separate screening checkpoint is produced by `screen34.py --make-ckpt` using iter30 local. Neither command was run here.

## Requested smoke attempts

PowerShell equivalents of the supplied commands:

```powershell
& 'C:/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe' -X utf8 -B iter34.py --mode auxhead --smoke 204800
& 'C:/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe' -X utf8 -B iter34.py --mode auxhead_bp --smoke 204800
```

Both returned exit code 1 before launching Python, with the same error:

```text
Program 'python.exe' failed to run: Access is denied
CategoryInfo          : ResourceUnavailable: (:) [], ApplicationFailedException
FullyQualifiedErrorId : NativeCommandFailed
```

Per the supplied fallback instruction, execution was skipped after the sandbox denied Python access. No full training or other GPU commands were attempted. No smoke metrics or result JSON were produced; startup-check success and the target of less than 20 seconds warm remain unverified. The completed files received static review only.
