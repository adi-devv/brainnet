# iter33: detached same-step top-down teaching

Added `iter33.py`; no existing files were edited. Read iter30/iter27, their model,
optimizer, sampler, copy-gate and evaluation dependencies, and all three
`results/iter30_*.json` controls. Their val losses are local 1.4564113111,
backprop 1.3868699436, and attention-local 1.4104292831.

## Implementation

- `--mode {topdown,topdown_bp}`, `--alpha` (default 0.5), `--seed`, and
  `--smoke N`; alpha is validated as finite and in [0, 1].
- `a = alpha * clamp((step / total_steps - .05) / .15, 0, 1)`, with
  1-indexed steps. Lower losses are `(1-a)*CE + a*KL(top || lower)` in
  both modes; the top's own CE and g's mixed-CE rule are unchanged.
- The teacher is the detached layer-2 own-readout distribution from the same
  forward pass, before updates. It excludes g and copy mixing. Both CE and KL
  average over valid target positions (`y >= 0`).
- `topdown` retains all local forward caches until the teacher exists, then
  applies manual gradients in layer order 0, 1, 2. Each upper input is explicitly
  detached, and no backward sweep crosses layers. There is one forward per
  layer. `topdown_bp` uses the same objective with a summed autograd backward
  across the network and one network Adam sweep, as in iter30 backprop.
- Inherits iter27 initialization, widths 256/256/320, sampler, batch 16, T=128,
  step count, LR schedule, Adam, per-tensor clipping, copy gate and g. Retains
  iter30 curve milestones, fixed first-quarter validation subset, final evals,
  result fields, exclusive-create output and temporary smoke-output convention.
- Normal output is `results/iter33_{mode}.json`, with `_seed{seed}` before
  `.json` for positive seeds. Smoke JSON is printed and its temporary file
  removed, matching iter30.
- Adds `alpha`, `mean_kl_last10` (layer 0, then layer 1), `train_peak_mib`,
  and the requested `references`. The KL statistic is the unweighted mean of
  per-step KL means over the final `ceil(0.1 * total_steps)` steps. Also adds
  `eval_peak_mib`, per-phase reserved peaks, and `train_loss_mean` (mean mixed
  network CE over training steps).
- CUDA peak counters reset after startup checks and at each train/eval phase
  boundary. Training peaks exclude curve and final evaluations; eval peaks
  include curve-subset preparation, curves, and final evals. Existing overall
  peak fields report the maximum of these post-check phase peaks. Allocated
  peaks include live baseline allocations; reserved peaks retain allocator-cache
  effects across phases.

## Startup checks implemented (not executed here)

Retains iter30's baseline checks, then adds tiny float64 checks for every
parameter's local weighted-loss gradient at a=0, .37, and 1; isolation of each
layer's loss from every other layer's parameters, including the detached
teacher; and KL logit gradients against autograd with a masked target.

Also checks the BP objective against an independent `F.kl_div` autograd
reference, alpha=0 BP against iter30 backprop, and nonzero cross-layer
derivatives in the BP control. Production-step checks cover both modes with
active teaching, one forward per layer, the same-step KL values, and actual
clipped Adam moments. Three alpha=0 local steps compare losses, parameters,
Adam states and copy-gate state against iter30 local. Checks preserve run state
and CPU/CUDA RNG, and verify the teaching ramp.

Delaying updates does not mathematically prevent alpha=0 equality: parameter
groups are disjoint, and iter27 already sends stored pre-update activations
upward. The alpha=0 branch invokes the original local-gradient routine. CUDA
`index_add_` uses unordered atomics, so differences within GPU nondeterminism
remain possible. Full-smoke numerical equivalence is unverified here.

## Smoke commands and outputs

Used the requested executable through its native Windows path in PowerShell
(equivalent to `/c/Users/...` in a POSIX shell). All four launches were denied
before Python started:

```text
C:/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B iter33.py --mode topdown --smoke 204800
Program 'python.exe' failed to run: Access is denied

C:/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B iter33.py --mode topdown_bp --smoke 204800
Program 'python.exe' failed to run: Access is denied

C:/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B iter33.py --mode topdown --alpha 0 --smoke 204800
Program 'python.exe' failed to run: Access is denied

C:/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B iter30.py --mode local --smoke 204800
Program 'python.exe' failed to run: Access is denied
```

As authorized for this sandbox limitation, smoke execution was skipped after
these failed launches. No training ran, no runtime/syntax-check success is
claimed, and no iter33 result JSON was generated. Static review covered both
training paths, loss derivatives, masking, update order, output fields and
phase-peak transitions. The remaining verification is to run the four commands
above in an environment that can execute the specified Python binary.

No algorithmic deviation from the requested weighted objective is intended.
The only verification deviation is the sandbox's Python execution denial.
