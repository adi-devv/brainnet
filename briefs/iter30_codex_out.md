# iter30 controls

Added `iter30.py`. Existing files were not edited. No full training ran.

- `local` inherits iter27's training method, forward, local derivatives, interleaved update order and manual Adam.
- `backprop` sums the three per-layer mean CE losses, allows gradients through every layer boundary, runs one backward, and applies one manual Adam sweep to all network tensors. The mixed-logit loss still drives only `g`; the copy gate keeps its original loss and optimizer.
- `attn` retains the EMAs and lags and appends a causal 4 x 64 softmax attention output to layer 2's input. Q/K use interleaved RoPE, base 10000; Q/K/V are bias-free, normal / sqrt(256); O starts at zero. The 256 extra W2 rows use normal / sqrt(1024). All additions use a separate generator seeded with `seed ^ 0x30A77E`; original parameters and global RNG draws are preserved. The original and extra W2 products are computed separately to preserve the original GEMM at O=0. Top-layer autograd receives detached layer-1 activations and updates only the top layer and attention tensors.
- Curves use the first quarter of the fixed validation windows at steps `ceil(k * total_steps / 10)`, k=1..10 (duplicate milestones coalesce for very short runs). Normal runs use 500 of 2000 windows; smokes retain iter27's 16-window convention and use the first four. Timing synchronizes at boundaries and excludes all evaluation work from training time. `eval_seconds` includes subset preparation, curve evaluations and every final evaluation.
- Final evaluation calls and fields are retained from iter27. Results add mode, curves, evaluation timing, parameter counts (including gate and attention breakdown), peak allocated/reserved GPU memory and the requested iter27 reference. Normal output is `results/iter30_{mode}[_seedN].json`, opened exclusively. Smoke JSON is temporary and cleaned, following iter27's convention, so smoke results cannot occupy a full-run filename.

Startup checks are implemented but could not be executed here. They compare every original parameter and the global RNG against a fresh seed-matched iter27 model, verify zero Adam state and initial forward equivalence, and check causality at every layer. Tiny masked-target float64 cases compare detached backprop gradients against iter27's analytic local derivatives, require nonzero cross-layer terms in both lower layers, and verify one actual backprop update uses the summed gradients in one sweep. Attention's top gradients are compared against an independent explicit float64 attention/RoPE implementation; nonzero O on a disposable copy exercises Q/K/V derivatives and causality. The top-loss graph must leave every lower parameter and `g` unused. Checks also verify the actual top update and preserve run parameters and RNG.

## Smoke verification

Attempted the requested local smoke using the same interpreter and arguments, with its native Windows path in PowerShell:

```powershell
& 'C:\Users\aadit\AppData\Local\Programs\Python\Python312\python.exe' -X utf8 -B iter30.py --mode local --smoke 204800
```

Output (exit status 1, before Python launched):

```text
Program 'python.exe' failed to run: Access is denied
CategoryInfo          : ResourceUnavailable: (:) [], ApplicationFailedException
FullyQualifiedErrorId : NativeCommandFailed
```

| Mode | Outcome |
|---|---|
| local | Interpreter launch denied by the sandbox; no Python output or metrics |
| backprop | Skipped after the interpreter denial, as authorized in the request |
| attn | Skipped after the interpreter denial, as authorized in the request |

No training, startup checks, numerical checks or Python syntax checks ran. Static review covered the inherited update paths, losses, detach boundaries, tensor shapes, initialization, schedule, clipping, evaluation fields and timing. Runtime correctness and numerical tolerances remain unverified until the supplied smoke commands run outside this sandbox.

No intended model-spec deviations. The explicitly recorded smoke-storage convention matches iter27.
