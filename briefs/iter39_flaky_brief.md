# Fix flaky configured_checks in iter39.py

The full run fails in about 1 of 3 fresh processes, at `configured_checks` (iter39.py ~line 1428): "configured outputs differ between identical runs". iter38 never fails (0/6).
- Repro: `../dbg39c.py` and `../dbg39b.py`, run from this folder with CAP=0. The sequence is checks(), then storage_checks('cpu'), then configured_checks(...).
- When it fails, maxdiff is 1.9e-6 over 182-376 elements of the final score (2,12,96).
- `../dbg39d.py` shows that all table count/tag tensors and mixer tensors are bit-identical across the 2 repeats, at every one of the 4 learn steps (6/6 processes). So the divergence is in scoring/reads (the eval score after set_collect(False)), or something new in iter39: the TorchScript `sequential_victims`, device-side stats, or a float reduction order.

Task:
1. Find the root cause.
2. Fix it in iter39.py so the scores are bit-deterministic, while keeping bit-exactness with iter38's tables. Don't weaken the assert.
3. Also review iter39's write path for any other race or nondeterminism.

If your sandbox can't run Python, say what you would run. Report in briefs/iter39_flaky_out.md: the cause, the fix, and the diff.
