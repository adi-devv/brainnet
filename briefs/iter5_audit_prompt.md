# Finish the audit, and review iter5.py

Work in `D:\Aadit\claudez\brainnet`. Read-only: do not change, create or delete anything. CPU only, at most 2 threads, about 3 minutes of Python total (the GPU is in use). No network, no pip, no git. Run Python as `"C:\Users\aadit\AppData\Local\Programs\Python\Python312\python.exe" -X utf8 -B`.

An earlier audit ran out of quota before reporting. Its measurements are in `results/iter4_sol_audit.log`, lines 5417-5640; the brief is `briefs/iter4_audit_prompt.md`. Do not repeat them. Already measured: train/test overlap, topics.py share of test, lr 0.02 instability, lr selection effect, causality and frozen eval, CPU vs GPU schedule.

Do only these:
1. Does `hedge_recheck` in `hybrid_mixer.py` really share the mixer's code path, and why does it match 1.6727 exactly?
2. Any bug in `hybrid_gpu.py` or `hybrid_mixer.py` that changes a reported result. The log shows `GPU_NGRAM_MIXED_DEVICES`: does it affect any reported number?
3. Does the backprop baseline see the same evaluation and a comparable data budget as these models?
4. Review `iter5.py` against `briefs/iter5_tests.md`. It is running now, so do not import-run it on the GPU. Look for anything that would make its numbers wrong: count tables and the tag/replace logic in `slot`, `Counts.write`, `Counts.predict`; `one_pass` windows and x/y alignment; predict-before-write; val isolation; whether `mix_full` and `ppm_full` are comparable; `rename` offsets and whether strings stay untouched; the Hebbian inputs; `baseline_full` capturing the right model function.

Reply with at most 20 lines, most serious first. For each finding: severity, evidence (file:line or a measured number), and which result it affects. Say plainly what holds. No praise.
