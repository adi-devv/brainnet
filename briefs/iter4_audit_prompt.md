# Audit: were the variant runs valid?

Work in `D:\Aadit\claudez\brainnet`. This is read-only: do not change, create or delete anything. You may run short read-only Python analysis using `"C:\Users\aadit\AppData\Local\Programs\Python\Python312\python.exe" -X utf8 -c ...`, CPU only, at most 2 threads, about 5 minutes total. No network, no pip, no git.

Project: a no-backprop char-level language model on Python stdlib source, scored in nats/char by `benchmark.py`. Read `benchmark.py`, `prepare_data.py`, `baseline.py`, `hybrid.py`, `hybrid_gpu.py`, `hybrid_novelty.py`, `run_iter3.py`, `hybrid_mixer.py`, `briefs/iter2_hybrid.md`, `briefs/iter3_novelty.md`, `briefs/iter4_mixer.md`, and every `results/iter2_*`, `results/iter3_*`, `results/iter4_*` file.

Claims under review (test loss):
- Backprop GPT at 1M chars: 2.3303. Plain counts: 1.6727.
- Hebbian on GPU with bs=976: random_keys 2.2828, hebb_memory 1.9740, full 2.2008, full_k32 2.2480, full_K8192 2.0237. CPU with bs=64: 2.2828, 2.1510, 2.2429, 2.1453, 2.0399.
- Novelty on `full`: novel_scale 2.2046 vs base 2.2429; the gate is worse.
- Logistic mixer: first run at lr 0.02 scored 2.2206; after rerun at lr 0.0003125 `mix_two_layer` scored 1.3803, vs Hedge 1.6727. `iter4_mix_*_lr*.json` hold these.

Look for anything that makes these numbers wrong or misleading, and verify each finding in the code or data. In particular:
1. Leakage: near-duplicate or shared content between train and test. The split is every 10th file. Measure how much of `test.bin` is covered by long substrings (for example 32+ chars) that appear in `train.bin`, and how the count model's loss depends on it. Which test files dominate, and are they generated tables or data?
2. Causality and in-sample errors in mixer training: are inputs truly out-of-sample, is state frozen at eval, can batch-internal ordering leak, are there pad-handling bugs?
3. The summed-batch delta, and whether the 0.02 runs were simply unstable.
4. Whether picking the lr by test loss matters here, and how much.
5. Whether the CPU vs GPU comparisons are fair: batch size, step count, device, the random keys matching exactly while others differ, threads, CPU contention from parallel runs.
6. Why `hedge_recheck` matches exactly, and whether it really shares the mixer's code path.
7. Whether the backprop baseline and these models see the same data budget and evaluation.
8. Any bug in `hybrid_gpu.py` or `hybrid_mixer.py` that changes results.

Reply with at most 25 lines, most serious first. For each finding: severity, evidence (file:line or a measured number), and its effect on which claim. Say plainly which claims hold. No praise, no suggestions beyond what the findings need.
