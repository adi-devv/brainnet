import sys, os, json, hashlib, math, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B, baseline as BL
H = B.HERE
BUDGET = 16_000_000; SEEDS = [0, 1, 2]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
# frequency reference
meta = json.load(open(H+"/data/meta.json"))
cnt = np.array([meta["train_counts"][c] for c in meta["vocab"]], dtype=np.float64)
p = cnt/cnt.sum(); logp = torch.tensor(np.log(p), dtype=torch.float32)
ref = B.evaluate(lambda x: logp.to(x.device).expand(x.size(0), x.size(1), -1))
print("freq reference", ref, flush=True)
bs, T = BL.CFG["batch"], BL.CFG["block"]
lock = dict(benchmark_py_sha256=sha(H+"/benchmark.py"), train_bin_sha256=sha(H+"/data/train.bin"), test_bin_sha256=sha(H+"/data/test.bin"),
  meta_json_sha256=sha(H+"/data/meta.json"), model_config=BL.CFG, block_size=B.BLOCK, eval_windows=B.N_EVAL_WINDOWS,
  batch_size=bs, train_budget_chars=BUDGET, steps=BUDGET//(bs*T), seeds=SEEDS, precision="fp16 autocast + GradScaler",
  baseline_py_sha256=sha(H+"/baseline.py"))
json.dump(lock, open(H+"/LOCK.json", "w"), indent=1)
runs = []
for s in SEEDS:
    r = BL.train(s, BUDGET, log_every_chars=1_600_000); runs.append(r)
    json.dump(dict(partial=True, runs=runs, freq_reference=ref), open(H+"/results/iter1_partial.json", "w"), indent=1)
l = [r["test_loss"] for r in runs]
out = dict(budget_chars=BUDGET, test_loss_per_seed=l, mean=float(np.mean(l)), std=float(np.std(l, ddof=1)), std_note="sample std (ddof=1)",
  freq_reference_loss=ref, params=runs[0]["params"], chars_per_sec=[r["chars_per_sec"] for r in runs],
  seconds_per_seed=[r["train_seconds"] for r in runs], peak_vram_mb=max(r["peak_vram_mb"] for r in runs), runs=runs)
json.dump(out, open(H+"/results/iter1.json", "w"), indent=1)
print("DONE", out["test_loss_per_seed"], out["mean"], out["std"])
