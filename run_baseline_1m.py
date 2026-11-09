"""Backprop baseline at the 1M-character quick-test budget.

Warmup is scaled to the short run (10% of steps), otherwise the locked 100-step
warmup would use up most of the 122 steps and handicap backprop.
"""
import json
import os
import baseline as M

BUDGET = 1_000_000
STEPS = BUDGET // (M.CFG["batch"] * M.CFG["block"])
M.CFG["warmup"] = max(1, STEPS // 10)

out = []
for seed in (0, 1, 2):
    r = M.train(seed, BUDGET)
    out.append({k: v for k, v in r.items() if k != "curve"})
    print(f"seed{seed} test={r['test_loss']:.4f} train_s={r['train_seconds']:.1f}", flush=True)

losses = [r["test_loss"] for r in out]
mean = sum(losses) / len(losses)
summary = {"budget_chars": BUDGET, "warmup": M.CFG["warmup"], "mean_test_loss": mean, "runs": out}
os.makedirs("results", exist_ok=True)
with open("results/baseline_1m.json", "w") as f:
    json.dump(summary, f, indent=1)
print(f"mean test loss {mean:.4f}")
