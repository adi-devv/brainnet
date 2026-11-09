import re, math, sys, torch, numpy as np
import benchmark as B
from hybrid import Hybrid
torch.set_num_threads(2)
res = {}

def nll(m, x, y):
    lp = m.logprobs(x)
    return -lp.gather(-1, y.unsqueeze(-1)).mean().item()

def train_on(m, data, steps, V, bs=64, T=128, seed=0):
    m.train(data, steps * bs * T, seed, bs, T)

for mode in ("hebb", "ngram", "random"):
    pass

for mode in ("hebb", "ngram"):
    V = 16
    # 1 repeating pattern
    pat = np.tile(np.array([3, 7, 1, 9, 4, 4, 2], dtype=np.uint8), 5000)
    m = Hybrid(V, mode=mode, K=256, lens=(2, 4, 8))
    train_on(m, pat, 6, V)
    x = torch.from_numpy(pat[:128 * 20].astype(np.int64)).reshape(20, 128)[:, :]
    y = torch.from_numpy(pat[1:128 * 20 + 1].astype(np.int64)).reshape(20, 128)
    l = nll(m, x, y); res[f"{mode}_pattern_loss"] = l
    # 2 random uniform
    rs = np.random.RandomState(1)
    rnd = rs.randint(0, V, 20_000_000).astype(np.uint8)
    m2 = Hybrid(V, mode=mode, K=256, lens=(2, 4, 8))
    train_on(m2, rnd, 12, V)
    te = rs.randint(0, V, 128 * 40 + 1).astype(np.int64)
    xt = torch.from_numpy(te[:-1]).reshape(40, 128); yt = torch.from_numpy(te[1:]).reshape(40, 128)
    res[f"{mode}_random_loss"] = nll(m2, xt, yt); res[f"{mode}_logV"] = math.log(V)
    # 3 causality
    xa = xt[:10].clone(); xb = xa.clone(); t = 60
    xb[:, t + 1:] = torch.randint(0, V, xb[:, t + 1:].shape)
    la, lb = m2.logprobs(xa), m2.logprobs(xb)
    res[f"{mode}_causal_maxdiff_le_t"] = (la[:, :t + 1] - lb[:, :t + 1]).abs().max().item()
    res[f"{mode}_causal_maxdiff_after_t"] = (la[:, t + 1:] - lb[:, t + 1:]).abs().max().item()
    # 4 frozen eval
    h0 = m2.state_hash(); a = nll(m2, xt, yt); b = nll(m2, xt, yt); h1 = m2.state_hash()
    res[f"{mode}_frozen_same_number"] = a == b; res[f"{mode}_frozen_state_unchanged"] = h0 == h1

# 5 source check
src = open("hybrid.py", encoding="utf8").read()
banned = [r"\.backward\s*\(", r"autograd", r"torch\.func", r"optim", r"requires_grad", r"\bgrad\(", r"set_grad_enabled\(True"]
found = [b for b in banned if re.search(b, src)]
res["source_has_no_banned_calls"] = not found; res["source_found"] = found
res["source_has_grad_disabled"] = "torch.set_grad_enabled(False)" in src
res["source_lines"] = src.count("\n")

ok = True
for mode in ("hebb", "ngram"):
    ok &= res[f"{mode}_pattern_loss"] < 0.1
    ok &= abs(res[f"{mode}_random_loss"] - res[f"{mode}_logV"]) < 0.05
    ok &= res[f"{mode}_causal_maxdiff_le_t"] == 0.0 and res[f"{mode}_causal_maxdiff_after_t"] > 0
    ok &= res[f"{mode}_frozen_same_number"] and res[f"{mode}_frozen_state_unchanged"]
ok &= res["source_has_no_banned_calls"] and res["source_has_grad_disabled"]
for k, v in res.items(): print(k, v)
print("ALL_PASS", bool(ok))
sys.exit(0 if ok else 1)
