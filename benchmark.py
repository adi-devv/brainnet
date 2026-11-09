"""Locked benchmark: char-level LM on stdlib .py files. Do not modify."""
import os, json, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
BLOCK = 128
N_EVAL_WINDOWS = 2000
def load():
    tr = np.fromfile(HERE+"/data/train.bin", dtype=np.uint8)
    te = np.fromfile(HERE+"/data/test.bin", dtype=np.uint8)
    meta = json.load(open(HERE+"/data/meta.json"))
    return tr, te, len(meta["vocab"])
def eval_windows(te):
    # fixed deterministic, evenly spaced starts; each window = BLOCK inputs + BLOCK targets
    n = len(te) - BLOCK - 1
    starts = np.linspace(0, n, N_EVAL_WINDOWS).astype(np.int64)
    x = np.stack([te[s:s+BLOCK] for s in starts]).astype(np.int64)
    y = np.stack([te[s+1:s+BLOCK+1] for s in starts]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)
@torch.no_grad()
def evaluate(model_fn, device="cuda", bs=100):
    """model_fn(x[B,BLOCK] long) -> logits [B,BLOCK,V]. Returns mean CE nats/char over all
    BLOCK positions of the fixed test windows (position t sees only chars <= t)."""
    _, te, V = load()
    x, y = eval_windows(te)
    tot, cnt = 0.0, 0
    for i in range(0, len(x), bs):
        xb, yb = x[i:i+bs].to(device), y[i:i+bs].to(device)
        lg = model_fn(xb).float()
        tot += torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), yb.reshape(-1), reduction="sum").item()
        cnt += yb.numel()
    return tot / cnt
