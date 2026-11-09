import sys, os, time, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B
CFG = dict(width=128, layers=6, heads=4, block=B.BLOCK, batch=64, lr=3e-3, min_lr=3e-4, wd=0.01, warmup=100, ffn_mult=4)

class Block(nn.Module):
    def __init__(s, d, h, m):
        super().__init__(); s.h = h
        s.ln1 = nn.LayerNorm(d); s.qkv = nn.Linear(d, 3*d); s.proj = nn.Linear(d, d)
        s.ln2 = nn.LayerNorm(d); s.f1 = nn.Linear(d, m*d); s.f2 = nn.Linear(m*d, d)
    def forward(s, x):
        Bz, T, D = x.shape
        q, k, v = s.qkv(s.ln1(x)).view(Bz, T, 3, s.h, D//s.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(Bz, T, D)
        x = x + s.proj(a)
        return x + s.f2(F.gelu(s.f1(s.ln2(x))))

class GPT(nn.Module):
    def __init__(s, V, c=CFG):
        super().__init__()
        s.tok = nn.Embedding(V, c["width"]); s.pos = nn.Embedding(c["block"], c["width"])
        s.blocks = nn.ModuleList(Block(c["width"], c["heads"], c["ffn_mult"]) for _ in range(c["layers"]))
        s.lnf = nn.LayerNorm(c["width"]); s.head = nn.Linear(c["width"], V)
        s.apply(s._init)
    def _init(s, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear): nn.init.zeros_(m.bias)
    def forward(s, idx):
        x = s.tok(idx) + s.pos(torch.arange(idx.size(1), device=idx.device))
        for b in s.blocks: x = b(x)
        return s.head(s.lnf(x))

def train(seed, budget_chars, log_every_chars=None, device="cuda", verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, te, V = B.load()
    tr_t = torch.from_numpy(tr.astype(np.int64)).to(device)
    m = GPT(V).to(device)
    nparams = sum(p.numel() for p in m.parameters())
    decay = [p for p in m.parameters() if p.dim() >= 2]; nod = [p for p in m.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=CFG["wd"]), dict(params=nod, weight_decay=0)], lr=CFG["lr"], betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler()
    bs, T = CFG["batch"], CFG["block"]; cps = bs*T
    steps = budget_chars // cps
    log_every = max(1, (log_every_chars or budget_chars//10) // cps)
    g = torch.Generator(device=device); g.manual_seed(seed)
    def fn(x):
        with torch.autocast("cuda", dtype=torch.float16): return m(x)
    curve = []; ar = torch.arange(T, device=device)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time(); train_time = 0.0
    for step in range(1, steps+1):
        lr = CFG["lr"]*step/CFG["warmup"] if step < CFG["warmup"] else CFG["min_lr"] + 0.5*(CFG["lr"]-CFG["min_lr"])*(1+math.cos(math.pi*(step-CFG["warmup"])/max(1, steps-CFG["warmup"])))
        for gr in opt.param_groups: gr["lr"] = lr
        st = torch.randint(0, len(tr_t)-T-1, (bs,), device=device, generator=g)
        ix = st[:, None] + ar[None]
        x, y = tr_t[ix], tr_t[ix+1]
        with torch.autocast("cuda", dtype=torch.float16):
            loss = F.cross_entropy(m(x).float().view(-1, V), y.view(-1))
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
        scaler.unscale_(opt); nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        if step % log_every == 0 or step == steps:
            torch.cuda.synchronize(); train_time = time.time()-t0
            m.eval(); tl = B.evaluate(fn); m.train()
            curve.append(dict(chars=step*cps, train_loss=loss.item(), test_loss=tl, elapsed_s=train_time))
            t0 = time.time() - train_time  # exclude eval time from train clock
            if verbose: print(f"seed{seed} chars={step*cps} train={loss.item():.4f} test={tl:.4f} t={train_time:.0f}s", flush=True)
    torch.cuda.synchronize()
    # train_time measured excluding evals: recompute
    m.eval(); final = B.evaluate(fn)
    return dict(seed=seed, test_loss=final, curve=curve, params=nparams, train_seconds=train_time,
                chars_per_sec=steps*cps/train_time, peak_vram_mb=torch.cuda.max_memory_allocated()/2**20, chars_trained=steps*cps)

if __name__ == "__main__":
    seed = int(sys.argv[1]); budget = int(float(sys.argv[2]))
    r = train(seed, budget); print(json.dumps({k: v for k, v in r.items() if k != "curve"}))
