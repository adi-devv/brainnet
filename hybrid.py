"""Iteration 2: Hebbian feature layer + sparse written memory, learning in the forward pass only.
No error signal is chained through layers. One scalar per prediction (surprise) gates learning."""
import sys, json, time, numpy as np, torch
import torch.nn.functional as F
import benchmark as B
torch.set_grad_enabled(False)

LENS = (2, 4, 8, 16, 32)
VARIANTS = {
    "count_ngram": dict(mode="ngram", surprise=False),
    "random_keys": dict(mode="random", surprise=False),
    "hebb_memory": dict(mode="hebb", surprise=False),
    "full":        dict(mode="hebb", surprise=True),
    "full_k32":    dict(mode="hebb", surprise=True, k=32),
    "full_K8192":  dict(mode="hebb", surprise=True, K=8192),
}


class Hybrid:
    def __init__(s, V, mode="hebb", surprise=True, lens=LENS, K=2048, k=8, D=128, beta=16.0, tau=0.6,
                 lr=0.05, alpha=0.5, eta=0.3, share=1e-3, seed=0):
        g = torch.Generator(); g.manual_seed(seed)
        s.V, s.mode, s.surprise, s.lens, s.K, s.k, s.D = V, mode, surprise, tuple(lens), K, k, D
        s.beta, s.tau, s.lr, s.alpha, s.eta, s.share = beta, tau, lr, alpha, eta, share
        s.hebb = mode == "hebb"
        L = len(s.lens)
        s.E = torch.randn(V + 1, D, generator=g) / D ** 0.5      # fixed character vectors (row V = pad)
        s.E[V] = 0
        s.P = torch.randint(0, 2, (max(s.lens), D), generator=g).float() * 2 - 1  # fixed position binders
        s.logw = torch.zeros(L)
        s.uni = torch.zeros(V)
        s.ema, s.t = torch.tensor(3.0), 0
        if mode == "ngram":
            s.dict = [dict() for _ in s.lens]
            s.M = [torch.zeros(1 << 16, V) for _ in s.lens]       # row 0 stays zero (unseen context)
        else:
            s.W = [F.normalize(torch.randn(K, D, generator=g), dim=-1) for _ in s.lens]
            s.M = [torch.zeros(K, V) for _ in s.lens]
            s.last = [torch.full((K,), -1.0) for _ in s.lens]

    def state_hash(s):
        parts = [s.logw, s.uni, s.ema.view(1)] + list(s.M) + (list(s.W) if s.mode != "ngram" else [])
        return float(sum(p.double().abs().sum() for p in parts)) + (sum(len(d) for d in s.dict) if s.mode == "ngram" else 0)

    def _codes(s, x):
        """Per length n: fixed code of the last n chars at every position (position t uses x[..t] only)."""
        Bn, T = x.shape; mx = max(s.lens)
        xp = torch.cat([torch.full((Bn, mx - 1), s.V, dtype=torch.long), x], 1)
        ng = s.mode == "ngram"
        h = torch.zeros(Bn, T, dtype=torch.long) if ng else None
        acc = None if ng else torch.zeros(Bn, T, s.D)
        out = {}
        for j in range(1, mx + 1):
            tok = xp[:, mx - j: mx - j + T]
            if ng: h = h * 1000003 + tok + 1
            else: acc += s.P[j - 1] * s.E[tok]
            if j in s.lens: out[j] = h.clone() if ng else F.normalize(acc, dim=-1)
        return out

    def _lookup(s, i, code, learn):
        if s.mode == "ngram":
            d, rows = s.dict[i], []
            for h in code.tolist():
                r = d.get(h)
                if r is None:
                    r = 0
                    if learn: r = len(d) + 1; d[h] = r
                rows.append(r)
            if learn and len(d) + 1 >= s.M[i].shape[0]:
                nm = torch.zeros(2 * s.M[i].shape[0], s.V); nm[:s.M[i].shape[0]] = s.M[i]; s.M[i] = nm
            idx = torch.tensor(rows).unsqueeze(1)
            return idx, torch.ones(len(rows), 1)
        W = s.W[i]
        sims = code @ W.T
        if learn and s.hebb:
            un = (sims.max(1).values < s.tau).nonzero().flatten()    # unexplained inputs claim stale neurons
            if len(un):
                new = torch.topk(s.last[i], len(un), largest=False).indices
                W[new] = code[un]; s.M[i][new] = 0; s.last[i][new] = s.t
                sims[:, new] = code @ W[new].T
        v, idx = torch.topk(sims, s.k, dim=1)
        return idx, torch.exp(s.beta * (v - v[:, :1]))

    def _predict(s, cod, learn):
        N = len(cod[s.lens[0]])
        p = ((s.uni + 1) / (s.uni.sum() + s.V)).expand(N, s.V)
        ps, look = [], []
        for i, n in enumerate(s.lens):
            idx, a = s._lookup(i, cod[n], learn)
            c = (a.unsqueeze(-1) * s.M[i][idx]).sum(1)                 # read: sum rows of active neurons
            p = (c + s.alpha * p) / (c.sum(1, keepdim=True) + s.alpha)  # back off to shorter context
            ps.append(p); look.append((idx, a))
        L = len(s.lens)
        w = (1 - s.share) * torch.softmax(s.logw, 0) + s.share / L
        return (torch.stack(ps, 1) * w[None, :, None]).sum(1), ps, look

    def step(s, cod, y):
        N = len(y); ar = torch.arange(N)
        mix, ps, look = s._predict(cod, True)
        lp = torch.stack([p[ar, y] for p in ps], 1)
        s.logw = s.logw + s.eta * lp.log().mean(0)                      # Hedge: multiplicative weights
        s.logw = s.logw - s.logw.max()
        surp = -mix[ar, y].log()                                        # the one scalar per prediction
        s.ema = 0.999 * s.ema + 0.001 * surp.mean()
        g = (surp / s.ema).clamp(0.25, 3.0) if s.surprise else torch.ones(N)
        s.uni.index_add_(0, y, torch.ones(N))
        for i, n in enumerate(s.lens):
            idx, a = look[i]
            flat = (idx * s.V + y[:, None]).flatten()
            s.M[i].view(-1).index_add_(0, flat, (a * g[:, None]).flatten())   # write true next char
            if s.hebb:
                W, fi = s.W[i], idx.flatten()
                cnt = torch.zeros(s.K).index_add_(0, fi, torch.ones(len(fi)))
                step_size = (s.lr * g[:, None] * a / cnt[idx])
                W.index_add_(0, fi, (step_size.unsqueeze(-1) * (cod[n][:, None, :] - W[idx])).reshape(-1, s.D))
                r = fi.unique(); W[r] = F.normalize(W[r], dim=-1)
                s.last[i][fi] = s.t
        s.t += 1

    def train(s, tr, budget, seed, bs=64, T=128):
        g = torch.Generator(); g.manual_seed(seed)
        trt = torch.from_numpy(tr.astype(np.int64)); ar = torch.arange(T)
        for _ in range(budget // (bs * T)):
            st = torch.randint(0, len(tr) - T - 1, (bs,), generator=g)
            x, y = trt[st[:, None] + ar], trt[st[:, None] + ar + 1]
            cods = s._codes(x)
            for t in range(T):
                s.step({n: c[:, t] for n, c in cods.items()}, y[:, t])

    def logprobs(s, x, chunk=4096):
        """Frozen forward pass: [B,T] long -> log-probs [B,T,V]. Reads memory only; nothing is written."""
        Bn, T = x.shape
        cods = {n: c.reshape(Bn * T, *c.shape[2:]) for n, c in s._codes(x).items()}
        out = []
        for i in range(0, Bn * T, chunk):
            mix, _, _ = s._predict({n: c[i:i + chunk] for n, c in cods.items()}, False)
            out.append(mix.log())
        return torch.cat(out).reshape(Bn, T, s.V)


def run(name, seed=0, budget=999_424, threads=2, **over):
    torch.set_num_threads(threads); torch.manual_seed(seed)
    tr, te, V = B.load()
    cfg = dict(VARIANTS[name]); cfg.update(over)
    m = Hybrid(V, seed=seed, **cfg)
    t0 = time.time(); m.train(tr, budget, seed); train_s = time.time() - t0
    h0 = m.state_hash()
    t0 = time.time(); loss = B.evaluate(m.logprobs, device="cpu"); eval_s = time.time() - t0
    assert m.state_hash() == h0, "model changed during eval"
    res = dict(name=name, test_loss=loss, train_seconds=train_s, eval_seconds=eval_s,
               chars_per_sec=budget / train_s, train_chars=budget, seed=seed, threads=threads,
               settings={**cfg, "lens": list(LENS)}, mixing_weights=torch.softmax(m.logw, 0).tolist())
    json.dump(res, open(f"{B.HERE}/results/iter2_{name}.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1]), indent=1), flush=True)
