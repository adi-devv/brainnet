"""Closed-form predictive-state codes; python iter6.py {mix_psc,mix_rand} [--budget N]."""
import argparse, json, time
import numpy as np
import torch
import iter5
torch.set_grad_enabled(False)
F, G, SIZE, BITS = 2**16, 2**12, 2**23, (8, 12, 16, 20, 24)
RULES = ['Reference mix_full: test=1.2528, renamed=1.7547, gap=0.502.',
         'Drop unless mix_psc beats mix_full by >=0.05 test nats.',
         'Learning helps only if mix_psc beats mix_rand by >=0.03 test nats.',
         'A renamed_loss - test_loss below 0.502 means less dependence on memorised names.']

def mix(h):
    h = (h ^ (h >> 16)) * 1000003
    return (h ^ (h >> 23)) * 1000033

def phi(p, V):
    groups = torch.tensor(list(range(8)) + [8]*8 + [9]*16, device=p.device)
    chars = mix(groups * 1000003 + p + 1) & (F-1)
    pairs = p[..., :8] * (V+1) + p[..., 1:9]
    pairs = mix((torch.arange(8, device=p.device)+10)*1000003 + pairs + 1) & (F-1)
    return torch.cat([chars, pairs], -1)

def features(x, V):
    p = torch.cat([torch.full((len(x), 31), V, device=x.device), x], 1)
    return phi(p.unfold(1, 32, 1).flip(-1), V)

def past(data, t, V):
    ix = t[:, None] - 1 - torch.arange(32, device='cuda')
    available = (ix >= 0) & (ix >= (t[:, None]//128)*128-1)
    return phi(torch.where(available, data[ix.clamp_min(0)], V), V)

def stream(data, V):
    for a in range(0, len(data), 2048):
        t = torch.arange(a, min(a+2048, len(data)), device='cuda')
        ix = t[:, None] + torch.arange(4, device='cuda')
        y = torch.where(ix < len(data), data[ix.clamp_max(len(data)-1)], V)
        g = mix(torch.arange(4, device='cuda')*1000003 + y + 1) & (G-1)
        tri = mix(4000012 + (y[:, 0]*(V+1)+y[:, 1])*(V+1)+y[:, 2]) & (G-1)
        yield past(data, t, V), torch.cat([g, tri[:, None]], 1)

def embed(f, W, scale):
    return (W[f] * scale[f, None]).sum(-2)

def learn(data, V, learned):
    cf, cg = torch.zeros(F, device='cuda'), torch.zeros(G, device='cuda')
    for f, g in stream(data, V):
        for count, ix in ((cf, f), (cg, g)):
            count.index_add_(0, ix.flatten(), torch.ones(ix.numel(), device='cuda'))
    sf, sg = (cf+1).rsqrt(), (cg+1).rsqrt()
    def cross(R, transpose=False):
        out = torch.zeros((G if transpose else F, R.shape[1]), device='cuda')
        for f, g in stream(data, V):
            src, dst, ss, ds = (f, g, sf, sg) if transpose else (g, f, sg, sf)
            v = embed(src, R, ss)
            out.index_add_(0, dst.flatten(), (v[:, None, :]*ds[dst, None]).reshape(-1, R.shape[1]))
        return out
    rng = torch.Generator(device='cuda').manual_seed(0); singular = []
    if learned:
        Q = torch.linalg.qr(cross(torch.randn(G, 80, generator=rng, device='cuda')), mode='reduced')[0]
        Q = torch.linalg.qr(cross(cross(Q, True)), mode='reduced')[0]
        U, S, _ = torch.linalg.svd(cross(Q, True).T, full_matrices=False)
        W, singular = (Q @ U[:, 1:65]).contiguous(), S[:64].tolist()
    else: W = torch.randn(F, 64, generator=rng, device='cuda')
    rng.manual_seed(0)
    sample = torch.randint(len(data), (min(32768, len(data)),), generator=rng, device='cuda')
    codes = torch.cat([embed(past(data, t, V), W, sf) for t in sample.split(1024)])
    return W, sf, codes.median(0).values, singular

class CodeCounts(iter5.Counts):
    def __init__(self, V):
        self.V, self.ppm, self.lens = V, False, BITS
        self.C = [torch.zeros(SIZE, dtype=torch.int32, device='cuda') for _ in BITS]
        self.T = [torch.zeros_like(c) for c in self.C]
        self.uni = torch.zeros(V, dtype=torch.int32, device='cuda')
        self.chars = torch.arange(V, device='cuda')
    def predict(self, codes):
        p = ((self.uni.float()+1)/(self.uni.sum()+self.V)).expand(len(codes[8]), self.V); ps = []
        for i, b in enumerate(BITS):
            ix, tag = iter5.slot(codes[b][:, None]*1000003+self.chars+1); ix = ix & (SIZE-1)
            c = (self.C[i][ix]*(self.T[i][ix] == tag)).float()
            p = (c + .5*p)/(c.sum(1, keepdim=True)+.5); ps.append(p)
        return ps
    def write(self, codes, y):
        one = torch.ones_like(y, dtype=torch.int32); self.uni.index_add_(0, y, one)
        for i, b in enumerate(BITS):
            ix, tag = iter5.slot(codes[b]*1000003+y+1); ix = ix & (SIZE-1)
            self.C[i][ix[self.T[i][ix] != tag]] = 0
            self.T[i][ix] = tag; ok = self.T[i][ix] == tag
            self.C[i].index_add_(0, ix[ok], one[ok])

class Model(iter5.Model):
    def __init__(self, V, W, scale, median):
        super().__init__(V, 1e-4)
        self.W, self.scale, self.median, self.code = W, scale, median, CodeCounts(V)
        self.first = [iter5.Neuron(r, 10, self.lr) for r in (1, V+1, 4096)]
        for n in self.first: n.w = n.w.cuda()
    def keys(self, x):
        out = []
        for part in x.split(8):
            code = embed(features(part, self.V), self.W, self.scale)
            out.append(((code > self.median).long() * (1 << torch.arange(64, device='cuda'))).sum(-1))
        bits = torch.cat(out)
        return {b: mix(bits & ((1 << b)-1)) for b in BITS}
    def output(self, cod, selectors, hc=None, y=None, unseen=None):
        return self.mix(self.base.predict(cod) + self.code.predict(hc), selectors, y)
    def logprobs(self, x, unseen=None):
        cod = {k: v.flatten() for k, v in self.base.codes(x).items()}
        hc = {k: v.flatten() for k, v in self.keys(x).items()}; sel = [s.flatten() for s in self.selectors(x)]
        out = [self.output({k: v[a:a+1024] for k, v in cod.items()}, [s[a:a+1024] for s in sel],
                          {k: v[a:a+1024] for k, v in hc.items()}) for a in range(0, x.numel(), 1024)]
        return torch.cat(out).reshape(*x.shape, self.V)

def fit(m, tr, budget):
    torch.cuda.synchronize(); start = time.perf_counter(); chars = 0
    for x, y in iter5.batches(tr, budget, 976, True, m.V):
        cod, hc, sel = m.base.codes(x), m.keys(x), m.selectors(x)
        for t in range(x.shape[1]):
            keep = y[:, t] >= 0; target = y[:, t][keep]
            if not len(target): continue
            c = {k: v[:, t][keep] for k, v in cod.items()}; h = {k: v[:, t][keep] for k, v in hc.items()}
            m.output(c, [s[:, t][keep] for s in sel], h, target)
            m.base.write(c, target); m.code.write(h, target)
        chars += int((y >= 0).sum())
    torch.cuda.synchronize(); return time.perf_counter()-start, chars

def sanity(data, V, W, scale):
    vocab = json.loads((iter5.ROOT/'data/meta.json').read_text())['vocab']; enc = {c: i for i, c in enumerate(vocab)}
    raw = bytes(data.cpu().tolist()); contexts = ('def ', 'return ', 'self.', 'for ', 'if ', 'import ', 'class ', ' = ')
    positions = []
    for i, s in enumerate(contexts):
        loc = raw.find(bytes(enc[c] for c in s), 24)
        positions.append(loc+len(s) if 0 <= loc < len(raw)-len(s) else min(len(raw)-1, 24+i*(len(raw)-25)//8))
    query = embed(past(data, torch.tensor(positions, device='cuda'), V), W, scale)
    best = torch.full((8, 5), float('inf'), device='cuda'); ids = torch.zeros(8, 5, dtype=torch.long, device='cuda')
    for a in range(0, len(data), 1024):
        t = torch.arange(a, min(a+1024, len(data)), device='cuda')
        d = (query[:, None, :] - embed(past(data, t, V), W, scale)[None, :, :]).square().sum(-1)
        d.masked_fill_(t[None, :] == torch.tensor(positions, device='cuda')[:, None], float('inf'))
        best, take = torch.cat([best, d], 1).topk(5, largest=False)
        ids = torch.cat([ids, t.expand(8, -1)], 1).gather(1, take)
    decode = lambda t: ''.join(vocab[c] for c in raw[max(0, t-24):t])
    return [dict(context=decode(t), position=t, nearest=[dict(position=j, context=decode(j)) for j in row])
            for t, row in zip(positions, ids.tolist())]

def run(name, budget=None):
    if budget is not None and budget < 128: raise ValueError('budget must be >=128')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.cuda.set_per_process_memory_fraction(2.7*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(0); torch.cuda.reset_peak_memory_stats()
    tr, te, V = iter5.B.load(); cut = int(len(tr)*.95); val = tr[cut:]; tr = tr[:cut]
    if budget is not None: tr = tr[:budget]
    data = torch.as_tensor(tr.astype(np.int64), device='cuda'); torch.cuda.synchronize(); start = time.perf_counter()
    W, scale, median, singular = learn(data, V, name == 'mix_psc')
    torch.cuda.synchronize(); fit_seconds = time.perf_counter()-start
    extra = dict(singular_values=singular, sanity=sanity(data, V, W, scale)) if name == 'mix_psc' else {}
    del data; torch.cuda.empty_cache(); m = Model(V, W, scale, median)
    train_seconds, chars = fit(m, tr, budget); renamed, info = iter5.rename(te); before = iter5.stamp(m)
    losses = {k: iter5.evaluate(m.logprobs, d) for k, d in [('test_loss', te), ('renamed_loss', renamed), ('val_loss', val)]}
    assert iter5.stamp(m) == before, 'evaluation wrote state'
    settings = dict(seed=0, device='cuda', bs=976, T=128, lr=1e-4, sampling='one_pass', budget=budget,
                    train_end=len(tr), val_start=cut, fit_chars=len(tr), F=F, G=G, rank=80, power_iterations=1,
                    retained_dimensions=[1, 64], median_sample=min(32768, len(tr)), bits=BITS, alpha=.5,
                    exact_entries=iter5.SIZE, code_entries=SIZE, mixer='mix_two_layer', layer1_inputs=10,
                    allocator_limit_gib=2.7, decision_rules=RULES, renaming=info, eval_windows=iter5.B.N_EVAL_WINDOWS,
                    feature_context='available tokens; training blocks include preceding char, pad V before block',
                    whitening='occurrence counts + 1, inverse square root; multiplicities retained', nearest_metric='squared Euclidean')
    result = dict(name=name, **losses, fit_seconds=fit_seconds, train_seconds=train_seconds, chars=chars,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, settings=settings, **extra)
    (iter5.ROOT/f'results/iter6_{name}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('name', choices=('mix_psc', 'mix_rand'))
    p.add_argument('--budget', type=int); a = p.parse_args(); print(json.dumps(run(a.name, a.budget), indent=2))
