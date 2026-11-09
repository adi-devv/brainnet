"""Recency-weighted window memory and a calibrated causal copy predictor."""
import argparse, json
import torch
import torch.nn.functional as F
import iter8 as R
from hybrid_mixer import Neuron
I = R.I
torch.set_grad_enabled(False)
RULES = ['Reference iter8 fast: test=1.1957, renamed=1.6485, gap=0.4528; backprop at 16M: test=1.1238, gap=0.406.',
         'Stop further fast-memory tuning unless all test <= 1.165.', 'Name dependence fixed only if all renamed_gap < 0.43.',
         'Keep recency only if all beats no_recency by >= 0.01 test; keep copy only if all beats no_copy by >= 0.01 test.']

def recency_weights(match):
    return match.float() * torch.exp2(-(match.flip(-1).cumsum(-1).flip(-1) - match.long()).float())

def copy_matches(x):
    b, t = x.shape; pos = torch.arange(t, device=x.device); dp = torch.zeros_like(x)
    lengths, predicted = torch.zeros_like(x), torch.zeros_like(x)
    for s in range(t):
        dp = (x[:, s:s+1] == x) * (F.pad(dp[:, :-1], (1, 0)) + 1) * (pos < s)
        length = dp.amax(-1); j = torch.where((dp == length[:, None]) & (dp > 0), pos, -1).amax(-1)
        lengths[:, s] = length; predicted[:, s] = torch.where(length > 0, x.gather(1, (j + 1)[:, None]).squeeze(1), 0)
    return lengths, predicted

def stamp(m):
    return I.stamp(m), m.hit.tolist(), m.seen.tolist()

class Model(R.Model):
    def __init__(self, V, lr=1e-4, variant='all'):
        super().__init__(V, lr); self.variant = variant
        self.first = [Neuron(r, 16, lr) for r in (1, V+1, 4096, 16, 17)]; self.second = Neuron(V+1, 5, lr)
        for n in self.first + [self.second]: n.w = n.w.cuda()
        self.hit = torch.zeros(17, device='cuda'); self.seen = torch.zeros_like(self.hit)
        self.bounds = torch.tensor([1, 2, 3, 5, 8, 16, 32], device='cuda')

    def prepare(self, x):
        cod = self.base.codes(x); b, t = x.shape; assert t <= 256
        prior = torch.arange(t, device=x.device)[None, :] < torch.arange(t, device=x.device)[:, None]
        nxt = torch.cat([x[:, 1:], torch.full_like(x[:, :1], self.V)], 1); fast, recent = [], []; selector = torch.zeros_like(x)
        for i, order in enumerate(self.base.lens):
            c = torch.zeros((b, t, self.V), dtype=torch.uint8, device=x.device); r = torch.zeros_like(c, dtype=torch.float32)
            for a in range(0, b, 32):
                h = cod[order][a:a+32]; match = (h[:, :, None] == h[:, None, :]) & prior
                chars = F.one_hot(nxt[a:a+32], self.V+1)[..., :self.V].float()
                c[a:a+32] = torch.bmm(match.float(), chars).to(torch.uint8)
                if self.variant != 'no_recency': r[a:a+32] = torch.bmm(recency_weights(match), chars)
            total = c.sum(-1); selector = torch.where(total > 0, i*3 + (c > 0).sum(-1).clamp(1, 3), selector)
            fast.append(c); recent.append(r)
        return cod, fast + recent + list(copy_matches(x)), self.selectors(x) + [selector]

    def output(self, cod, selectors, memory, y=None):
        ps = self.base.predict(cod); p = ps[-1]
        for group in (memory[:5], memory[5:10]):
            q = p
            for c in group:
                c = c.float(); total = c.sum(-1, keepdim=True); q = torch.where(total > 0, (c + .5*q)/(total + .5), q); ps.append(q)
        length, k = memory[-2:]; active = (length > 0) & (self.variant != 'no_copy')
        row = torch.where(active, torch.bucketize(length.contiguous(), self.bounds)*2 + (k == p.argmax(-1)).long() + 1, 0)
        conf = (self.hit[row] + .5)/(self.seen[row] + 1)
        q = (1-conf[:, None])*p/(1-p.gather(1, k[:, None])).clamp_min(1e-12)
        q.scatter_(1, k[:, None], conf[:, None]); q = q/q.sum(-1, keepdim=True)
        ps.append(torch.where(active[:, None], q, p)); out = self.mix(ps, selectors + [row], y)
        if y is not None and self.variant != 'no_copy':
            self.seen.index_add_(0, row[active], torch.ones_like(conf[active]))
            self.hit.index_add_(0, row[active], (k[active] == y[active]).float())
        return out

    def logprobs(self, x, hits=None, copies=None, y=None):
        cod, memory, sel = self.prepare(x)
        if hits is not None: hits.add_(torch.stack([(c.sum(-1) > 0).sum() for c in memory[:5]]))
        if copies is not None: copies.add_(torch.stack([(memory[-2] > 0).sum(), ((memory[-2] > 0) & (memory[-1] == y)).sum()]))
        cod = {n: c.reshape(-1) for n, c in cod.items()}; memory = [c.reshape(-1, self.V) for c in memory[:10]] + [c.reshape(-1) for c in memory[-2:]]
        sel = [s.reshape(-1) for s in sel]
        return torch.cat([self.output({n: c[a:a+1024] for n, c in cod.items()}, [s[a:a+1024] for s in sel],
                                     [c[a:a+1024] for c in memory]) for a in range(0, x.numel(), 1024)]).reshape(*x.shape, self.V)

def reference(m):
    x = torch.tensor([[0, 1, 0, 1, 2, 0, 1, 0, 1, 2] + [0, 1]*20], device='cuda'); a = x[0].tolist(); lengths, predicted = copy_matches(x)
    for t in range(len(a)):
        candidates = []
        for j in range(t):
            n = 0
            while j-n >= 0 and a[t-n] == a[j-n]: n += 1
            candidates.append((n, j))
        n, j = max(candidates, default=(0, -1))
        assert int(lengths[0, t]) == n and (n == 0 or int(predicted[0, t]) == a[j+1]), 'copy reference failed'
    for codes in m.base.codes(x).values():
        h = codes[0].tolist(); match = (codes[:, :, None] == codes[:, None, :]) & (torch.arange(len(a), device=x.device)[:, None] > torch.arange(len(a), device=x.device))
        expected = [[.5**sum(h[k] == h[t] for k in range(j+1, t)) if j < t and h[j] == h[t] else 0. for j in range(len(a))] for t in range(len(a))]
        assert torch.equal(recency_weights(match), torch.tensor([expected], device=x.device)), 'recency reference failed'

def causality(m, te):
    x = I.B.eval_windows(te)[0][:4].cuda(); before = stamp(m); p = m.logprobs(x); changed = x.clone()
    g = torch.Generator(device=x.device).manual_seed(100)
    changed[:, 100:] = (x[:, 100:] + torch.randint(1, m.V, x[:, 100:].shape, device=x.device, generator=g)) % m.V
    ca, ma, sa = m.prepare(x); cb, mb, sb = m.prepare(changed)
    assert all(torch.equal(a[:, :100], b[:, :100]) for a, b in zip(list(ca.values()) + ma + sa, list(cb.values()) + mb + sb)), 'memory input leaked'
    assert torch.equal(p[:, :100], m.logprobs(changed)[:, :100]), 'future input leaked'
    assert stamp(m) == before, 'causality evaluation wrote state'

def run(name, budget=None):
    if budget is not None and budget < 128: raise ValueError('budget must be >= 128')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.cuda.set_per_process_memory_fraction(2.7 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(0); torch.cuda.reset_peak_memory_stats(); tr, te, V = I.B.load(); cut = int(len(tr)*.95)
    m = Model(V, variant=name); reference(m); seconds, chars = R.fit(m, tr[:cut], budget)
    before = stamp(m); renamed, info = I.rename(te); hits = torch.zeros(5, dtype=torch.long, device='cuda'); copies = torch.zeros(2, dtype=torch.long, device='cuda')
    targets = I.B.eval_windows(te)[1]; offset = 0
    def score(x):
        nonlocal offset
        y = targets[offset:offset+len(x)].cuda(); offset += len(x); return m.logprobs(x, hits, copies, y)
    test = I.evaluate(score, te); renamed_loss = I.evaluate(m.logprobs, renamed); val = I.evaluate(m.logprobs, tr[cut:]); causality(m, te)
    assert stamp(m) == before, 'evaluation wrote slow/mixer/copy state'
    settings = dict(seed=0, device='cuda', bs=976, T=128, lr=1e-4, budget=budget, sampling='one_pass', train_end=cut, val_start=cut,
                    requested_chars=budget if budget is not None else cut, lens=list(m.base.lens), alpha=.5, allocator_limit_gib=2.7, inputs=16,
                    first_rows=[1, V+1, 4096, 16, 17], second_inputs=5, fast_counts='iter8 uint8, sub-batches of 32, j < t', recency='float32, 0.5 ** rank, sub-batches of 32',
                    copy_buckets=['1', '2', '3', '4-5', '6-8', '9-16', '17-32', '33+'], copy_selector='0 if disabled/no match, else (bucket-1)*2+agree+1',
                    fourth_selector='iter8: 0 if no hit, else (longest-1)*3+clamp(distinct,1,3)', copy_confidence='(hit+0.5)/(seen+1), 17 zero-initialized rows, pre-write updates',
                    copy_ties='most recent j', variant=name, probability_floor=1e-6, batch_delta='sum', evaluation_frozen=True, eval_windows=I.B.N_EVAL_WINDOWS, renaming=info, decision_rules=RULES)
    result = dict(name=name, test_loss=test, renamed_loss=renamed_loss, val_loss=val, renamed_gap=renamed_loss-test, train_seconds=seconds, chars=chars,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, hit_rate=dict(zip(map(str, m.base.lens), (hits.double()/targets.numel()).tolist())),
                  copy_rate=copies[0].item()/targets.numel(), copy_accuracy=copies[1].item()/max(1, copies[0].item()), settings=settings)
    with (I.ROOT / f'results/iter9_{name}.json').open('w', encoding='utf-8') as f: json.dump(result, f, indent=2)
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('name', choices=('all', 'no_recency', 'no_copy')); p.add_argument('--budget', type=int)
    a = p.parse_args(); print(json.dumps(run(a.name, a.budget), indent=2), flush=True)
