"""Two-speed memory: frozen global tables plus causal, window-local counts."""
import argparse, json, time
import torch
import torch.nn.functional as F
import iter5 as I
from hybrid_mixer import Neuron
torch.set_grad_enabled(False)
RULES = ['Best: iter7 boost_flat test=1.2401; mix_full test=1.2528, renamed=1.7547, gap=0.502.',
         'Backprop at 16M chars: test=1.1238, gap=0.406.',
         'Drop fast unless test <= 1.195.', 'Name dependence fixed only if renamed_gap < 0.45.',
         'Memory helps only if fast beats fast_off by >= 0.03 test nats.']

class Model(I.Model):
    def __init__(self, V, lr=1e-4, off=False):
        super().__init__(V, lr)
        self.off = off
        assert len(self.base.lens) == 5 and tuple(sorted(self.base.lens)) == self.base.lens
        self.first = [Neuron(r, 10, lr) for r in (1, V + 1, 4096, 16)]
        self.second = Neuron(V + 1, 4, lr)
        for n in self.first + [self.second]: n.w = n.w.cuda()

    def prepare(self, x):
        cod = self.base.codes(x); b, t = x.shape
        assert t <= 256  # uint8 can hold every strictly earlier match.
        fast = []; selector = torch.zeros_like(x)
        prior = torch.arange(t, device=x.device)[None, :] < torch.arange(t, device=x.device)[:, None]
        nxt = torch.cat([x[:, 1:], torch.full_like(x[:, :1], self.V)], 1)
        for i, order in enumerate(self.base.lens):
            c = torch.zeros((b, t, self.V), dtype=torch.uint8, device=x.device)
            if not self.off:
                for a in range(0, b, 32):
                    h = cod[order][a:a+32]
                    match = (h[:, :, None] == h[:, None, :]) & prior
                    chars = F.one_hot(nxt[a:a+32], self.V + 1)[..., :self.V].float()
                    c[a:a+32] = torch.bmm(match.float(), chars).to(torch.uint8)
            total = c.sum(-1)
            distinct = (c > 0).sum(-1).clamp(1, 3)
            selector = torch.where(total > 0, i * 3 + distinct, selector)
            fast.append(c)
        return cod, fast, self.selectors(x) + [selector]

    def output(self, cod, selectors, fast, y=None):
        ps = self.base.predict(cod); p = ps[-1]
        for c in fast:
            c = c.float(); total = c.sum(-1, keepdim=True)
            p = torch.where(total > 0, (c + 0.5 * p) / (total + 0.5), p)
            ps.append(p)
        return self.mix(ps, selectors, y)

    def logprobs(self, x, hits=None):
        cod, fast, sel = self.prepare(x)
        if hits is not None: hits.add_(torch.stack([(c.sum(-1) > 0).sum() for c in fast]))
        cod = {n: c.reshape(-1) for n, c in cod.items()}
        fast = [c.reshape(-1, self.V) for c in fast]; sel = [s.reshape(-1) for s in sel]
        out = [self.output({n: c[a:a+1024] for n, c in cod.items()},
                          [s[a:a+1024] for s in sel], [c[a:a+1024] for c in fast])
               for a in range(0, x.numel(), 1024)]
        return torch.cat(out).reshape(*x.shape, self.V)

def fit(m, tr, budget):
    torch.cuda.synchronize(); start = time.perf_counter(); chars = 0
    for x, y in I.batches(tr, budget, 976, True, m.V):
        cod, fast, sel = m.prepare(x)
        for t in range(x.shape[1]):
            keep = y[:, t] >= 0; target = y[:, t][keep]
            if not len(target): continue
            c = {n: v[:, t][keep] for n, v in cod.items()}
            m.output(c, [s[:, t][keep] for s in sel], [v[:, t][keep] for v in fast], target)
            m.base.write(c, target)
        chars += int((y >= 0).sum())
    torch.cuda.synchronize()
    return time.perf_counter() - start, chars

def causality(m, te):
    x = I.B.eval_windows(te)[0][:4].cuda(); before = I.stamp(m)
    p = m.logprobs(x); changed = x.clone()
    g = torch.Generator(device=x.device).manual_seed(100)
    changed[:, 100:] = (x[:, 100:] + torch.randint(1, m.V, x[:, 100:].shape, device=x.device, generator=g)) % m.V
    assert torch.equal(p[:, :100], m.logprobs(changed)[:, :100]), 'future input leaked'
    assert I.stamp(m) == before, 'causality evaluation wrote state'

def run(name, budget=None):
    if budget is not None and budget < 128: raise ValueError('budget must be >= 128')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.cuda.set_per_process_memory_fraction(2.7 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(0); torch.cuda.reset_peak_memory_stats()
    tr, te, V = I.B.load(); cut = int(len(tr) * .95)
    m = Model(V, off=name == 'fast_off'); seconds, chars = fit(m, tr[:cut], budget)
    before = I.stamp(m); renamed, info = I.rename(te)
    hits = torch.zeros(len(m.base.lens), dtype=torch.long, device='cuda')
    test = I.evaluate(lambda x: m.logprobs(x, hits), te)
    renamed_loss = I.evaluate(m.logprobs, renamed); val = I.evaluate(m.logprobs, tr[cut:])
    causality(m, te)
    assert I.stamp(m) == before, 'evaluation wrote state'
    settings = dict(seed=0, device='cuda', bs=976, T=128, lr=1e-4, budget=budget,
                    sampling='one_pass', train_end=cut, val_start=cut, requested_chars=budget if budget is not None else cut,
                    lens=list(m.base.lens), alpha=.5, allocator_limit_gib=2.7, fast_counts='uint8, sub-batches of 32, j < t',
                    fast_off=m.off, inputs=10, first_rows=[1, V+1, 4096, 16], second_inputs=4,
                    fourth_selector='0 if no hit, else (longest-1)*3 + clamp(distinct,1,3)',
                    probability_floor=1e-6, batch_delta='sum', evaluation_frozen=True,
                    eval_windows=I.B.N_EVAL_WINDOWS, renaming=info, decision_rules=RULES)
    result = dict(name=name, test_loss=test, renamed_loss=renamed_loss, val_loss=val,
                  renamed_gap=renamed_loss-test, train_seconds=seconds, chars=chars,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  hit_rate=dict(zip(map(str, m.base.lens), (hits.double()/(I.B.N_EVAL_WINDOWS*128)).tolist())), settings=settings)
    with (I.ROOT / f'results/iter8_{name}.json').open('w', encoding='utf-8') as f: json.dump(result, f, indent=2)
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('name', choices=('fast', 'fast_off'))
    p.add_argument('--budget', type=int); a = p.parse_args()
    print(json.dumps(run(a.name, a.budget), indent=2), flush=True)
