"""Surprise-clocked contexts alongside iter8's slow and causal fast memory."""
import argparse, json
import torch
import iter8 as R
from hybrid_mixer import Neuron
I = R.I
torch.set_grad_enabled(False)
SIZE = 1 << 23
RULES = ['Reference iter8 fast: test=1.1957, gap=0.4528; iter9 all: test=1.1843, gap=0.437.',
         'Backprop at 16M chars: test=1.1238, gap=0.406.', 'clock_off within 0.005 of 1.1957, else comparison suspect.',
         'Drop unless clock beats clock_off by >=0.03 test.', 'clock fire_rate_A <=0.70, else not sparse.',
         'clock beats clock_random by >=0.01 test, else surprise is not what helps.', 'clock renamed_gap <0.43 means less name dependence.']

def clocked(x, fire, V):
    b, t = x.shape; pos = torch.arange(1, t+1, device=x.device); rank = fire.long().cumsum(-1)
    events = torch.zeros((b, 2, t+1), dtype=x.dtype, device=x.device)
    events.scatter_reduce_(2, rank, torch.where(fire, pos, 0), reduce='amax', include_self=True)
    last = [events.gather(2, (rank-n).clamp_min(0)) for n in range(3)]
    xp = torch.cat([torch.full_like(x[:, :1], V), x], 1)
    regs = [xp[:, None, :].expand(-1, 2, -1).gather(2, j) for j in last]
    h = ((xp[:, :-1]+1)*1000003+x+1)[:, None, :].expand(-1, 2, -1); hashes = []
    for r in regs: h = h*1000003+r+1; hashes.append(h)
    return list(torch.stack(hashes, 2).reshape(b, 6, t).unbind(1)), torch.stack(regs, -1), pos-last[0][:, 0]

class Model(R.Model):
    def __init__(self, V, lr=1e-4, variant='clock', rates=None):
        super().__init__(V, lr); self.variant, self.rates = variant, rates
        self.first = [Neuron(r, 16, lr) for r in (1, V+1, 4096, 16, 12)]; self.second = Neuron(V+1, 5, lr)
        for n in self.first + [self.second]: n.w = n.w.cuda()
        self.C = [torch.zeros(SIZE, dtype=torch.int32, device='cuda') for _ in range(6)]
        self.T = [torch.zeros_like(c) for c in self.C]

    def prepare(self, x):
        cod, fast, sel = super().prepare(x); s = torch.full_like(x, float('inf'), dtype=torch.float32)
        if self.variant == 'clock':
            prev = {n: c[:, :-1].reshape(-1) for n, c in cod.items()}; target = x[:, 1:].reshape(-1); parts = []
            for a in range(0, target.numel(), 1024):
                p = self.base.predict({n: c[a:a+1024] for n, c in prev.items()})[-1]
                parts.append(-p.gather(1, target[a:a+1024, None]).squeeze(1).log())
            if parts: s[:, 1:] = torch.cat(parts).reshape(x.shape[0], -1)
            fire = torch.stack([s > 1., s > 3.], 1)
        elif self.variant == 'clock_random':
            fire = torch.stack([I.slot(cod[32]+salt)[0] % 100000 < rate*100000 for salt, rate in zip((0, 77), self.rates)], 1)
        else: fire = torch.zeros((len(x), 2, x.shape[1]), dtype=torch.bool, device=x.device)
        hashes, _, gap = clocked(x, fire, self.V)
        bucket = (gap > 0).long()+(gap > 1)+(gap > 2)+(gap > 4)+(gap > 8)
        return cod, fast+hashes+list(fire.unbind(1)), sel+[bucket*2 if self.variant != 'clock_off' else torch.zeros_like(x)]

    def output(self, cod, selectors, memory, y=None):
        ps = self.base.predict(cod); high = p = ps[-1]; low = ps[0]; hit = torch.zeros_like(selectors[-1])
        for c in memory[:5]:
            c = c.float(); total = c.sum(-1, keepdim=True); p = torch.where(total > 0, (c+.5*p)/(total+.5), p); ps.append(p)
        for i, h in enumerate(memory[5:11]):
            if self.variant == 'clock_off': ps.append(high); continue
            if i % 3 == 0: p = low
            ix, tag = I.slot(h[:, None]*1000003+self.base.chars+1); ix = ix & (SIZE-1)
            c = (self.C[i][ix]*(self.T[i][ix] == tag)).float(); total = c.sum(-1, keepdim=True)
            p = torch.where(total > 0, (c+.5*p)/(total+.5), p); ps.append(p)
            if i == 2: hit = (total[:, 0] > 0).long()
        out = self.mix(ps, selectors[:-1]+[selectors[-1]+hit], y)
        if y is not None and self.variant != 'clock_off':
            one = torch.ones_like(y, dtype=torch.int32)
            for i, h in enumerate(memory[5:11]):
                ix, tag = I.slot(h*1000003+y+1); ix = ix & (SIZE-1)
                self.C[i][ix[self.T[i][ix] != tag]] = 0; self.T[i][ix] = tag; ok = self.T[i][ix] == tag
                self.C[i].index_add_(0, ix[ok], one[ok])
        return out

    def logprobs(self, x, hits=None, fires=None):
        cod, memory, sel = self.prepare(x)
        if hits is not None: hits.add_(torch.stack([(c.sum(-1) > 0).sum() for c in memory[:5]]))
        if fires is not None: fires.add_(torch.stack([c.sum() for c in memory[-2:]]))
        cod = {n: c.reshape(-1) for n, c in cod.items()}; sel = [s.reshape(-1) for s in sel]
        memory = [c.reshape(-1, self.V) for c in memory[:5]]+[c.reshape(-1) for c in memory[5:]]
        return torch.cat([self.output({n: c[a:a+1024] for n, c in cod.items()}, [s[a:a+1024] for s in sel],
                                     [c[a:a+1024] for c in memory]) for a in range(0, x.numel(), 1024)]).reshape(*x.shape, self.V)

def reference():
    x = torch.tensor([[4, 0, 1, 2, 0, 3, 1, 2, 3, 0]]); s = torch.tensor([[float('inf'), 0., 1., 3., 3.1, 1.1, 0., 4., 1., 2.]])
    hashes, regs, gap = clocked(x, torch.stack([s > 1, s > 3], 1), 4); registers = [[4]*3 for _ in range(2)]; last = -1
    for t, c in enumerate(x[0].tolist()):
        for level, threshold in enumerate((1., 3.)):
            if s[0, t] > threshold:
                registers[level] = [c]+registers[level][:2]
                if level == 0: last = t
            h = ((int(x[0, t-1]) if t else 4)+1)*1000003+c+1
            assert regs[0, level, t].tolist() == registers[level], 'shift register reference failed'
            for n, r in enumerate(registers[level]):
                h = ((h*1000003+r+1+2**63) % 2**64)-2**63
                assert int(hashes[level*3+n][0, t]) == h, 'clock hash reference failed'
        assert int(gap[0, t]) == t-last, 'gap reference failed'

def causality(m, te):
    R.causality(m, te); x = I.B.eval_windows(te)[0][:4].cuda(); changed = x.clone(); changed[:, 100:] = (changed[:, 100:]+1) % m.V
    a = m.prepare(x)[1][5:11]; b = m.prepare(changed)[1][5:11]
    assert all(torch.equal(u[:, :100], v[:, :100]) for u, v in zip(a, b)), 'clock hashes leaked future input'

def run(name, budget=None):
    if budget is not None and budget < 128: raise ValueError('budget must be >= 128')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.cuda.set_per_process_memory_fraction(2.7*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(0); torch.cuda.reset_peak_memory_stats(); tr, te, V = I.B.load(); cut = int(len(tr)*.95); rates = None
    if name == 'clock_random':
        source = json.loads((I.ROOT/'results/iter10_clock.json').read_text(encoding='utf-8'))
        if source['chars'] != (cut if budget is None else min(cut, budget)): raise ValueError('clock chars differs from this run')
        rates = (source['fire_rate_A'], source['fire_rate_B'])
    reference(); m = Model(V, variant=name, rates=rates); seconds, chars = R.fit(m, tr[:cut], budget)
    before = I.stamp(m); renamed, info = I.rename(te); hits = torch.zeros(5, dtype=torch.long, device='cuda'); fires = torch.zeros(2, dtype=torch.long, device='cuda')
    test = I.evaluate(lambda x: m.logprobs(x, hits, fires), te); renamed_loss = I.evaluate(m.logprobs, renamed); val = I.evaluate(m.logprobs, tr[cut:]); causality(m, te)
    assert I.stamp(m) == before, 'evaluation wrote state'
    settings = dict(seed=0, device='cuda', bs=976, T=128, lr=1e-4, budget=budget, sampling='one_pass', train_end=cut, val_start=cut,
                    requested_chars=cut if budget is None else budget, lens=list(m.base.lens), alpha=.5, allocator_limit_gib=2.7, inputs=16,
                    first_rows=[1,V+1,4096,16,12], second_inputs=5, variant=name, thresholds=[1.,3.], random_rates=rates, random_key='order-32 hash + salt', clock_entries=SIZE, clock_dtype='int32',
                    surprise='pre-loop frozen highest-order slow; preceding context; chunks=1024; s[0]=inf', gap_buckets=['0','1','2','3-4','5-8','9+'],
                    fifth_selector='gap_bucket*2 + A3 hit; clock_off=0', fast_counts='iter8 uint8, sub-batches of 32, j<t',
                    probability_floor=1e-6, batch_delta='sum', evaluation_frozen=True, eval_windows=I.B.N_EVAL_WINDOWS, renaming=info, decision_rules=RULES)
    denom = I.B.N_EVAL_WINDOWS*128
    result = dict(name=name, test_loss=test, renamed_loss=renamed_loss, val_loss=val, renamed_gap=renamed_loss-test, train_seconds=seconds, chars=chars,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, hit_rate=dict(zip(map(str,m.base.lens),(hits.double()/denom).tolist())), fire_rate_A=fires[0].item()/denom, fire_rate_B=fires[1].item()/denom, settings=settings)
    with (I.ROOT/f'results/iter10_{name}.json').open('w', encoding='utf-8') as f: json.dump(result, f, indent=2)
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('name', choices=('clock','clock_off','clock_random')); p.add_argument('--budget', type=int)
    a = p.parse_args(); print(json.dumps(run(a.name, a.budget), indent=2), flush=True)
