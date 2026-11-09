"""Iteration 9 recency/copy inputs plus Iteration 10 surprise-clocked contexts."""
import argparse, json
import torch
import iter9 as N
import iter10 as S
from hybrid_mixer import Neuron
R, I, SIZE = N.R, N.I, S.SIZE
torch.set_grad_enabled(False)
RULES = ['Reference iter9 all: test=1.1843, gap=0.4373; iter10 clock: test=1.1711, gap=0.4559.',
         'Backprop at 16M chars: test=1.1238, gap=0.406.',
         'combo test <= 1.160 means the two sets of inputs add up; above that they overlap and input stacking stops.',
         'combo renamed_gap < 0.43 means name dependence improved.']

class Model(N.Model):
    def __init__(self, V, lr=1e-4):
        super().__init__(V, lr, variant='all')
        self.first = [Neuron(r, 22, lr) for r in (1, V+1, 4096, 16, 17, 12)]; self.second = Neuron(V+1, 6, lr)
        for n in self.first + [self.second]: n.w = n.w.cuda()
        self.C = [torch.zeros(SIZE, dtype=torch.int32, device='cuda') for _ in range(6)]
        self.T = [torch.zeros_like(c) for c in self.C]

    def prepare(self, x):
        cod, memory, sel = super().prepare(x); s = torch.full_like(x, float('inf'), dtype=torch.float32)
        prev = {n: c[:, :-1].reshape(-1) for n, c in cod.items()}; target = x[:, 1:].reshape(-1); parts = []
        for a in range(0, target.numel(), 1024):
            p = self.base.predict({n: c[a:a+1024] for n, c in prev.items()})[-1]
            parts.append(-p.gather(1, target[a:a+1024, None]).squeeze(1).log())
        if parts: s[:, 1:] = torch.cat(parts).reshape(x.shape[0], -1)
        fire = torch.stack([s > 1., s > 3.], 1); hashes, _, gap = S.clocked(x, fire, self.V)
        bucket = (gap > 0).long()+(gap > 1)+(gap > 2)+(gap > 4)+(gap > 8)
        return cod, memory, sel+hashes+[bucket*2]+list(fire.unbind(1))

    def mix(self, ps, selectors, y=None):
        p = low = ps[0]; hit = torch.zeros_like(selectors[10])
        for i, h in enumerate(selectors[4:10]):
            if i % 3 == 0: p = low
            ix, tag = I.slot(h[:, None]*1000003+self.base.chars+1); ix = ix & (SIZE-1)
            c = (self.C[i][ix]*(self.T[i][ix] == tag)).float(); total = c.sum(-1, keepdim=True)
            p = torch.where(total > 0, (c+.5*p)/(total+.5), p); ps.append(p)
            if i == 2: hit = (total[:, 0] > 0).long()
        out = super().mix(ps, selectors[:4]+[selectors[-1], selectors[10]+hit], y)
        if y is not None:
            one = torch.ones_like(y, dtype=torch.int32)
            for i, h in enumerate(selectors[4:10]):
                ix, tag = I.slot(h*1000003+y+1); ix = ix & (SIZE-1)
                self.C[i][ix[self.T[i][ix] != tag]] = 0; self.T[i][ix] = tag; ok = self.T[i][ix] == tag
                self.C[i].index_add_(0, ix[ok], one[ok])
        return out

def causality(m, te):
    N.causality(m, te)
    x = I.B.eval_windows(te)[0][:4].cuda(); changed = x.clone(); changed[:, 100:] = (changed[:, 100:]+1) % m.V
    a = m.prepare(x)[2][4:10]; b = m.prepare(changed)[2][4:10]
    assert all(torch.equal(u[:, :100], v[:, :100]) for u, v in zip(a, b)), 'clock hashes leaked future input'

def run(name, budget=None):
    if name != 'combo': raise ValueError('only combo is supported')
    if budget is not None and budget < 128: raise ValueError('budget must be >= 128')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.cuda.set_per_process_memory_fraction(2.7*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(0); torch.cuda.reset_peak_memory_stats(); tr, te, V = I.B.load(); cut = int(len(tr)*.95)
    m = Model(V); N.reference(m); S.reference(); seconds, chars = R.fit(m, tr[:cut], budget)
    before = I.stamp(m); renamed, info = I.rename(te); targets = I.B.eval_windows(te)[1]; offset = 0
    hits = torch.zeros(5, dtype=torch.long, device='cuda'); copies = torch.zeros(2, dtype=torch.long, device='cuda'); fires = torch.zeros_like(copies)
    def score(x):
        nonlocal offset
        y = targets[offset:offset+len(x)].cuda(); offset += len(x)
        fires.add_(torch.stack([c.sum() for c in m.prepare(x)[2][-2:]]))
        return m.logprobs(x, hits, copies, y)
    test = I.evaluate(score, te); renamed_loss = I.evaluate(m.logprobs, renamed); val = I.evaluate(m.logprobs, tr[cut:]); causality(m, te)
    assert I.stamp(m) == before, 'evaluation wrote slow/clocked/mixer/copy state'
    settings = dict(seed=0, device='cuda', bs=976, T=128, lr=1e-4, budget=budget, sampling='one_pass', train_end=cut, val_start=cut,
                    requested_chars=cut if budget is None else budget, lens=list(m.base.lens), alpha=.5, allocator_limit_gib=2.7,
                    inputs=22, input_order='iter9 all 16, A1 A2 A3 B1 B2 B3', first_rows=[1,V+1,4096,16,17,12], second_inputs=6,
                    variant=name, recency=True, copy=True, clock_entries=SIZE, clock_tables=6, clock_dtype='int32', thresholds=[1.,3.],
                    surprise='pre-loop frozen highest-order slow; preceding context; chunks=1024; s[0]=inf', clock_backoff='ps[0]; independent A and B chains',
                    sixth_selector='gap_bucket*2 + A3 hit', gap_buckets=['0','1','2','3-4','5-8','9+'],
                    fast_counts='iter8 uint8, sub-batches of 32, j < t', recency_weights='float32, 0.5 ** rank',
                    copy_confidence='(hit+0.5)/(seen+1), 17 zero-initialized rows, pre-write updates', copy_ties='most recent j',
                    copy_buckets=['1','2','3','4-5','6-8','9-16','17-32','33+'], copy_selector='0 if no match, else bucket*2+agree+1',
                    probability_floor=1e-6, batch_delta='sum', evaluation_frozen=True, eval_windows=I.B.N_EVAL_WINDOWS, renaming=info, decision_rules=RULES)
    denom = targets.numel()
    result = dict(name=name, test_loss=test, renamed_loss=renamed_loss, val_loss=val, renamed_gap=renamed_loss-test, train_seconds=seconds, chars=chars,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, hit_rate=dict(zip(map(str,m.base.lens),(hits.double()/denom).tolist())),
                  copy_rate=copies[0].item()/denom, copy_accuracy=copies[1].item()/max(1,copies[0].item()),
                  fire_rate_A=fires[0].item()/denom, fire_rate_B=fires[1].item()/denom, settings=settings)
    with (I.ROOT/'results/iter11_combo.json').open('w', encoding='utf-8') as f: json.dump(result, f, indent=2)
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('name', choices=('combo',)); p.add_argument('--budget', type=int)
    a = p.parse_args(); print(json.dumps(run(a.name, a.budget), indent=2), flush=True)
