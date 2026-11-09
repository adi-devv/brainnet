"""Table-free causal stack with local direct feedback alignment and manual Adam."""
import argparse, json, math, time
import torch
import iter5 as I
torch.set_grad_enabled(False)
RULES = ['Backprop at 1M chars: test 2.330; iter12 pure at 6.2M: test 2.53, renamed 2.59, gap 0.061.',
         'dfa test < 2.33: local rule learns representations at least as fast as backprop here; scale it.',
         'dfa test <= shallow test - 0.03: depth helps without backprop; keep stacking.',
         'Otherwise: the local rule cannot learn representations here; next is a different local rule or hybrid.']

def error(z, y):
    e = z.softmax(-1); e.scatter_add_(-1, y.clamp_min(0)[..., None], -torch.ones_like(e[..., :1]))
    return e * (y >= 0)[..., None] / (y >= 0).sum()

def readout_delta(h, e):
    return h.flatten(0, 1).T @ e.flatten(0, 1)

class Model:
    def __init__(self, V, L, lr):
        self.V, self.L, self.lr, self.step = V, L, lr, 0
        self.p, self.state = {}, {}
        def add(key, shape, scale=0.):
            self.p[key] = torch.randn(shape, device='cuda') * scale
            self.state[key] = [torch.zeros_like(self.p[key]), torch.zeros_like(self.p[key])]
        for j in range(6): add(f'A{j}', (V+1, 256), 1/math.sqrt(6*256))
        add('E', (V+1, 32), 1/math.sqrt(32))
        for k in range(3): add(f'P{k}', (32, 256), 1/math.sqrt(3*32))
        for l in range(L):
            if l: add(f'W{l}', (1024, 256), 1/math.sqrt(1024))
            add(f'b{l}', (256,)); add(f'R{l}', (256, V), 1/math.sqrt(256)); add(f'c{l}', (V,))
        add('g', (L,)); self.p['g'].fill_(1/L)
        self.feedback = [torch.randn(V, 256, device='cuda') / math.sqrt(V) for _ in range(L)]
        t = torch.arange(128, device='cuda'); distance = t[:, None]-t[None, :]
        self.M = torch.stack([(1-d)*d**distance.clamp_min(0)*(distance >= 0) for d in (.5, .8, .95)])
        assert not torch.is_grad_enabled() and all(type(p) is torch.Tensor and not p.requires_grad for p in self.p.values())

    def forward(self, x):
        p = self.p; T = x.shape[1]; M = self.M[:, :T, :T]
        ids = [torch.cat([torch.full((len(x), j), self.V, device=x.device), x], 1)[:, :T] for j in range(6)]
        em = M @ p['E'][x][:, None]
        a = sum(p[f'A{j}'][ids[j]] for j in range(6)) + sum(em[:, k] @ p[f'P{k}'] for k in range(3)) + p['b0']
        hs, zs, us = [], [], []
        for l in range(self.L):
            if l:
                u = torch.cat([hs[-1], *(M @ hs[-1][:, None]).unbind(1)], -1); us.append(u)
                a = u @ p[f'W{l}'] + p[f'b{l}']
            h = a.tanh(); hs.append(h); zs.append(h @ p[f'R{l}'] + p[f'c{l}'])
        return hs, zs, us, ids, em

    def logits(self, x, layer=None):
        zs = self.forward(x)[1]
        return zs[layer] if layer is not None else sum(w*z for w, z in zip(self.p['g'], zs)).log_softmax(-1)

    def train(self, x, y):
        hs, zs, us, ids, em = self.forward(x); p = self.p; grads = {}
        final = sum(w*z for w, z in zip(p['g'], zs)); ef = error(final, y)
        grads['g'] = torch.stack([(ef*z).sum() for z in zs])
        loss = torch.nn.functional.cross_entropy(final.flatten(0, 1), y.flatten(), ignore_index=-1).item()
        for l, (h, z) in enumerate(zip(hs, zs)):
            e = error(z, y); delta = (e @ self.feedback[l]) * (1-h.square())
            grads[f'R{l}'] = readout_delta(h, e); grads[f'c{l}'] = e.sum((0, 1)); grads[f'b{l}'] = delta.sum((0, 1))
            if l: grads[f'W{l}'] = readout_delta(us[l-1], delta)
            else:
                for j in range(6):
                    grads[f'A{j}'] = torch.zeros_like(p[f'A{j}']).index_add_(0, ids[j].flatten(), delta.flatten(0, 1))
                de = torch.zeros_like(p['E'][x])
                for k in range(3):
                    grads[f'P{k}'] = readout_delta(em[:, k], delta)
                    de += self.M[k].T @ (delta @ p[f'P{k}'].T)
                grads['E'] = torch.zeros_like(p['E']).index_add_(0, x.flatten(), de.flatten(0, 1))
        self.step += 1
        for key, grad in grads.items():
            assert torch.isfinite(grad).all(), key
            grad = grad / grad.norm().clamp_min(1); m, v = self.state[key]
            m.mul_(.9).add_(grad, alpha=.1); v.mul_(.999).addcmul_(grad, grad, value=.001)
            p[key].addcdiv_(m/(1-.9**self.step), (v/(1-.999**self.step)).sqrt().add_(1e-8), value=-self.lr)
        return loss

def checks(m):
    assert not torch.is_grad_enabled() and all(not t.requires_grad for t in [*m.p.values(), *m.feedback, m.M])
    for l in range(m.L):
        h = torch.randn(2, 3, 256, device='cuda', dtype=torch.float64); y = torch.randint(m.V, (2, 3), device='cuda'); y[0, 0] = -1
        r, c = m.p[f'R{l}'].double().clone(), m.p[f'c{l}'].double(); analytic = readout_delta(h, error(h @ r+c, y))
        def ce(): return torch.nn.functional.cross_entropy((h @ r+c).flatten(0, 1), y.flatten(), ignore_index=-1)
        for i, j in ((0, 0), (17, m.V//2), (255, m.V-1)):
            old = r[i, j].item(); eps = 1e-5; r[i, j] = old+eps; plus = ce(); r[i, j] = old-eps; minus = ce(); r[i, j] = old
            torch.testing.assert_close(analytic[i, j], (plus-minus)/(2*eps), atol=1e-8, rtol=1e-5)
    x = torch.randint(m.V+1, (2, 128), device='cuda'); other = x.clone(); other[:, 47:] = (other[:, 47:]+1) % (m.V+1)
    a, b = m.forward(x), m.forward(other)
    for old, new in zip(a[0]+a[1], b[0]+b[1]): torch.testing.assert_close(old[:, :47], new[:, :47], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(m.logits(x)[:, :47], m.logits(other)[:, :47], atol=1e-6, rtol=1e-6)

def run(name, budget=999_424, bs=16, lr=1e-3, smoke=False):
    if budget < 1 or bs < 1 or not math.isfinite(lr) or lr <= 0: raise ValueError('positive budget, bs and finite lr required')
    if smoke and budget > 256: raise ValueError('smoke requires --budget <=256')
    path = I.ROOT/f'results/iter14_{"smoke_" if smoke else ""}{name}.json'
    if path.exists(): raise FileExistsError(path)
    assert torch.cuda.is_available(); torch.manual_seed(0); torch.cuda.reset_peak_memory_stats()
    tr, te, V = I.B.load(); cut = int(len(tr)*.95); m = Model(V, 3 if name == 'dfa' else 1, lr); checks(m)
    torch.cuda.synchronize(); start = time.perf_counter(); chars = 0; losses = []
    for x, y in I.batches(tr[:cut], budget, bs, True, V):
        losses.append(m.train(x, y)); chars += int((y >= 0).sum())
    torch.cuda.synchronize(); seconds = time.perf_counter()-start; checks(m)
    n = max(1, math.ceil(len(losses)*.1))
    if not smoke: assert sum(losses[-n:])/n < sum(losses[:n])/n, 'training loss did not decrease'
    before = I.stamp(m); renamed, info = I.rename(te); old_windows = I.B.N_EVAL_WINDOWS
    try:
        if smoke: I.B.N_EVAL_WINDOWS = 2
        test = I.evaluate(m.logits, te); ren = I.evaluate(m.logits, renamed); val = I.evaluate(m.logits, tr[cut:])
        per_layer = [I.evaluate(lambda x, l=l: m.logits(x, l), te) for l in range(m.L)]
        assert I.stamp(m) == before, 'evaluation wrote parameters, Adam state or g'
    finally: I.B.N_EVAL_WINDOWS = old_windows
    settings = dict(seed=0, device='cuda', bs=bs, T=128, D=256, embedding_width=32, layers=m.L, decays=[.5,.8,.95], lags=list(range(6)), lr=lr, budget=budget,
                    optimizer='manual Adam', betas=[.9,.999], eps=1e-8, weight_decay=0, gradient_clip_norm=1, feedback_std=1/math.sqrt(V), feedback_fixed=True,
                    sampling='one_pass', train_end=cut, val_start=cut, variant=name, smoke=smoke, eval_windows=2 if smoke else old_windows, renaming=info,
                    local_rule='per-layer CE; fixed feedback; no cross-layer derivative', g_init=1/m.L, g_constraint='none', evaluation_frozen=True, decision_rules=RULES)
    result = dict(test_loss=test, renamed_loss=ren, renamed_gap=ren-test, val_loss=val, per_layer_test_loss=per_layer, g=m.p['g'].tolist(),
                  train_seconds=seconds, chars=chars, steps=m.step, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, settings=settings)
    with path.open('x', encoding='utf-8') as f: json.dump(result, f, indent=2, allow_nan=False)
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('name', choices=('dfa', 'shallow')); p.add_argument('--budget', type=int, default=999_424)
    p.add_argument('--bs', type=int, default=16); p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); print(json.dumps(run(a.name, a.budget, a.bs, a.lr, a.smoke), indent=2), flush=True)
