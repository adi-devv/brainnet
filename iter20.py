"""Forward-pass local learning with fixed, window-local associative attention."""
import argparse, json, math, tempfile, time
from pathlib import Path
import torch
import torch.nn.functional as F
torch.set_grad_enabled(False)
import iter14 as N
import iter15 as Q
import iter17 as R
I = N.I; B = I.B
D, H, BLOCKS, LENS, INPUT_SCALE = 256, 128, 3, (2, 4, 8), 16
MODES = ('assoc', 'noretrieve', 'learned_input')


def retrieval(x):
    """Exact suffix equality; return most recent successor indices and masks [B,T,3]."""
    t = x.shape[1]; pos = torch.arange(t, device=x.device)
    equal = x[:, :, None] == x[:, None, :]; suffix = equal.clone(); indices = []; masks = []
    causal = pos[:, None] > pos[None, :]
    for d in range(max(LENS)):
        if d and d < t: suffix[:, d:, d:] &= equal[:, :-d, :-d]
        if d+1 not in LENS: continue
        match = suffix & causal & (pos[:, None] >= d) & (pos[None, :] >= d)
        ix = torch.where(match, pos+1, 0).amax(-1)
        indices.append(ix); masks.append(ix > 0)
    return torch.stack(indices, -1), torch.stack(masks, -1)


def block_gradients(x, h, z, y, r):
    e = N.error(z, y); d = (e @ r.T)*(1-h.square())
    return dict(R=N.readout_delta(h, e), c=e.sum((0,1)), b=d.sum((0,1)), W=N.readout_delta(x, d)), d


class Model:
    def __init__(self, V, mode='assoc'):
        if mode not in MODES: raise ValueError(mode)
        self.V, self.mode, self.lr, self.step = V, mode, 1e-3, 0
        self.input_scale = 1 if mode == 'learned_input' else INPUT_SCALE
        self.p, self.state = {}, {}
        def add(key, shape, scale=0., trainable=True):
            self.p[key] = torch.randn(shape, device='cuda')*scale
            if trainable: self.state[key] = [torch.zeros_like(self.p[key]), torch.zeros_like(self.p[key])]
        for j in range(6): add(f'A{j}', (V+1,D), 1/math.sqrt(6*D), mode == 'learned_input')
        add('E', (V+1,32), 1/math.sqrt(32), mode == 'learned_input')
        for k in range(3): add(f'P{k}', (32,D), 1/math.sqrt(3*32), mode == 'learned_input')
        self.U = []
        for l in range(BLOCKS):
            add(f'W{l}', (D,H), 1/math.sqrt(D)); add(f'b{l}', (H,))
            add(f'R{l}', (H,V), 1/math.sqrt(H)); add(f'c{l}', (V,))
            self.U.append((torch.randint(2, (H,D), device='cuda').float()*2-1)/math.sqrt(H))
        t = torch.arange(128, device='cuda'); distance = t[:,None]-t[None,:]
        self.M = torch.stack([(1-d)*d**distance.clamp_min(0)*(distance >= 0) for d in (.5,.8,.95)])

    def adam(self, grads):
        for key, grad in grads.items():
            assert torch.isfinite(grad).all(), key
            grad = grad/grad.norm().clamp_min(1); m, v = self.state[key]
            m.mul_(.9).add_(grad, alpha=.1); v.mul_(.999).addcmul_(grad, grad, value=.001)
            self.p[key].addcdiv_(m/(1-.9**self.step), (v/(1-.999**self.step)).sqrt().add_(1e-8), value=-self.lr)

    def forward(self, x, y=None):
        assert not torch.is_grad_enabled()
        p = self.p; T = x.shape[1]; M = self.M[:, :T, :T]
        ids = [torch.cat([torch.full((len(x),j), self.V, device=x.device), x], 1)[:, :T] for j in range(6)]
        em = M @ p['E'][x][:,None]
        s = (sum(p[f'A{j}'][ids[j]] for j in range(6)) + sum(em[:,k] @ p[f'P{k}'] for k in range(3)))*self.input_scale
        ix, matched = retrieval(x); divisor = 1+matched.sum(-1, keepdim=True)
        rows = torch.arange(len(x), device=x.device)[:,None]; z = 0; zs = []
        for l in range(BLOCKS):
            a = s if self.mode == 'noretrieve' else (s + sum(s[rows, ix[:,:,k]]*matched[:,:,k,None] for k in range(len(LENS))))/divisor
            h = (a @ p[f'W{l}'] + p[f'b{l}']).tanh()
            z = z + h @ p[f'R{l}'] + p[f'c{l}']; s = a + h @ self.U[l]; zs.append(z)
            if y is None: continue
            local, d = block_gradients(a, h, z, y, p[f'R{l}'])
            grads = {f'{key}{l}': value for key, value in local.items()}
            if l == 0 and self.mode == 'learned_input':
                # Three maps: input tables/projections -> W0 -> R0; retrieved routes are constants.
                d0 = (d @ p['W0'].T)/divisor
                for j in range(6): grads[f'A{j}'] = torch.zeros_like(p[f'A{j}']).index_add_(0, ids[j].flatten(), d0.flatten(0,1))
                de = torch.zeros_like(p['E'][x])
                for k in range(3):
                    grads[f'P{k}'] = N.readout_delta(em[:,k], d0); de += M[k].T @ (d0 @ p[f'P{k}'].T)
                grads['E'] = torch.zeros_like(p['E']).index_add_(0, x.flatten(), de.flatten(0,1))
            self.adam(grads)
        return zs

    def logits(self, x, block=None): return self.forward(x)[-1 if block is None else block]

    def train(self, x, y):
        self.step += 1; z = self.forward(x, y)[-1]
        return F.cross_entropy(z.flatten(0,1), y.flatten(), ignore_index=-1).item()


def checks(net):
    assert not torch.is_grad_enabled(); Q.plain(net)
    x = torch.tensor([[4,0,1,0,1,0,1,0,1,0,1,0,2,0,1,0,1,0],
                      [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]], device='cuda')
    for length in (1, 3, x.shape[1]):
        probe = x[:,:length]; ix, matched = retrieval(probe); want = torch.zeros_like(ix)
        for b, row in enumerate(probe.tolist()):
            for t in range(len(row)):
                for k, L in enumerate(LENS):
                    if t < L-1: continue
                    for j in range(L-1,t):
                        if row[j-L+1:j+1] == row[t-L+1:t+1]: want[b,t,k] = j+1
        torch.testing.assert_close(ix, want, atol=0, rtol=0)
        torch.testing.assert_close(matched, want > 0, atol=0, rtol=0)
    def values(shape, offset):
        return (torch.arange(math.prod(shape), device='cuda', dtype=torch.float64)+offset).sin().reshape(shape)*.3
    a = values((2,3,4), 1); w = values((4,3), 2); r = values((3,5), 3); b = values((3,), 4)
    c = values((5,), 5); prev = values((2,3,5), 6); y = torch.tensor([[0,2,-1],[4,1,3]], device='cuda')
    h = (a @ w+b).tanh(); grads, _ = block_gradients(a, h, prev+h @ r+c, y, r)
    def loss(): return F.cross_entropy((prev+(a @ w+b).tanh() @ r+c).flatten(0,1), y.flatten(), ignore_index=-1)
    for key, param in (('R',r), ('W',w), ('b',b)):
        for j in range(param.numel()):
            flat = param.view(-1); old = flat[j].item(); eps = 1e-5
            flat[j] = old+eps; plus = loss(); flat[j] = old-eps; minus = loss(); flat[j] = old
            torch.testing.assert_close(grads[key].flatten()[j], (plus-minus)/(2*eps), atol=1e-8, rtol=1e-5)
    x = (torch.arange(256, device='cuda').reshape(2,128)//3)%net.V; x[0,0] = net.V
    before = I.stamp(net); ref = net.forward(x)
    for t in (0,7,46,126):
        other = x.clone(); other[:,t+1:] = (other[:,t+1:]+17)%net.V
        for old, new in zip(ref, net.forward(other)): torch.testing.assert_close(old[:,:t+1], new[:,:t+1], atol=1e-6, rtol=1e-6)
    assert I.stamp(net) == before, 'evaluation wrote model state'
    return dict(retrieval=True, block_finite_difference=True, causality=True, evaluation_frozen=True)


def run(mode, seed=0, smoke=None):
    if mode not in MODES: raise ValueError(mode)
    if smoke is not None and smoke < 1: raise ValueError('--smoke must be positive')
    temporary = tempfile.TemporaryDirectory(prefix='iter20_') if smoke is not None else None
    filename = f'iter20_{mode}{"_seed"+str(seed) if seed != 0 else ""}.json'
    path = (Path(temporary.name) if temporary else I.ROOT/'results')/filename
    try:
        if path.exists(): raise FileExistsError(path)
        assert torch.cuda.is_available() and not torch.is_grad_enabled()
        torch.manual_seed(seed); torch.cuda.reset_peak_memory_stats()
        tr, te, V = B.load(); cut = int(len(tr)*.95); train = tr[:cut]
        if smoke is not None: train = train[:smoke]
        net = Model(V, mode); result = dict(checks=checks(net)); chars = 0
        frozen = (net.U, net.M, {k:v for k,v in net.p.items() if k not in net.state}); frozen_before = I.stamp(frozen)
        torch.cuda.synchronize(); start = time.perf_counter()
        for x, y in R.epoch(train, seed, len(train), V):
            net.train(x, y); chars += int((y >= 0).sum())
        torch.cuda.synchronize(); seconds = time.perf_counter()-start
        assert chars == len(train) and net.step == math.ceil(len(train)/(16*128))
        assert I.stamp(frozen) == frozen_before, 'training wrote fixed tensors'
        before = I.stamp(net); renamed, info = I.rename(te); old_windows = B.N_EVAL_WINDOWS
        try:
            if smoke is not None: B.N_EVAL_WINDOWS = 16
            test = I.evaluate(net.logits, te); ren = I.evaluate(net.logits, renamed); val = I.evaluate(net.logits, tr[cut:])
            per_block = [I.evaluate(lambda x,l=l: net.logits(x,l), te) for l in range(BLOCKS)]
            result.update(Q.novel(net.logits, te, renamed, json.loads((I.ROOT/'data/meta.json').read_text())['vocab']))
            x, _ = B.eval_windows(te); counts = torch.zeros(len(LENS), device='cuda', dtype=torch.long)
            for xb in x.split(16): counts += retrieval(xb.cuda())[1].sum((0,1))
            rates = dict(zip(map(str,LENS), (counts/x.numel()).tolist()))
            assert I.stamp(net) == before, 'evaluation wrote model state'
        finally: B.N_EVAL_WINDOWS = old_windows
        settings = dict(seed=seed, device='cuda', mode=mode, bs=16, T=128, D=D, H=H, blocks=BLOCKS, passes=1,
                        embedding_width=32, lags=list(range(6)), decays=[.5,.8,.95], pad_id=V, input_scale=net.input_scale,
                        input_scale_scope='completed s[0]; iter14 initialization stds unchanged', input_trainable=mode == 'learned_input',
                        input_rule='three maps: input tables/projections -> W0 -> R0; block 0 only; current-position route divided by 1+matches' if mode == 'learned_input' else 'fixed',
                        lr=net.lr, lr_schedule='constant', optimizer='manual Adam', betas=[.9,.999], eps=1e-8, gradient_clip_norm=1, weight_decay=0,
                        sampling='iter17.epoch(train, seed, len(train), V)', train_end=cut, val_start=cut, training_partition_chars=len(train),
                        smoke=smoke, eval_windows=16 if smoke is not None else old_windows, eval_chunk=16, renaming=info, evaluation_frozen=True,
                        suffix_lengths=LENS, retrieval='most recent exact suffix; successor j+1 <= t; computed once per batch in all modes',
                        retrieval_enabled=mode != 'noretrieve', memory='window-local', retrieval_match_rate_denominator='all test input positions',
                        U_fixed=True, U_init='independent uniform signs / sqrt(H)', layer_mixing=False, prediction='z2: cumulative block logits',
                        local_rule='own cumulative CE; previous logits, block input and retrieval constant; emit old-weight z and s before immediate block Adam',
                        novel_rule='iter15.novel: aligned changed targets; maximal ASCII runs; earlier complete run required for repeat',
                        pass_bars=dict(train_seconds=90, test_loss=1.329, renamed_loss=1.667), autograd=False)
        result.update(test_loss=test, renamed_loss=ren, renamed_gap=ren-test, val_loss=val, per_block_test_loss=per_block,
                      retrieval_match_rate=rates, train_seconds=seconds, chars=chars, steps=net.step,
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, settings=settings)
        result['pass'] = dict(train=seconds <= 90, test=test <= 1.329, renamed=ren <= 1.667)
        with path.open('x', encoding='utf-8') as f: json.dump(result, f, indent=2, allow_nan=False)
        return result
    finally:
        if temporary: temporary.cleanup()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=MODES); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--smoke', type=int, metavar='N', help='train at most N prefix chars; evaluate 16 windows; clean temporary JSON outside the repo')
    a = p.parse_args(); print(json.dumps(run(a.mode, a.seed, a.smoke), indent=2), flush=True)
