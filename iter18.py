"""Window-local suffix copy gates on unchanged iter15 local; matched-sampler backprop control."""
import argparse, json, math, tempfile, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import iter15 as Q
import iter17 as R
import baseline
I = Q.I; B = I.B
LENS = (1, 2, 4, 8); FLOOR = 1e-6


def heads(x, V):
    b, t = x.shape; pos = torch.arange(t, device=x.device)
    equal = x[:, :, None] == x[:, None, :]; suffix = equal.clone()
    causal = pos[:, None] > pos[None, :]; ps = []; ns = []
    proposals = x[:, None, 1:].expand(b, t, t-1)
    for d in range(max(LENS)):
        if d and d < t: suffix[:, d:, d:] &= equal[:, :-d, :-d]
        if d+1 not in LENS: continue
        match = suffix & causal & (pos[:, None] >= d) & (pos[None, :] >= d)
        counts = torch.zeros(b, t, V, device=x.device).scatter_add_(-1, proposals, match[:, :, :-1].float())
        n = counts.sum(-1); ps.append(counts/n.clamp_min(1)[..., None]); ns.append(n)
    return torch.stack(ps, -2), torch.stack(ns, -1)


def features(p, h, n):
    return torch.cat([torch.ones_like(n[..., :1]), (n > 0).to(p.dtype), torch.log2(1+n)/7,
                      h.amax(-1), p.amax(-1, keepdim=True), -(p*p.clamp_min(FLOOR).log()).sum(-1, keepdim=True)/math.log(p.shape[-1])], -1).detach()


def mixture(A, phi, p, mask):
    pi = (phi @ A.T).masked_fill(~mask, -torch.inf).softmax(-1)
    return (pi[..., None]*p).sum(-2), pi


def gate_gradient(A, phi, p, mask, y):
    py = p.gather(-1, y.clamp_min(0)[..., None, None].expand(*y.shape, p.shape[-2], 1)).squeeze(-1)
    pi = (phi @ A.T).masked_fill(~mask, -torch.inf).softmax(-1)
    qy = (pi*py).sum(-1).clamp_min(FLOOR); valid = y >= 0
    # Negative of d log q(y) / d logit, averaged over real target positions.
    e = pi*(1-py/qy[..., None])*valid[..., None]/valid.sum().clamp_min(1)
    return e.flatten(0, 1).T @ phi.flatten(0, 1)


class Gate:
    def __init__(self, V):
        self.V, self.step = V, 0
        self.A = torch.zeros(5, 15, device='cuda'); self.m = torch.zeros_like(self.A); self.v = torch.zeros_like(self.A)
    def prepare(self, x, p):
        p = p.detach(); h, n = heads(x, self.V); phi = features(p, h, n)
        return phi, torch.cat([p[..., None, :], h], -2), torch.cat([torch.ones_like(n[..., :1], dtype=torch.bool), n > 0], -1)
    def predict(self, x, p):
        phi, ps, mask = self.prepare(x, p); q, pi = mixture(self.A, phi, ps, mask)
        return q.clamp_min(FLOOR).log(), pi
    def train(self, x, p, y):
        phi, ps, mask = self.prepare(x, p); grad = gate_gradient(self.A, phi, ps, mask, y)
        self.step += 1; self.m.mul_(.9).add_(grad, alpha=.1); self.v.mul_(.999).addcmul_(grad, grad, value=.001)
        self.A.addcdiv_(self.m/(1-.9**self.step), (self.v/(1-.999**self.step)).sqrt().add_(1e-8), value=-1e-2)


def checks(net, gate):
    x = torch.tensor([[4,0,1,0,1,0,1,0,1,0,1,0,2,0,1,0,1,0],
                      [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]], device='cuda')
    h, n = heads(x, 4); want = torch.zeros_like(h); counts = torch.zeros_like(n); rows = x.tolist()
    for b, row in enumerate(rows):
        for t in range(len(row)):
            for k, L in enumerate(LENS):
                if t < L-1: continue
                for j in range(L-1, t):
                    if row[j-L+1:j+1] == row[t-L+1:t+1]: want[b,t,k,row[j+1]] += 1; counts[b,t,k] += 1
    torch.testing.assert_close(n, counts, rtol=0, atol=0)
    torch.testing.assert_close(h, want/counts.clamp_min(1)[..., None], rtol=0, atol=0)
    p = torch.arange(1, 1+x.numel()*4, dtype=torch.float64, device='cuda').reshape(*x.shape, 4).sin().softmax(-1)
    h = h.double(); n = n.double(); phi = features(p, h, n); ps = torch.cat([p[..., None, :], h], -2)
    mask = torch.cat([torch.ones_like(n[..., :1], dtype=torch.bool), n > 0], -1)
    A = torch.arange(75, dtype=torch.float64, device='cuda').reshape(5, 15).cos()*.1
    y = (x+1)%4; y[0, -1] = -1; grad = gate_gradient(A, phi, ps, mask, y)
    def loss():
        q = mixture(A, phi, ps, mask)[0].gather(-1, y.clamp_min(0)[..., None]).squeeze(-1)
        return -q[y >= 0].clamp_min(FLOOR).log().mean()
    for k in range(5):
        for f in range(15):
            old = A[k,f].item(); eps = 1e-5; A[k,f] = old+eps; plus = loss(); A[k,f] = old-eps; minus = loss(); A[k,f] = old
            torch.testing.assert_close(grad[k,f], (plus-minus)/(2*eps), atol=1e-8, rtol=1e-5)
    x = (torch.arange(256, device='cuda').reshape(2,128)//3)%net.V; x[0,0] = net.V
    def predict(z):
        zs = net.forward(z)[1]; return gate.predict(z, sum(w*v for w,v in zip(net.p['g'], zs)).softmax(-1))[0]
    before = I.stamp((net, gate)); ref = predict(x)
    for t in (0, 7, 46, 126):
        other = x.clone(); other[:,t+1:] = (other[:,t+1:]+17)%net.V
        torch.testing.assert_close(ref[:,:t+1], predict(other)[:,:t+1], atol=1e-6, rtol=1e-6)
    assert I.stamp((net, gate)) == before, 'checks wrote model state'
    return dict(matches=True, gate_finite_difference=True, causality=True)


def backprop_epoch(tr, starts, bs=64):
    data = torch.as_tensor(tr.astype(np.int64), device='cuda'); pos = torch.arange(128, device='cuda')
    for a in range(0, len(starts), bs):
        ix = starts[a:a+bs].cuda()[:, None]+pos
        yield data[(ix-1).clamp_max(len(tr)-1)], torch.where(ix < len(tr), data[ix.clamp_max(len(tr)-1)], -1)


def run(name, smoke=None, nocopy=False):
    if smoke is not None and smoke < 1: raise ValueError('--smoke must be positive')
    if nocopy and name != 'copy': raise ValueError('--no-copy applies to copy mode')
    temporary = tempfile.TemporaryDirectory(prefix='iter18_') if smoke is not None else None
    path = Path(temporary.name)/f'iter18_{name}.json' if temporary else I.ROOT/f'results/iter18_{name}.json'
    if path.exists(): raise FileExistsError(path)
    assert torch.cuda.is_available() and not torch.is_grad_enabled()
    torch.manual_seed(0); np.random.seed(0); torch.cuda.reset_peak_memory_stats()
    tr, te, V = B.load(); cut = int(len(tr)*.95); train = tr[:cut]
    # A small prefix also exercises the short final window during smoke runs.
    if smoke is not None: train = train[:smoke]
    if name == 'backprop_same' and len(train) <= 128: raise ValueError('backprop_same needs >128 chars after dropping start zero')
    renamed, info = I.rename(te); old_windows = B.N_EVAL_WINDOWS
    settings = dict(seed=0, device='cuda', mode=name, bs=16 if name == 'copy' else 64, T=128, passes=1,
                    train_end=cut, val_start=cut, training_partition_chars=len(train), smoke=smoke, eval_windows=16 if smoke is not None else old_windows,
                    sampling='non-overlapping target windows, shuffled with CPU generator seed 0', renaming=info, evaluation_frozen=True,
                    novel_rule='Q.novel: aligned changed targets; maximal ASCII runs; earlier complete run required for repeat')
    chars = 0; result = {}
    if name == 'copy':
        net = Q.Model(V, False); gate = Gate(V); result['checks'] = checks(net, gate)
        def predict(x, copy=True):
            zs = net.forward(x)[1]; logits = sum(w*z for w,z in zip(net.p['g'], zs))
            return gate.predict(x, logits.softmax(-1))[0] if copy else logits.log_softmax(-1)
        forward = net.forward
        def training_forward(x):
            out = forward(x)
            # Same forward as net.train; gate finishes before the unchanged network update.
            gate.train(x, sum(w*z for w,z in zip(net.p['g'], out[1])).softmax(-1), y)
            return out
        net.forward = training_forward
        torch.cuda.synchronize(); start = time.perf_counter()
        try:
            for x, y in R.epoch(train, 0, len(train), V):
                net.train(x, y, 0); chars += int((y >= 0).sum())
        finally: del net.forward
        torch.cuda.synchronize(); seconds = time.perf_counter()-start; steps = net.step
        settings.update(D=256, layers=3, embedding_width=32, lags=list(range(6)), decays=[.5,.8,.95], lr=1e-3, lr_schedule='constant',
                        optimizer='manual Adam', betas=[.9,.999], eps=1e-8, gradient_clip_norm=1, weight_decay=0,
                        local_rule='unchanged iter15.Model(V, False).train(x,y,0)', g_init=1/3, g_constraint='none',
                        suffix_lengths=LENS, memory='window-local', gate_shape=[5,15], gate_init='zero', gate_lr=1e-2,
                        gate_optimizer='manual Adam', gate_betas=[.9,.999], gate_eps=1e-8, gate_clip=None, gate_weight_decay=0,
                        gate_features=['bias','matched x4','log2(1+count)/7 x4','max proposal probability x4','net max probability','net entropy/log V'],
                        probability_floor=FLOOR, floor_scope='mixture log and target denominator; entropy log', gate_prediction='same forward, before network update',
                        copy_enabled=not nocopy, novel_copy_enabled=True, net_reference_test_loss=1.4922)
        result['g'] = net.p['g'].tolist()
    else:
        cfg = baseline.CFG; assert cfg['block'] == 128 and cfg['batch'] == 64
        starts = torch.randperm(math.ceil(len(train)/128), generator=torch.Generator().manual_seed(0))*128
        starts = starts[starts != 0]; total = math.ceil(len(starts)/64)
        with torch.enable_grad():
            net = baseline.GPT(V).cuda()
            decay = [p for p in net.parameters() if p.dim() >= 2]; nod = [p for p in net.parameters() if p.dim() < 2]
            opt = torch.optim.AdamW([dict(params=decay, weight_decay=cfg['wd']), dict(params=nod, weight_decay=0)], lr=cfg['lr'], betas=(.9,.95))
            scaler = torch.amp.GradScaler(); torch.cuda.synchronize(); start = time.perf_counter(); steps = 0
            for step, (x, y) in enumerate(backprop_epoch(train, starts), 1):
                lr = cfg['lr']*step/cfg['warmup'] if step < cfg['warmup'] else cfg['min_lr']+.5*(cfg['lr']-cfg['min_lr'])*(1+math.cos(math.pi*(step-cfg['warmup'])/max(1, total-cfg['warmup'])))
                for gr in opt.param_groups: gr['lr'] = lr
                with torch.autocast('cuda', dtype=torch.float16): loss = F.cross_entropy(net(x).float().view(-1,V), y.view(-1), ignore_index=-1)
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); scaler.step(opt); scaler.update()
                chars += int((y >= 0).sum()); steps = step
            torch.cuda.synchronize(); seconds = time.perf_counter()-start
        assert steps == total and chars == len(train)-128
        net.eval()
        def predict(x):
            with torch.autocast('cuda', dtype=torch.float16): return net(x).float()
        settings.update(baseline_cfg=cfg, optimizer='AdamW', betas=[.9,.95], eps=1e-8, gradient_clip_norm=1,
                        amp='float16 + GradScaler', lr_schedule='baseline warmup then cosine; true batch count', schedule_steps=total,
                        dropped_start_zero=True, dropped_chars=128, ignore_index=-1)
        result['params'] = sum(p.numel() for p in net.parameters())
    try:
        if smoke is not None: B.N_EVAL_WINDOWS = 16
        state = (net, gate) if name == 'copy' else (dict(net.named_parameters()), dict(net.named_buffers()))
        before = I.stamp(state)
        fn = (lambda x: predict(x, not nocopy)) if name == 'copy' else predict
        test = I.evaluate(fn, te); ren = I.evaluate(fn, renamed); val = I.evaluate(fn, tr[cut:])
        result.update(Q.novel(predict, te, renamed, json.loads((I.ROOT/'data/meta.json').read_text())['vocab']))
        if name == 'copy':
            result['test_loss_nocopy'] = I.evaluate(lambda x: predict(x, False), te)
            result['renamed_loss_nocopy'] = I.evaluate(lambda x: predict(x, False), renamed)
            result['test_loss_nocopy_difference'] = result['test_loss_nocopy']-1.4922
            weights = torch.zeros(5, device='cuda'); x, _ = B.eval_windows(te)
            for xb in x.split(16):
                xb = xb.cuda(); zs = net.forward(xb)[1]; p = sum(w*z for w,z in zip(net.p['g'], zs)).softmax(-1)
                weights += gate.predict(xb, p)[1].sum((0,1))
            result['mean_gate_weight_test'] = dict(zip(['net', *[f'L{L}' for L in LENS]], (weights/x.numel()).tolist()))
        assert I.stamp(state) == before, 'evaluation wrote model state'
    finally: B.N_EVAL_WINDOWS = old_windows
    result.update(test_loss=test, renamed_loss=ren, renamed_gap=ren-test, val_loss=val, train_seconds=seconds, chars=chars,
                  steps=steps, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, settings=settings)
    result['pass'] = dict(test=test <= 1.30, renamed=ren <= 1.64, train=seconds <= 180, peak=result['peak_allocated_mib'] <= 321,
                          repeat_name=result['novel_repeat_loss'] is not None and result['novel_repeat_loss'] <= 4.0)
    with path.open('x', encoding='utf-8') as f: json.dump(result, f, indent=2, allow_nan=False)
    if temporary: temporary.cleanup()
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('name', choices=('copy','backprop_same'))
    p.add_argument('--smoke', type=int, metavar='N', help='train on at most N prefix chars, evaluate 16 windows, use a cleaned temporary result outside results/')
    p.add_argument('--no-copy', action='store_true', help='pure-net primary evaluation in copy mode; novel and gate diagnostics still measure copy')
    a = p.parse_args(); print(json.dumps(run(a.name, a.smoke, a.no_copy), indent=2), flush=True)
