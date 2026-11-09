"""Bohning closed-form logit boosting; no differentiation through stages."""
import argparse, json, sys, time
import torch
import torch.nn.functional as F
import iter5, iter6
torch.set_grad_enabled(False)
D = 2**13
RULES = ['Reference mix_full: test=1.2528, renamed=1.7547, gap=0.502.',
         'Backprop at 16M chars: test=1.1238, gap=0.406.',
         'Drop boost unless final test beats 1.2528 by >=0.05.',
         'Depth helps only if stage 3 improves test by >=0.02.',
         'Crossing helps only if boost beats boost_flat by >=0.02.',
         'A renamed gap below 0.502 means less dependence on memorised names.']

def features(x, s, V, k, name):
    f = iter6.features(x, V) & (D-1)
    if k > 1 and name == 'boost':
        top = s.topk(2, dim=-1).indices; a, b = top.unbind(-1)
        conf = (s.softmax(-1).amax(-1)*4).long().clamp_max(3)
        f = iter6.mix(f*(V+1) + a[..., None]+1) & (D-1)
        extra = torch.stack([a*(V+1)+b+1000003, a*4+conf+2000006,
                             a*(V+1)+x+3000009], -1)
        f = torch.cat([f, iter6.mix(extra) & (D-1)], -1)
    f = torch.cat([f, torch.zeros_like(x[..., None])], -1).sort(-1).values
    active = torch.cat([torch.ones_like(f[..., :1], dtype=torch.bool), f[..., 1:] != f[..., :-1]], -1)
    return f, active

def blocks(x, y, s):
    return [[a.clone(), b.clone(), c.clone()] for a, b, c in
            zip(x.split(2), y.split(2), s.split(2)) if bool((b >= 0).any())]

def fit_base(m, tr, budget):
    torch.cuda.synchronize(); start = time.perf_counter(); chars = 0; cache = []
    half = min(len(tr), budget if budget is not None else len(tr))//2
    for x, y in iter5.batches(tr, budget, 976, True, m.V):
        cod, sel = m.base.codes(x), m.selectors(x)
        sy = torch.full(y.shape, -1, dtype=torch.int16); ss = None
        for t in range(x.shape[1]):
            keep = y[:, t] >= 0; target = y[:, t][keep]; n = len(target)
            if not n: continue
            c = {k: v[:, t][keep] for k, v in cod.items()}
            lp = m.output(c, [s[:, t][keep] for s in sel], {}, target)
            if chars+n > half:
                if ss is None: ss = torch.zeros((*y.shape, m.V), dtype=torch.float32)
                skip = max(0, half-chars); ix = keep.nonzero().flatten()[skip:].cpu()
                sy[ix, t] = target[skip:].cpu().short(); ss[ix, t] = lp[skip:].cpu().float()
            m.base.write(c, target); chars += n
        if ss is not None: cache.extend(blocks(x.cpu().to(torch.uint8), sy, ss))
    torch.cuda.synchronize()
    assert sum(int((b[1] >= 0).sum()) for b in cache) == chars-half
    return cache, chars, time.perf_counter()-start

def rows(block, V, k, name):
    x, y, s = [t.cuda().long() if i < 2 else t.cuda() for i, t in enumerate(block)]
    keep = y >= 0; f, active = features(x, s, V, k, name)
    return f[keep], active[keep], y[keep], s[keep]

def delta(f, active, W):
    return (W[f]*active[..., None]).sum(1)

def apply(x, s, W, eta, V, k, name):
    f, active = features(x, s, V, k, name); shape = s.shape
    f, active, s = f.flatten(0, 1), active.flatten(0, 1), s.reshape(-1, V)
    return torch.cat([s[a:a+256]+eta*delta(f[a:a+256], active[a:a+256], W)
                      for a in range(0, len(s), 256)]).reshape(shape)

def normal_equations(cache, V, k, name):
    A = torch.zeros((D, D), device='cuda'); b = torch.zeros((D, V), device='cuda')
    for block in cache:
        f, active, y, s = rows(block, V, k, name)
        r = -s.softmax(-1); r[torch.arange(len(y), device='cuda'), y] += 1
        pairs = f[:, :, None]*D + f[:, None, :]
        values = .5*(active[:, :, None] & active[:, None, :]).float()
        A.view(-1).index_add_(0, pairs.flatten(), values.flatten())
        b.index_add_(0, f.flatten(), (r[:, None, :]*active[..., None]).reshape(-1, V))
    return A, b

def search(A, b, val, V, k, name, diagnostics):
    mean = A.diagonal().mean().item(); old = 0.; best = None
    for mult in (1e-3, 1e-2, 1e-1):
        lam = mult*mean; A.diagonal().add_(lam-old); old = lam
        L, info = torch.linalg.cholesky_ex(A); code = int(info.item())
        record = dict(stage=k, lam=lam, multiplier=mult, cholesky_info=code); diagnostics.append(record)
        if code:
            print(f'Stage {k}: Cholesky failed, lam={lam}, info={code}', file=sys.stderr, flush=True)
            del L
            continue
        W = torch.cholesky_solve(b, L); del L
        totals = [0., 0., 0.]; count = 0
        for block in val:
            f, active, y, s = rows(block, V, k, name); correction = delta(f, active, W)
            for j, eta in enumerate((1, 2, 4)):
                totals[j] += F.cross_entropy(s+eta*correction, y, reduction='sum').item()
            count += len(y)
        losses = [t/count for t in totals]; record['val_losses_eta_1_2_4'] = losses
        j = min(range(3), key=losses.__getitem__)
        if best is None or losses[j] < best[0]: best = (losses[j], lam, (1, 2, 4)[j], W)
    if best is None: raise RuntimeError(f'Stage {k}: all Cholesky factorisations failed')
    return best

def advance(cache, W, eta, V, k, name):
    for block in cache:
        f, active, _, s = rows(block, V, k, name)
        block[2][block[1] >= 0] = (s+eta*delta(f, active, W)).cpu()

def run(name, budget=None):
    if budget is not None and budget < 128: raise ValueError('budget must be >=128')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.cuda.set_per_process_memory_fraction(2.7*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(0); torch.cuda.reset_peak_memory_stats()
    tr, te, V = iter5.B.load(); cut = int(len(tr)*.95); val = tr[cut:]
    m = iter5.Model(V, 1e-4); cache, chars, seconds = fit_base(m, tr[:cut], budget)
    renamed, info = iter5.rename(te); weights = []; stages = []; diagnostics = []
    def predict(x):
        s = m.logprobs(x)
        for k, (W, eta) in enumerate(weights, 1): s = apply(x, s, W, eta, V, k, name)
        return s
    def measure(k, lam, eta, elapsed):
        before = iter5.stamp((m, weights))
        losses = {key: iter5.evaluate(predict, data) for key, data in
                  [('test_loss', te), ('renamed_loss', renamed), ('val_loss', val)]}
        assert iter5.stamp((m, weights)) == before, 'evaluation wrote state'
        stages.append(dict(stage=k, **losses, lam=lam, eta=eta, fit_seconds=elapsed))
    measure(0, None, None, seconds)
    print(f'Stage 0 test_loss={stages[0]["test_loss"]:.7f}; reference=1.2528 (full pass only)', file=sys.stderr, flush=True)
    if budget is None: assert abs(stages[0]['test_loss']-1.2528) <= .005, 'stage 0 reference mismatch'
    vx, vy = iter5.B.eval_windows(val); validation = []; before = iter5.stamp(m)
    for a in range(0, len(vx), 16):
        validation.extend(blocks(vx[a:a+16].to(torch.uint8), vy[a:a+16].short(), m.logprobs(vx[a:a+16].cuda()).cpu().float()))
    assert iter5.stamp(m) == before, 'evaluation wrote state'
    for k in range(1, 4):
        torch.cuda.synchronize(); start = time.perf_counter()
        A, b = normal_equations(cache, V, k, name)
        _, lam, eta, W = search(A, b, validation, V, k, name, diagnostics)
        del A, b
        torch.cuda.empty_cache()
        advance(cache, W, eta, V, k, name); advance(validation, W, eta, V, k, name)
        weights.append((W, eta)); torch.cuda.synchronize(); elapsed = time.perf_counter()-start
        seconds += elapsed; measure(k, lam, eta, elapsed)
    settings = dict(seed=0, device='cuda', bs=976, T=128, lr=1e-4, budget=budget,
                    sampling='one_pass', train_end=cut, val_start=cut, D=D, binary_features=True, bias=0,
                    residual_targets='predict-before-write; second half in pass order; CPU float32',
                    feature_context='iter6.features on original x windows; strictly before target',
                    confidence_bins=4, lam_multipliers=[1e-3, 1e-2, 1e-1], eta_grid=[1, 2, 4],
                    allocator_limit_gib=2.7, count_entries=iter5.SIZE, eval_windows=iter5.B.N_EVAL_WINDOWS,
                    stage0_reference_checked=budget is None, cholesky=diagnostics, renaming=info, decision_rules=RULES)
    result = dict(name=name, stages=stages, **{key: stages[-1][key] for key in ('test_loss', 'renamed_loss', 'val_loss')},
                  train_seconds=seconds, chars=chars, stage_chars=sum(int((b[1] >= 0).sum()) for b in cache),
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, settings=settings)
    (iter5.ROOT/f'results/iter7_{name}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('name', choices=('boost', 'boost_flat'))
    p.add_argument('--budget', type=int); args = p.parse_args()
    print(json.dumps(run(args.name, args.budget), indent=2), flush=True)
