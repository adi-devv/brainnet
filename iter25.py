"""iter24 with detached cumulative-logit residual learning: each layer corrects the stored logits below it."""
import argparse, copy, json, math, tempfile, time
from itertools import accumulate
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import iter24 as S
Q, R, C, I, B = S.Q, S.R, S.C, S.I, S.B
torch.set_grad_enabled(False)
LOWER = [*[f'A{j}' for j in range(6)], 'E', *[f'P{k}' for k in range(3)], 'b0', 'R0', 'c0']
close = torch.testing.assert_close


class Model(S.Model):
    def __init__(self, V, total_steps):
        super().__init__(V, total_steps)  # Same RNG draws as iter24.
        for l in (1, 2): self.p[f'R{l}'].zero_(); self.p[f'c{l}'].zero_()
        self.p['g'].copy_(self.p['g'].new_tensor([0., 0., 1.]))  # Fixed selector of s_2; never updated.

    def forward(self, x):
        hs, rs, us, ids, em = super().forward(x)
        return hs, list(accumulate(rs)), us, ids, em  # Cumulative s_l fills the logits slot.

    def train(self, x, y, gate):
        self.lr = S.schedule(self.step+1, self.total_steps); self.step += 1
        M, ids, em = self._prepare(x); lower = None; s = 0
        for l in range(self.L):
            h, r, u = self._layer(l, lower, M, ids, em); s = s+r  # s holds pre-update values, so no derivative reaches lower layers.
            self._update(self._local_grads(l, x, y, h, s, u, ids, em, M)); lower = h
        gate.train(x, s.softmax(-1), y)
        return F.cross_entropy(s.flatten(0, 1), y.flatten(), ignore_index=-1).item()


def checks(net, gate, x, y, train):
    before = I.stamp((net, gate)); assert net.lr == S.TOP and net.step == gate.step == 0
    result = C.checks(net, gate); Q.plain((net, gate)); assert not torch.is_grad_enabled() and net.step == gate.step == 0
    cpu_rng = torch.get_rng_state(); cuda_rng = torch.cuda.get_rng_state()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(0); ref = S.Model(net.V, net.total_steps)
        close(torch.get_rng_state(), cpu_rng, atol=0, rtol=0); close(torch.cuda.get_rng_state(), cuda_rng, atol=0, rtol=0)
    zeroed = {'R1', 'c1', 'R2', 'c2'}
    for key, p in ref.p.items():
        if key in zeroed: assert not net.p[key].count_nonzero() and (key[0] == 'c' or p.count_nonzero())  # iter24's biases already start at zero.
        elif key == 'g': close(net.p[key], p.new_tensor([0., 0., 1.]), atol=0, rtol=0)
        else: close(net.p[key], p, atol=0, rtol=0)
        for s in net.state[key]: assert not s.count_nonzero()
    ss = net.forward(x)[1]; z0 = ref.forward(x)[1][0]; assert len(ss) == 3
    for s in ss: close(s, z0, atol=1e-6, rtol=0)  # Upper layers begin as exact no-ops.

    # Float64 finite differences of the cumulative loss, with frozen inputs and frozen random s_{l-1}.
    probe = copy.deepcopy(net); pg = copy.deepcopy(gate)
    probe.p = {k: p.double() for k, p in probe.p.items()}; probe.M = probe.M.double()
    probe.state = {k: [s.double() for s in ss] for k, ss in probe.state.items()}
    for key in ('A', 'm', 'v'): setattr(pg, key, getattr(pg, key).double())
    fy = y.clone(); fy[0, -1] = -1; V = net.V; rows = (0, 127, 255); cols = (0, V//2, V-1); eps = 1e-5; errors = {}
    gen = torch.Generator(device='cuda').manual_seed(25)
    def rand(*shape): return torch.randn(*shape, dtype=torch.float64, device='cuda', generator=gen)
    for l in (1, 2): probe.p[f'R{l}'] = .1*rand(256, V); probe.p[f'c{l}'] = .1*rand(V)  # R=0 would make W gradients trivially zero.
    hs, ss, us, ids, em = probe.forward(x)
    for l in (1, 2):
        u = us[l-1]; w, b, r, c = (probe.p[f'{k}{l}'] for k in 'WbRc'); prev = .5*rand(1, 128, V); h = (u @ w+b).tanh()
        grads = probe._local_grads(l, x, fy, h, prev+h @ r+c, u, ids, em, probe.M)
        def ce(): return F.cross_entropy((prev+(u @ w+b).tanh() @ r+c).flatten(0, 1), fy.flatten(), ignore_index=-1)
        def fd(tensor, index, analytic):
            value = tensor[index].item(); tensor[index] = value+eps; plus = ce(); tensor[index] = value-eps; minus = ce(); tensor[index] = value
            numeric = (plus-minus)/(2*eps); close(analytic[index], numeric, atol=1e-8, rtol=1e-5)
            return abs((analytic[index]-numeric).item())
        errors[l] = max(fd(w, (i, j), grads[f'W{l}']) for block in ((0, 511, 1023), (1024, 1151, 1279), (1280, 1407, 1535)) for i, j in zip(block, rows))
        errors[l] = max(errors[l], *(fd(b, j, grads[f'b{l}']) for j in rows), *(fd(r, (i, j), grads[f'R{l}']) for i, j in zip(rows, cols)), *(fd(c, j, grads[f'c{l}']) for j in cols))
    # A real step must use the stored cumulative logits at each layer.
    wanted = {l: probe._local_grads(l, x, fy, hs[l], ss[l], us[l-1] if l else None, ids, em, probe.M) for l in range(3)}
    probe.train(x, fy, pg)
    for l, grads in wanted.items():
        for key, g in grads.items():
            g = g/g.norm().clamp_min(1); close(probe.state[key][0], .1*g, atol=1e-12, rtol=1e-9); close(probe.state[key][1], .001*g.square(), atol=1e-12, rtol=1e-9)
    Q.plain((probe, pg)); del probe, pg, hs, ss, us, ids, em, wanted

    batches = list(R.epoch(train[:6141], 0, min(len(train), 6141), net.V)); assert len(batches) == 3
    # Zero readouts: step 1 moves only R and c upstairs; step 2 starts W and b.
    a = copy.deepcopy(net); ag = copy.deepcopy(gate); moved = ('W1', 'W2', 'b1', 'b2'); init = {k: a.p[k].clone() for k in moved}
    a.train(*batches[0], ag); assert a.lr == S.TOP
    for key in moved: close(a.p[key], init[key], atol=0, rtol=0)
    for key in zeroed: assert a.p[key].count_nonzero()
    a.train(*batches[1], ag)
    for key in moved: assert not torch.equal(a.p[key], init[key]), key
    close(a.p['g'], net.p['g'], atol=0, rtol=0); del a, ag, init
    # Layer 0 is untouched by the residual change, so it must track iter24.
    def lower_same(a, b):
        for key in LOWER:
            close(a.p[key], b.p[key], atol=1e-6, rtol=0)
            for s, t in zip(a.state[key], b.state[key]): close(s, t, atol=1e-6, rtol=0)
    a = copy.deepcopy(net); b = copy.deepcopy(ref); ag = copy.deepcopy(gate); bg = copy.deepcopy(gate)
    for bx, by in batches: a.train(bx, by, ag); b.train(bx, by, bg); lower_same(a, b)
    assert not torch.equal(a.p['W1'], b.p['W1']) and not torch.equal(a.p['R2'], b.p['R2']); del a, b, ag, bg
    # Decay reaches the update.
    n = net.total_steps; a = copy.deepcopy(net); b = copy.deepcopy(ref); ag = copy.deepcopy(gate); bg = copy.deepcopy(gate)
    a.step = b.step = n-1; a.train(*batches[0], ag); b.train(*batches[0], bg); assert a.lr == b.lr == S.BOTTOM; lower_same(a, b); del a, b, ag, bg, ref
    # Real order, one step in so that every parameter group has a nonzero gradient.
    trace = copy.deepcopy(net); tg = copy.deepcopy(gate); trace.train(*batches[0], tg); bx, by = batches[1]; expected = trace.forward(bx)
    initial = {k: p.clone() for k, p in trace.p.items()}; old_gate = tg.A.clone(); gate_before = I.stamp([tg.A, tg.m, tg.v]); events = []; running = [0]
    groups = [[f'R{l}', f'c{l}', f'b{l}'] + ([f'W{l}'] if l else LOWER[:10]) for l in range(3)]
    activate, update, gate_train = trace._layer, trace._update, tg.train
    def recorded_layer(l, *args):
        events.append(f'activate {l}'); out = activate(l, *args); running[0] = running[0]+out[1]
        close(out[0], expected[0][l], atol=1e-6, rtol=0); close(running[0], expected[1][l], atol=1e-6, rtol=0)
        if l: close(out[2], expected[2][l-1], atol=1e-6, rtol=0)
        return out
    def recorded_update(grads):
        l = next(l for l in range(3) if f'b{l}' in grads); assert set(grads) == set(groups[l]); update(grads); events.append(f'update {l}')
        assert all(not torch.equal(trace.p[k], initial[k]) for k in groups[l]), 'layer did not update'
        for keys in groups[l+1:]:
            for key in keys: close(trace.p[key], initial[key], atol=0, rtol=0)
        close(trace.p['g'], initial['g'], atol=0, rtol=0); assert I.stamp([tg.A, tg.m, tg.v]) == gate_before
    def recorded_gate(gx, p, gy):
        assert events[-1] == 'update 2'; close(p, expected[1][2].softmax(-1), atol=1e-6, rtol=0); gate_train(gx, p, gy); events.append('gate')
    trace._layer = recorded_layer; trace._update = recorded_update; tg.train = recorded_gate
    try: loss = trace.train(bx, by, tg)
    finally: del trace._layer, trace._update, tg.train
    assert events == ['activate 0', 'update 0', 'activate 1', 'update 1', 'activate 2', 'update 2', 'gate'] and not torch.equal(tg.A, old_gate)
    assert abs(loss-F.cross_entropy(expected[1][2].flatten(0, 1), by.flatten(), ignore_index=-1).item()) <= 1e-5
    close(trace.p['g'], net.p['g'], atol=0, rtol=0)
    hs, ss = trace.forward(bx)[:2]
    for l in (1, 2): close(ss[l]-ss[l-1], hs[l] @ trace.p[f'R{l}']+trace.p[f'c{l}'], atol=1e-4, rtol=0)
    Q.plain((trace, tg)); assert I.stamp((net, gate)) == before, 'checks wrote model state'
    close(torch.get_rng_state(), cpu_rng, atol=0, rtol=0); close(torch.cuda.get_rng_state(), cuda_rng, atol=0, rtol=0)
    result.update(residual_init=True, initial_rng=True, initial_logits_equal_layer0=True, residual_finite_difference=True, fd_max_abs_error={str(k): v for k, v in errors.items()},
                  train_uses_stored_cumulative=True, zero_readout_start=True, layer0_equals_iter24=True, schedule_applied=True, order_events=events, g_fixed=True, state_untouched=True)
    return result


def run(smoke=None, seed=0):
    if smoke is not None and smoke < 1: raise ValueError('--smoke must be positive')
    if seed < 0: raise ValueError('--seed must be non-negative')
    temporary = tempfile.TemporaryDirectory(prefix='iter25_') if smoke is not None else None
    try:
        name = 'iter25_residual.json' if seed == 0 else f'iter25_residual_seed{seed}.json'
        path = Path(temporary.name)/name if temporary else I.ROOT/'results'/name
        if path.exists(): raise FileExistsError(path)
        old = json.loads((I.ROOT/'results'/('iter24_decay.json' if seed == 0 else f'iter24_decay_seed{seed}.json')).read_text())
        assert old['settings']['seed'] == seed
        assert torch.cuda.is_available() and not torch.is_grad_enabled()
        torch.manual_seed(0); np.random.seed(0); torch.cuda.reset_peak_memory_stats()
        tr, te, V = B.load(); cut = int(len(tr)*.95); train = tr[:cut]
        if smoke is not None: train = train[:smoke]
        total = math.ceil(len(train)/(16*128))
        renamed, info = I.rename(te); old_windows = B.N_EVAL_WINDOWS
        net = Model(V, total); gate = C.Gate(V)
        x = torch.as_tensor(tr[:128].astype(np.int64), device='cuda')[None]
        y = torch.as_tensor(tr[1:129].astype(np.int64), device='cuda')[None]
        result = dict(checks=checks(net, gate, x, y, tr[:cut])); chars = 0
        if seed:
            seed0_embedding = net.p['E'].clone(); del net, gate; torch.cuda.empty_cache()
            torch.manual_seed(seed); np.random.seed(seed)
            net = Model(V, total); gate = C.Gate(V)
            assert not torch.equal(net.p['E'], seed0_embedding) and net.step == gate.step == 0; del seed0_embedding
        g0 = net.p['g'].clone()
        def predict(x, copy=True):
            logits = net.forward(x)[1][-1]
            return gate.predict(x, logits.softmax(-1))[0] if copy else logits.log_softmax(-1)
        torch.cuda.synchronize(); start = time.perf_counter()
        for x, y in R.epoch(train, seed, len(train), V):
            net.train(x, y, gate); chars += int((y >= 0).sum())
        torch.cuda.synchronize(); seconds = time.perf_counter()-start
        assert chars == len(train) and net.step == gate.step == total and net.lr == S.BOTTOM and torch.equal(net.p['g'], g0), (chars, len(train), net.step, gate.step, total, net.lr, net.p['g'].tolist(), g0.tolist())
        Q.plain((net, gate)); assert not torch.is_grad_enabled()
        try:
            if smoke is not None: B.N_EVAL_WINDOWS = 16
            before = I.stamp((net, gate))
            test = I.evaluate(predict, te); ren = I.evaluate(predict, renamed); val = I.evaluate(predict, tr[cut:])
            result.update(Q.novel(predict, te, renamed, json.loads((I.ROOT/'data/meta.json').read_text())['vocab']))
            result['test_loss_nocopy'] = I.evaluate(lambda x: predict(x, False), te)
            result['renamed_loss_nocopy'] = I.evaluate(lambda x: predict(x, False), renamed)
            result['val_loss_nocopy'] = I.evaluate(lambda x: predict(x, False), tr[cut:])
            result['per_layer_cumulative_test_loss'] = [I.evaluate(lambda x, l=l: net.logits(x, l), te) for l in range(net.L)]
            weights = torch.zeros(5, device='cuda'); x, _ = B.eval_windows(te)
            for xb in x.split(16):
                xb = xb.cuda(); weights += gate.predict(xb, net.forward(xb)[1][-1].softmax(-1))[1].sum((0, 1))
            result['mean_gate_weight_test'] = dict(zip(['net', *[f'L{L}' for L in C.LENS]], (weights/x.numel()).tolist()))
            assert I.stamp((net, gate)) == before, 'evaluation wrote model state'
        finally: B.N_EVAL_WINDOWS = old_windows
        settings = dict(seed=seed, device='cuda', mode='copy', variant='residual', base='iter24', bs=16, T=128, passes=1, train_end=cut, val_start=cut, training_partition_chars=len(train), smoke=smoke,
                        eval_windows=16 if smoke is not None else old_windows, sampling=f'non-overlapping target windows, shuffled with CPU generator seed {seed}', renaming=info, evaluation_frozen=True,
                        D=256, layers=3, embedding_width=32, lags=list(range(6)), decays=[.5,.8,.95], layer_input_width=1536, state_lags=[1,2], lr=S.TOP, lr_final=S.BOTTOM, lr_hold_fraction=S.HOLD,
                        lr_schedule='hold then cosine', lr_applies_to='network; gate fixed', optimizer='manual Adam', betas=[.9,.999], eps=1e-8, gradient_clip_norm=1, weight_decay=0,
                        residual='s_0=r_0; s_l=stored s_{l-1}+r_l; layer l trains CE on s_l, own parameters only', upper_readout_init='R1,c1,R2,c2 zero', final_logits='s_2', g='removed from learning; fixed [0,0,1] selector',
                        gate_input='old-weight s_2 probabilities', update_order='per-layer during forward sweep; gate after', suffix_lengths=C.LENS, gate_lr=1e-2, gate_lr_schedule='constant')
        reference = dict(test_loss=old['test_loss'], renamed_loss=old['renamed_loss'], val_loss=old['val_loss'], train_seconds=old['train_seconds'])
        result.update(test_loss=test, renamed_loss=ren, renamed_gap=ren-test, val_loss=val, val_improvement=reference['val_loss']-val, train_seconds=seconds, chars=chars, steps=net.step,
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, settings=settings, iter24_reference=reference, same_sampler_backprop_reference=dict(test_loss=1.340673, renamed_loss=1.686344))
        result['pass'] = dict(train=seconds < 90, val=val <= reference['val_loss']-.010)
        with path.open('x', encoding='utf-8') as f: json.dump(result, f, indent=2, allow_nan=False)
        return result
    finally:
        if temporary: temporary.cleanup()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', type=int, metavar='N', help='train on at most N prefix chars, evaluate 16 windows, use a cleaned temporary result outside results/')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(); print(json.dumps(run(a.smoke, a.seed), indent=2), flush=True)
