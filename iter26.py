"""iter24 with content-dependent softmax gates over the three EMA branches at each upper layer."""
import argparse, copy, json, math, tempfile, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import iter24 as S
Q, R, C, I, B = S.Q, S.R, S.C, S.I, S.B
torch.set_grad_enabled(False)
LOWER = [*[f'A{j}' for j in range(6)], 'E', *[f'P{k}' for k in range(3)], 'b0', 'R0', 'c0']
GATE_LR = .1  # Fraction of the network schedule: peak 1e-4, floor 1e-5.
close = torch.testing.assert_close


class Model(S.Model):
    def __init__(self, V, total_steps):
        super().__init__(V, total_steps)  # Zero-initialized gates draw no RNG, so construction matches iter24.
        for l in (1, 2):
            for key, shape in ((f'G{l}', (256, 3)), (f'd{l}', (3,))):
                self.p[key] = torch.zeros(shape, device='cuda'); self.state[key] = [torch.zeros(shape, device='cuda'), torch.zeros(shape, device='cuda')]

    def _pi(self, l, lower): return (lower @ self.p[f'G{l}']+self.p[f'd{l}']).softmax(-1)

    def _layer(self, l, lower, M, ids, em):
        if not l: return super()._layer(l, lower, M, ids, em)
        p = self.p; lag1 = torch.zeros_like(lower); lag2 = torch.zeros_like(lower)
        lag1[:, 1:] = lower[:, :-1]; lag2[:, 2:] = lower[:, :-2]
        gated = 3*self._pi(l, lower).movedim(-1, 1)[..., None]*(M @ lower[:, None])  # B,3,T,256
        u = torch.cat([lower, *gated.unbind(1), lag1, lag2], -1)
        h = (u @ p[f'W{l}']+p[f'b{l}']).tanh()
        return h, h @ p[f'R{l}']+p[f'c{l}'], u

    def _local_grads(self, l, x, y, h, z, u, ids, em, M):
        if not l: return super()._local_grads(l, x, y, h, z, u, ids, em, M)
        p = self.p; e = Q.N.error(z, y); delta = Q.delta(h, e, p[f'R{l}']); w = p[f'W{l}']; lower = u[..., :256]
        # Branch k adds u_k @ W_k = 3 pi_k v_k to the activation, so its projection on delta is pi_k q_k, with no division by pi.
        s = torch.stack([(delta*(u[..., 256*(k+1):256*(k+2)] @ w[256*(k+1):256*(k+2)])).sum(-1) for k in range(3)], -1)
        logit = s-self._pi(l, lower)*s.sum(-1, keepdim=True)
        return {f'R{l}': Q.N.readout_delta(h, e), f'c{l}': e.sum((0, 1)), f'b{l}': delta.sum((0, 1)), f'W{l}': Q.N.readout_delta(u, delta),
                f'G{l}': Q.N.readout_delta(lower, logit), f'd{l}': logit.sum((0, 1))}

    def _update(self, grads):
        gates = {k: g for k, g in grads.items() if k[0] in 'Gd'}; rest = {k: g for k, g in grads.items() if k not in gates}
        if rest: super()._update(rest)
        if gates:
            lr = self.lr; self.lr = lr*GATE_LR
            try: super()._update(gates)
            finally: self.lr = lr


def checks(net, gate, x, y, train):
    before = I.stamp((net, gate)); assert net.lr == S.TOP and net.step == gate.step == 0
    result = C.checks(net, gate); Q.plain((net, gate)); assert not torch.is_grad_enabled() and net.step == gate.step == 0
    cpu_rng = torch.get_rng_state(); cuda_rng = torch.cuda.get_rng_state()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(0); ref = S.Model(net.V, net.total_steps)
        close(torch.get_rng_state(), cpu_rng, atol=0, rtol=0); close(torch.cuda.get_rng_state(), cuda_rng, atol=0, rtol=0)
    assert set(net.p) == set(ref.p)|{'G1', 'd1', 'G2', 'd2'} and set(net.state) == set(net.p)
    for key, p in net.p.items():
        if key in ref.p: close(p, ref.p[key], atol=0, rtol=0)
        else: assert not p.count_nonzero()
        for s in net.state[key]: assert not s.count_nonzero() and s.shape == p.shape
    a, b = net.forward(x), ref.forward(x)
    for l in range(3): close(a[0][l], b[0][l], atol=1e-5, rtol=0); close(a[1][l], b[1][l], atol=1e-5, rtol=0)
    for l in range(2): close(a[2][l], b[2][l], atol=1e-5, rtol=0)  # Starts as iter24.
    del a, b

    # Float64 finite differences of each layer's own loss, with gates moved away from zero and frozen lower states.
    probe = copy.deepcopy(net); pg = copy.deepcopy(gate)
    probe.p = {k: p.double() for k, p in probe.p.items()}; probe.M = probe.M.double()
    probe.state = {k: [s.double() for s in ss] for k, ss in probe.state.items()}
    for key in ('A', 'm', 'v'): setattr(pg, key, getattr(pg, key).double())
    fy = y.clone(); fy[0, -1] = -1; rows = (0, 127, 255); eps = 1e-5; errors = {}
    gen = torch.Generator(device='cuda').manual_seed(26)
    def rand(*shape): return torch.randn(*shape, dtype=torch.float64, device='cuda', generator=gen)
    for l in (1, 2): probe.p[f'G{l}'] = .1*rand(256, 3); probe.p[f'd{l}'] = .3*rand(3)
    hs, zs, us, ids, em = probe.forward(x); M = probe._prepare(x)[0]
    for l in (1, 2):
        lower = hs[l-1]; h, z, u = probe._layer(l, lower, M, ids, em); grads = probe._local_grads(l, x, fy, h, z, u, ids, em, M)
        pi = probe._pi(l, lower); assert not torch.isclose(pi, torch.full_like(pi, 1/3)).all()
        def ce(): return F.cross_entropy(probe._layer(l, lower, M, ids, em)[1].flatten(0, 1), fy.flatten(), ignore_index=-1)
        def fd(tensor, index, analytic):
            value = tensor[index].item(); tensor[index] = value+eps; plus = ce(); tensor[index] = value-eps; minus = ce(); tensor[index] = value
            numeric = (plus-minus)/(2*eps); close(analytic[index], numeric, atol=1e-8, rtol=1e-5)
            return abs((analytic[index]-numeric).item())
        w, b, G, d = (probe.p[f'{k}{l}'] for k in 'WbGd')
        assert grads[f'G{l}'].count_nonzero() and grads[f'd{l}'].count_nonzero()
        errors[l] = max(*(fd(G, (i, k), grads[f'G{l}']) for i in rows for k in range(3)), *(fd(d, k, grads[f'd{l}']) for k in range(3)),
                        *(fd(w, (i, j), grads[f'W{l}']) for block in ((0, 100, 255), (256, 511, 1023), (1024, 1151, 1279), (1280, 1407, 1535)) for i, j in zip(block, rows)),
                        *(fd(b, j, grads[f'b{l}']) for j in rows))
    # A real step uses stored pre-update values, and gates move at 0.1x the network LR.
    wanted = {l: probe._local_grads(l, x, fy, hs[l], zs[l], us[l-1] if l else None, ids, em, M) for l in range(3)}
    old = {k: probe.p[k].clone() for l in (1, 2) for k in (f'G{l}', f'd{l}', f'W{l}')}; probe.train(x, fy, pg)
    assert probe.lr == S.TOP
    for l, grads in wanted.items():
        for key, g in grads.items():
            g = g/g.norm().clamp_min(1); close(probe.state[key][0], .1*g, atol=1e-12, rtol=1e-9); close(probe.state[key][1], .001*g.square(), atol=1e-12, rtol=1e-9)
            if key in old: close(probe.p[key]-old[key], -S.TOP*(GATE_LR if key[0] in 'Gd' else 1)*g/(g.abs()+1e-8), atol=1e-11, rtol=1e-6)
    Q.plain((probe, pg)); del probe, pg, hs, zs, us, ids, em, wanted, old

    # LR routing: gates at 0.1x, everything else at the current LR, and the LR is restored.
    a = copy.deepcopy(net); a.lr = 5e-4; a.step = 1; init = {k: a.p[k].clone() for k in ('G1', 'd1', 'W1')}
    a._update({'G1': torch.ones_like(a.p['G1']), 'd1': torch.ones_like(a.p['d1']), 'W1': torch.ones_like(a.p['W1'])}); assert a.lr == 5e-4
    for key, lr in (('G1', 5e-5), ('d1', 5e-5), ('W1', 5e-4)): close(a.p[key]-init[key], torch.full_like(init[key], -lr), atol=0, rtol=2e-3)
    del a, init

    batches = list(R.epoch(train[:6141], 0, min(len(train), 6141), net.V)); assert len(batches) == 3
    # One step from zero gates is exactly iter24: stored activations mean a gate update cannot reach this step's other updates.
    a = copy.deepcopy(net); b = copy.deepcopy(ref); ag = copy.deepcopy(gate); bg = copy.deepcopy(gate)
    a.train(*batches[0], ag); b.train(*batches[0], bg); S.same(b, a, bg, ag)
    for key in ('G1', 'd1', 'G2', 'd2'): assert a.p[key].count_nonzero(), key
    # After that the gates matter upstairs, but layer 0 never sees them.
    def lower_same(a, b):
        for key in LOWER:
            close(a.p[key], b.p[key], atol=1e-6, rtol=0)
            for s, t in zip(a.state[key], b.state[key]): close(s, t, atol=1e-6, rtol=0)
    for bx, by in batches[1:]: a.train(bx, by, ag); b.train(bx, by, bg); lower_same(a, b)
    assert not torch.equal(a.p['W1'], b.p['W1']) and not torch.equal(a.p['W2'], b.p['W2']); del a, b, ag, bg
    # The schedule reaches the last step.
    n = net.total_steps; a = copy.deepcopy(net); b = copy.deepcopy(ref); ag = copy.deepcopy(gate); bg = copy.deepcopy(gate)
    a.step = b.step = n-1; a.train(*batches[0], ag); b.train(*batches[0], bg); assert a.lr == b.lr == S.BOTTOM; S.same(b, a, bg, ag); del a, b, ag, bg, ref
    # Real order, one step in so that every parameter group has a nonzero gradient.
    trace = copy.deepcopy(net); tg = copy.deepcopy(gate); trace.train(*batches[0], tg); bx, by = batches[1]; expected = trace.forward(bx)
    initial = {k: p.clone() for k, p in trace.p.items()}; old_gate = tg.A.clone(); gate_before = I.stamp([tg.A, tg.m, tg.v]); events = []
    final = sum(w*z for w, z in zip(initial['g'], expected[1]))
    groups = [LOWER, *[[f'{k}{l}' for k in 'RcbWGd'] for l in (1, 2)]]
    activate, update, gate_train = trace._layer, trace._update, tg.train
    def recorded_layer(l, *args):
        events.append(f'activate {l}'); out = activate(l, *args)
        close(out[0], expected[0][l], atol=1e-5, rtol=0); close(out[1], expected[1][l], atol=1e-5, rtol=0)
        if l: close(out[2], expected[2][l-1], atol=1e-5, rtol=0)
        return out
    def recorded_update(grads):
        if set(grads) == {'g'}: update(grads); events.append('update g'); return
        l = next(l for l in range(3) if f'b{l}' in grads); assert set(grads) == set(groups[l]); update(grads); events.append(f'update {l}')
        assert all(not torch.equal(trace.p[k], initial[k]) for k in groups[l]), 'layer did not update'
        for keys in groups[l+1:]:
            for key in keys: close(trace.p[key], initial[key], atol=0, rtol=0)
        close(trace.p['g'], initial['g'], atol=0, rtol=0); assert I.stamp([tg.A, tg.m, tg.v]) == gate_before
    def recorded_gate(gx, p, gy):
        assert events[-1] == 'update 2'; close(p, final.softmax(-1), atol=1e-5, rtol=0); gate_train(gx, p, gy); events.append('gate')
    trace._layer = recorded_layer; trace._update = recorded_update; tg.train = recorded_gate
    try: loss = trace.train(bx, by, tg)
    finally: del trace._layer, trace._update, tg.train
    assert events == ['activate 0', 'update 0', 'activate 1', 'update 1', 'activate 2', 'update 2', 'gate', 'update g'] and not torch.equal(tg.A, old_gate)
    assert abs(loss-F.cross_entropy(final.flatten(0, 1), by.flatten(), ignore_index=-1).item()) <= 1e-5
    Q.plain((trace, tg)); assert I.stamp((net, gate)) == before, 'checks wrote model state'
    close(torch.get_rng_state(), cpu_rng, atol=0, rtol=0); close(torch.cuda.get_rng_state(), cuda_rng, atol=0, rtol=0)
    result.update(gates_zero_init=True, initial_rng=True, initial_logits_equal_iter24=True, gate_finite_difference=True, fd_max_abs_error={str(k): v for k, v in errors.items()}, gate_lr_fraction=GATE_LR,
                  first_step_equals_iter24=True, layer0_equals_iter24=True, schedule_applied=True, order_events=events, state_untouched=True)
    return result


def run(smoke=None, seed=0):
    if smoke is not None and smoke < 1: raise ValueError('--smoke must be positive')
    if seed < 0: raise ValueError('--seed must be non-negative')
    temporary = tempfile.TemporaryDirectory(prefix='iter26_') if smoke is not None else None
    try:
        name = 'iter26_ema_gates.json' if seed == 0 else f'iter26_ema_gates_seed{seed}.json'
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
        def predict(x, copy=True):
            zs = net.forward(x)[1]; logits = sum(w*z for w, z in zip(net.p['g'], zs))
            return gate.predict(x, logits.softmax(-1))[0] if copy else logits.log_softmax(-1)
        torch.cuda.synchronize(); start = time.perf_counter()
        for x, y in R.epoch(train, seed, len(train), V):
            net.train(x, y, gate); chars += int((y >= 0).sum())
        torch.cuda.synchronize(); seconds = time.perf_counter()-start
        assert chars == len(train) and net.step == gate.step == total and net.lr == S.BOTTOM, (chars, len(train), net.step, gate.step, total, net.lr)
        Q.plain((net, gate)); assert not torch.is_grad_enabled()
        try:
            if smoke is not None: B.N_EVAL_WINDOWS = 16
            before = I.stamp((net, gate))
            test = I.evaluate(predict, te); ren = I.evaluate(predict, renamed); val = I.evaluate(predict, tr[cut:])
            result.update(Q.novel(predict, te, renamed, json.loads((I.ROOT/'data/meta.json').read_text())['vocab']))
            result['test_loss_nocopy'] = I.evaluate(lambda x: predict(x, False), te)
            result['renamed_loss_nocopy'] = I.evaluate(lambda x: predict(x, False), renamed)
            result['val_loss_nocopy'] = I.evaluate(lambda x: predict(x, False), tr[cut:])
            result['per_layer_test_loss'] = [I.evaluate(lambda x, l=l: net.logits(x, l), te) for l in range(net.L)]
            weights = torch.zeros(5, device='cuda'); x, _ = B.eval_windows(te)
            pis = {l: [torch.zeros(3, device='cuda'), torch.zeros(3, device='cuda')] for l in (1, 2)}; saturated = torch.zeros(3, device='cuda')
            for xb in x.split(16):
                xb = xb.cuda(); hs, zs = net.forward(xb)[:2]; p = sum(w*z for w, z in zip(net.p['g'], zs)).softmax(-1)
                weights += gate.predict(xb, p)[1].sum((0, 1))
                for l in (1, 2):
                    pi = net._pi(l, hs[l-1]); pis[l][0] += pi.sum((0, 1)); pis[l][1] += pi.square().sum((0, 1))
                saturated += torch.stack([(h.abs() > .99).float().mean(-1).sum() for h in hs])
            result['mean_gate_weight_test'] = dict(zip(['net', *[f'L{L}' for L in C.LENS]], (weights/x.numel()).tolist()))
            result['ema_gate_test'] = {f'layer{l}': dict(mean=(m/x.numel()).tolist(), std=((s/x.numel()-(m/x.numel()).square()).clamp_min(0).sqrt()).tolist()) for l, (m, s) in pis.items()}
            result['saturated_fraction_test'] = (saturated/x.numel()).tolist()
            assert I.stamp((net, gate)) == before, 'evaluation wrote model state'
        finally: B.N_EVAL_WINDOWS = old_windows
        settings = dict(seed=seed, device='cuda', mode='copy', variant='ema_gates', base='iter24', bs=16, T=128, passes=1, train_end=cut, val_start=cut, training_partition_chars=len(train), smoke=smoke,
                        eval_windows=16 if smoke is not None else old_windows, sampling=f'non-overlapping target windows, shuffled with CPU generator seed {seed}', renaming=info, evaluation_frozen=True,
                        D=256, layers=3, embedding_width=32, lags=list(range(6)), decays=[.5,.8,.95], layer_input_width=1536, state_lags=[1,2], lr=S.TOP, lr_final=S.BOTTOM, lr_hold_fraction=S.HOLD,
                        lr_schedule='hold then cosine', lr_applies_to='network; copy gate fixed; EMA gates 0.1x network', optimizer='manual Adam', betas=[.9,.999], eps=1e-8, gradient_clip_norm=1, weight_decay=0,
                        ema_gate='pi=softmax(lower @ G + d), one per position, each EMA block scaled by 3*pi_k; G,d zero-init; layer-local exact gradient', ema_gate_lr_fraction=GATE_LR,
                        final_logits='learned g mix', gate_input='old-weight mixed probabilities', update_order='per-layer during forward sweep; copy gate, then g', suffix_lengths=C.LENS, gate_lr=1e-2, gate_lr_schedule='constant')
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
