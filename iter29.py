"""iter27's 256/256/320 stack with iter26's layer-local softmax EMA gates."""
import argparse, copy, json, math, tempfile, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import iter24 as S
import iter26 as G
import iter27 as W
Q, R, C, I, B = S.Q, S.R, S.C, S.I, S.B
WIDTHS, LOWER, GATE_LR = W.WIDTHS, G.LOWER, G.GATE_LR
GATES = ('G1', 'd1', 'G2', 'd2')
close = torch.testing.assert_close
torch.set_grad_enabled(False)


class Model(W.Model):
    def __init__(self, V, total_steps):
        # Inherit iter27's dedicated-generator redraw verbatim; gates draw no RNG.
        super().__init__(V, total_steps)
        for l in (1, 2):
            for key, shape in ((f'G{l}', (256, 3)), (f'd{l}', (3,))):
                self.p[key] = self.p[f'W{l}'].new_zeros(shape)
                self.state[key] = [torch.zeros_like(self.p[key]), torch.zeros_like(self.p[key])]

    _pi = G.Model._pi

    # Port iter26's cooperative methods onto iter27. All output dimensions come
    # from W, b and R (320 at layer 2); the lower/input branch width stays 256.
    def _layer(self, l, lower, M, ids, em):
        if not l: return super()._layer(l, lower, M, ids, em)
        p = self.p; lag1 = torch.zeros_like(lower); lag2 = torch.zeros_like(lower)
        lag1[:, 1:] = lower[:, :-1]; lag2[:, 2:] = lower[:, :-2]
        gated = 3*self._pi(l, lower).movedim(-1, 1)[..., None]*(M @ lower[:, None])
        u = torch.cat([lower, *gated.unbind(1), lag1, lag2], -1)
        h = (u @ p[f'W{l}']+p[f'b{l}']).tanh()
        return h, h @ p[f'R{l}']+p[f'c{l}'], u

    def _local_grads(self, l, x, y, h, z, u, ids, em, M):
        if not l: return super()._local_grads(l, x, y, h, z, u, ids, em, M)
        p = self.p; e = Q.N.error(z, y); delta = Q.delta(h, e, p[f'R{l}']); w = p[f'W{l}']; lower = u[..., :256]
        # s_k already contains 3*pi_k; the softmax Jacobian needs no division by pi.
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
    result = C.checks(net, gate); Q.plain((net, gate)); assert not torch.is_grad_enabled()
    cpu_rng = torch.get_rng_state(); cuda_rng = torch.cuda.get_rng_state()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(0); ref = W.Model(net.V, net.total_steps); rg = C.Gate(net.V)
        close(torch.get_rng_state(), cpu_rng, atol=0, rtol=0)
        close(torch.cuda.get_rng_state(), cuda_rng, atol=0, rtol=0)
    assert set(net.p) == set(ref.p)|set(GATES) and set(net.state) == set(net.p)
    for key, p in net.p.items():
        if key in ref.p: close(p, ref.p[key], atol=0, rtol=0)
        else: assert not p.count_nonzero()
        assert len(net.state[key]) == 2
        for state in net.state[key]: assert state.shape == p.shape and not state.count_nonzero()
    assert vars(gate).keys() == vars(rg).keys()
    for key, value in vars(gate).items():
        if isinstance(value, torch.Tensor): close(value, getattr(rg, key), atol=0, rtol=0)
        else: assert value == getattr(rg, key)
    assert net.p['W2'].shape == (1536, 320) and net.p['b2'].shape == (320,) and net.p['R2'].shape == (320, net.V)
    for l in (1, 2):
        assert net.p[f'W{l}'].shape == (1536, WIDTHS[l]) and not net.p[f'W{l}'][1024:].count_nonzero()
        assert net.p[f'G{l}'].shape == (256, 3) and net.p[f'd{l}'].shape == (3,)
    hs, zs, us, _, _ = net.forward(x); old = ref.forward(x)
    for l, width in enumerate(WIDTHS):
        assert hs[l].shape == (*x.shape, width) and zs[l].shape == (*x.shape, net.V)
        close(hs[l], old[0][l], atol=1e-6, rtol=0); close(zs[l], old[1][l], atol=1e-6, rtol=0)
    for l in (1, 2):
        pi = net._pi(l, hs[l-1]); close(pi, torch.full_like(pi, 1/3), atol=0, rtol=0)
        close(3*pi, torch.ones_like(pi), atol=0, rtol=0)
        assert us[l-1].shape == (*x.shape, 1536)
        close(us[l-1], old[2][l-1], atol=1e-6, rtol=0)
        for lag, start in ((1, 1024), (2, 1280)):
            block = us[l-1][..., start:start+256]
            close(block[:, lag:], hs[l-1][:, :-lag], atol=0, rtol=0)
            assert not block[:, :lag].count_nonzero()
    del hs, zs, us, old

    # Check causality with active, content-dependent gates, not just zero gates.
    probe = copy.deepcopy(net)
    gen = torch.Generator(device=x.device).manual_seed(29)
    for l in (1, 2):
        probe.p[f'G{l}'] = .1*torch.randn((256, 3), device=x.device, generator=gen)
        probe.p[f'd{l}'] = .3*torch.randn((3,), device=x.device, generator=gen)
    hs, zs = probe.forward(x)[:2]
    for t in (0, 7, 46, 126):
        other = x.clone(); other[:, t+1:] = (other[:, t+1:]+1)%net.V
        changed = probe.forward(other)
        for a, b in zip(hs+zs, changed[0]+changed[1]): close(a[:, :t+1], b[:, :t+1], atol=1e-6, rtol=0)
    del hs, zs, changed

    # Tiny float64 losses with detached lower states: autograd rebuilds the EMA
    # gates and all six input blocks independently, checking every coordinate.
    pg = copy.deepcopy(gate)
    probe.p = {k: p.double() for k, p in probe.p.items()}; probe.M = probe.M.double()
    probe.state = {k: [s.double() for s in ss] for k, ss in probe.state.items()}
    for key in ('A', 'm', 'v'): setattr(pg, key, getattr(pg, key).double())
    fx, fy = x[:, :4], y[:, :4].clone(); fy[0, 1] = -1
    hs, zs, us, ids, em = probe.forward(fx); M = probe._prepare(fx)[0]; errors = {}
    wanted = {l: probe._local_grads(l, fx, fy, hs[l], zs[l], us[l-1] if l else None, ids, em, M) for l in range(3)}
    for l in (1, 2):
        lower = hs[l-1].detach(); analytic = wanted[l]
        assert analytic[f'G{l}'].count_nonzero() and analytic[f'd{l}'].count_nonzero()
        pi = probe._pi(l, lower); assert not torch.isclose(pi, torch.full_like(pi, 1/3)).all()
        keys = [f'{k}{l}' for k in ('W', 'b', 'R', 'c', 'G', 'd')]
        with torch.enable_grad():
            w, b, r, c, gg, d = params = [probe.p[k].detach().clone().requires_grad_() for k in keys]
            pi = (lower @ gg+d).softmax(-1)
            ema = M @ lower[:, None]
            lag1 = torch.zeros_like(lower); lag2 = torch.zeros_like(lower)
            lag1[:, 1:] = lower[:, :-1]; lag2[:, 2:] = lower[:, :-2]
            u = torch.cat([lower, *[3*pi[..., k, None]*ema[:, k] for k in range(3)], lag1, lag2], -1)
            a = u @ w+b; h = a.tanh(); z = h @ r+c
            loss = F.cross_entropy(z.flatten(0, 1), fy.flatten(), ignore_index=-1)
            grads = torch.autograd.grad(loss, [*params, a])
        for key, grad in zip(keys, grads):
            close(analytic[key], grad, atol=1e-12, rtol=1e-10)
            errors[key] = (analytic[key]-grad).abs().max().item()
        close(Q.delta(hs[l], Q.N.error(zs[l], fy), probe.p[f'R{l}']), grads[-1], atol=1e-12, rtol=1e-10)
    assert not torch.is_grad_enabled()
    # Stored pre-update activations must also generate the actual Adam moments.
    old = {k: p.clone() for k, p in probe.p.items()}; probe.train(fx, fy, pg)
    assert probe.lr == S.schedule(1, net.total_steps)
    for local in wanted.values():
        for key, grad in local.items():
            grad = grad/grad.norm().clamp_min(1)
            close(probe.state[key][0], .1*grad, atol=1e-12, rtol=1e-10)
            close(probe.state[key][1], .001*grad.square(), atol=1e-12, rtol=1e-10)
            lr = probe.lr*(GATE_LR if key in GATES else 1)
            close(probe.p[key]-old[key], -lr*grad/(grad.abs()+1e-8), atol=1e-11, rtol=1e-6)
    Q.plain((probe, pg))
    del probe, pg, hs, zs, us, ids, em, wanted, old, params, grads, a, h, z, loss, analytic, local

    # iter26's LR routing check, extended to both upper layers.
    a = copy.deepcopy(net); a.lr = 5e-4; a.step = 1
    keys = [f'{k}{l}' for l in (1, 2) for k in ('G', 'd', 'W')]; init = {k: a.p[k].clone() for k in keys}
    a._update({k: torch.ones_like(a.p[k]) for k in keys}); assert a.lr == 5e-4
    for key in keys:
        lr = 5e-4*(GATE_LR if key in GATES else 1)
        close(a.p[key]-init[key], torch.full_like(init[key], -lr), atol=0, rtol=2e-3)
    del a, init

    batches = list(R.epoch(train[:6141], 0, min(len(train), 6141), net.V)); assert len(batches) == 3
    # Paired steps use deterministic index_add_ to avoid amplifying its roundoff
    # through Adam. Restore the caller's setting before actual training.
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        a, b = copy.deepcopy(net), copy.deepcopy(ref); ag, bg = copy.deepcopy(gate), copy.deepcopy(rg)
        assert abs(a.train(*batches[0], ag)-b.train(*batches[0], bg)) <= 1e-6
        S.same(b, a, bg, ag)
        for key in GATES: assert a.p[key].count_nonzero(), key
        for bx, by in batches[1:]:
            a.train(bx, by, ag); b.train(bx, by, bg)
            for key in LOWER:
                close(a.p[key], b.p[key], atol=1e-6, rtol=0)
                for s, t in zip(a.state[key], b.state[key]): close(s, t, atol=1e-6, rtol=0)
        assert not torch.equal(a.p['W1'], b.p['W1']) and not torch.equal(a.p['W2'], b.p['W2'])
        del a, b, ag, bg
        # iter24/27 schedule checks, including 0.1x gate LR during decay.
        assert S.schedule(1, 8) == S.schedule(6, 8) == S.TOP and S.schedule(8, 8) == S.BOTTOM
        assert abs(S.schedule(7, 8)-5.5e-4) < 1e-15
        n = net.total_steps; hold = math.floor(S.HOLD*n); lrs = [S.schedule(s, n) for s in range(1, n+1)]
        assert all(v == S.TOP for v in lrs[:hold]) and lrs[hold] < S.TOP and lrs[-1] == S.BOTTOM
        assert all(a > b for a, b in zip(lrs[hold:], lrs[hold+1:]))
        for step in sorted({1, hold+1, n}):
            a, b = copy.deepcopy(net), copy.deepcopy(net); ag, bg = copy.deepcopy(gate), copy.deepcopy(gate)
            # Active gates exercise schedule routing after the zero-init step.
            a.train(*batches[0], ag); b.train(*batches[0], bg)
            a.step = b.step = ag.step = bg.step = step-1; b.lr = S.schedule(step, n)
            a.train(*batches[1], ag); S.S.Model.train(b, *batches[1], bg)
            assert a.lr == b.lr == S.schedule(step, n); S.same(a, b, ag, bg)
        del a, b, ag, bg
        a, b = copy.deepcopy(net), copy.deepcopy(ref); ag, bg = copy.deepcopy(gate), copy.deepcopy(rg)
        a.step = b.step = n-1; a.train(*batches[0], ag); b.train(*batches[0], bg)
        assert a.lr == b.lr == S.BOTTOM; S.same(b, a, bg, ag)
        del a, b, ag, bg
    finally: torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    del ref, rg

    # iter26's observed interleaving, with nonzero gradients in every group.
    trace, tg = copy.deepcopy(net), copy.deepcopy(gate); trace.train(*batches[0], tg)
    bx, by = batches[1]; expected = trace.forward(bx)
    initial = {k: p.clone() for k, p in trace.p.items()}; old_gate = tg.A.clone(); events = []
    final = sum(w*z for w, z in zip(initial['g'], expected[1]))
    groups = [LOWER, *[[f'{k}{l}' for k in 'RcbWGd'] for l in (1, 2)]]
    activate, update, gate_train = trace._layer, trace._update, tg.train
    def recorded_layer(l, *args):
        assert trace.step == 2; events.append(f'activate {l}'); out = activate(l, *args)
        close(out[0], expected[0][l], atol=1e-6, rtol=0); close(out[1], expected[1][l], atol=1e-6, rtol=0)
        if l: close(out[2], expected[2][l-1], atol=1e-6, rtol=0)
        return out
    def recorded_update(grads):
        if set(grads) == {'g'}:
            assert events[-1] == 'gate' and tg.step == 2
            update(grads); events.append('update g'); assert not torch.equal(trace.p['g'], initial['g']); return
        l = next(l for l in range(3) if f'b{l}' in grads); assert set(grads) == set(groups[l])
        update(grads); events.append(f'update {l}')
        assert all(not torch.equal(trace.p[k], initial[k]) for k in groups[l]), 'layer did not update'
        for keys in groups[l+1:]:
            for key in keys: close(trace.p[key], initial[key], atol=0, rtol=0)
        close(trace.p['g'], initial['g'], atol=0, rtol=0); assert I.stamp(tg) == gate_before
    def recorded_gate(gx, p, gy):
        assert events[-1] == 'update 2'; close(p, final.softmax(-1), atol=1e-6, rtol=0)
        gate_train(gx, p, gy); events.append('gate')
    trace._layer = recorded_layer; trace._update = recorded_update; tg.train = recorded_gate
    gate_before = I.stamp(tg)  # vars(tg) includes the monkeypatch: stamp afterwards.
    try: loss = trace.train(bx, by, tg)
    finally: del trace._layer, trace._update, tg.train
    assert events == ['activate 0', 'update 0', 'activate 1', 'update 1', 'activate 2', 'update 2', 'gate', 'update g']
    assert not torch.equal(tg.A, old_gate)
    assert abs(loss-F.cross_entropy(final.flatten(0, 1), by.flatten(), ignore_index=-1).item()) <= 1e-6
    Q.plain((trace, tg)); assert I.stamp((net, gate)) == before, 'checks wrote model state'
    close(torch.get_rng_state(), cpu_rng, atol=0, rtol=0); close(torch.cuda.get_rng_state(), cuda_rng, atol=0, rtol=0)
    Q.plain((net, gate)); assert not torch.is_grad_enabled()
    result.update(widths=WIDTHS, shapes_and_zero_states=True, lag_rows_zero=True, gates_zero_init=True,
                  initial_non_gate_equal_iter27=True, initial_copy_gate_equal_iter27=True, initial_rng=True,
                  initial_logits_equal_iter27=True, initial_forward_atol=1e-6, input_blocks=True, all_layer_causality=True,
                  local_autograd=True, local_autograd_max_abs_error=errors, train_gradient=True, gate_lr_fraction=GATE_LR,
                  first_step_equals_iter27=True, layer0_equals_iter27=True, schedule_values=True, scheduled_lr_applied=True,
                  order_interleaved=True, order_events=events, state_untouched=True)
    return result


def run(smoke=None, seed=0):
    if smoke is not None and smoke < 1: raise ValueError('--smoke must be positive')
    if seed < 0: raise ValueError('--seed must be non-negative')
    temporary = tempfile.TemporaryDirectory(prefix='iter29_') if smoke is not None else None
    try:
        name = 'iter29_gates320.json' if seed == 0 else f'iter29_gates320_seed{seed}.json'
        path = Path(temporary.name)/name if temporary else I.ROOT/'results'/name
        if path.exists(): raise FileExistsError(path)
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
        if seed:  # Startup checks always compare seed-0 construction.
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
        assert chars == len(train) and net.step == gate.step == total and net.lr == S.BOTTOM
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
        settings = dict(seed=seed, device='cuda', mode='copy', variant='gates320', base='iter27 + iter26 EMA gates', bs=16, T=128, passes=1,
                        train_end=cut, val_start=cut, training_partition_chars=len(train), smoke=smoke, eval_windows=16 if smoke is not None else old_windows,
                        sampling=f'non-overlapping target windows, shuffled with CPU generator seed {seed}', renaming=info, evaluation_frozen=True,
                        D=256, widths=WIDTHS, layers=3, embedding_width=32, lags=list(range(6)), decays=[.5,.8,.95], layer_input_width=1536, state_lags=[1,2],
                        lr=S.TOP, lr_final=S.BOTTOM, lr_hold_fraction=S.HOLD, lr_schedule='constant for s <= .75*S, then cosine to lr_final at s = S (1-indexed)',
                        lr_applies_to='network including g; EMA gates 0.1x network; copy gate constant', optimizer='manual Adam', betas=[.9,.999], eps=1e-8, gradient_clip_norm=1, weight_decay=0,
                        ema_gate='pi=softmax(lower @ G + d), one per position, each EMA block scaled by 3*pi_k; G,d zero-init; layer-local exact gradient',
                        ema_gate_layers=[1,2], ema_gate_shapes=dict(G=[256,3], d=[3]), ema_gate_lr_fraction=GATE_LR,
                        final_logits='learned g mix', gate_input='old-weight mixed probabilities',
                        update_order='per-layer during forward sweep; copy gate, then g', suffix_lengths=C.LENS, gate_lr=1e-2, gate_lr_schedule='constant')
        result.update(test_loss=test, renamed_loss=ren, renamed_gap=ren-test, val_loss=val, g=net.p['g'].tolist(), train_seconds=seconds,
                      chars=chars, steps=net.step, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, settings=settings,
                      iter27_reference=dict(val_loss=1.45627, test_loss=1.3232, renamed_loss=1.6232),
                      iter24_reference=dict(val_loss=1.46686, test_loss=1.3320, renamed_loss=1.6298),
                      iter22_reference=dict(test_loss=1.3568, renamed_loss=1.6462),
                      iter23_reference=dict(test_loss=1.3530, renamed_loss=1.6430),
                      same_sampler_backprop_reference=dict(test_loss=1.340673, renamed_loss=1.686344))
        result['pass'] = dict(train=seconds < 90, val=val < result['iter24_reference']['val_loss'], test=test <= 1.329, renamed=ren <= 1.667)
        with path.open('x', encoding='utf-8') as f: json.dump(result, f, indent=2, allow_nan=False)
        return result
    finally:
        if temporary: temporary.cleanup()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', type=int, metavar='N', help='train on at most N prefix chars, evaluate 16 windows, use a cleaned temporary result outside results/')
    p.add_argument('--seed', type=int, default=0, help='init and sampler seed; startup checks always use a seed-0 model')
    a = p.parse_args(); print(json.dumps(run(a.smoke, a.seed), indent=2), flush=True)
