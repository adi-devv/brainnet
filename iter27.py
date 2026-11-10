"""iter24 with only the final layer widened: 256/256/320."""
import argparse, copy, json, math, tempfile, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import iter24 as S
Q, R, C, I, B = S.Q, S.R, S.C, S.I, S.B
torch.set_grad_enabled(False)
WIDTHS = [256, 256, 320]
REPLACED = ('W2', 'b2', 'R2')
close = torch.testing.assert_close


class Model(S.Model):
    def __init__(self, V, total_steps):
        # Consume exactly iter24's global RNG draws, including its original top layer.
        super().__init__(V, total_steps)
        gen = torch.Generator(device=self.p['W2'].device).manual_seed(torch.initial_seed())
        def rand(shape): return torch.randn(shape, device=self.p['W2'].device, generator=gen)
        # iter14 initializes 1024 projection rows; iter22 appends zero lag rows.
        self.p['W2'] = torch.cat([rand((1024, 320))/math.sqrt(1024), self.p['W2'].new_zeros(512, 320)], 0)
        self.p['b2'] = rand((320,))*0.
        self.p['R2'] = rand((320, V))*(1/math.sqrt(320))
        for key in REPLACED:
            self.state[key] = [torch.zeros_like(self.p[key]), torch.zeros_like(self.p[key])]
        # Inherited forward, local delta, Adam, logits and schedule are shape-driven.


def checks(net, gate, x, y, train):
    before = I.stamp((net, gate)); assert net.lr == S.TOP and net.step == gate.step == 0
    result = C.checks(net, gate); Q.plain((net, gate)); assert not torch.is_grad_enabled()
    cpu_rng = torch.get_rng_state(); cuda_rng = torch.cuda.get_rng_state()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(0); ref = S.Model(net.V, net.total_steps); rg = C.Gate(net.V)
        close(torch.get_rng_state(), cpu_rng, atol=0, rtol=0)
        close(torch.cuda.get_rng_state(), cuda_rng, atol=0, rtol=0)
    assert set(net.p) == set(ref.p) == set(net.state)
    for key, p in net.p.items():
        if key not in REPLACED: close(p, ref.p[key], atol=0, rtol=0)
        assert len(net.state[key]) == 2
        for state in net.state[key]: assert state.shape == p.shape and not state.count_nonzero()
    assert vars(gate).keys() == vars(rg).keys()
    for key, value in vars(gate).items():
        if isinstance(value, torch.Tensor): close(value, getattr(rg, key), atol=0, rtol=0)
        else: assert value == getattr(rg, key)
    assert net.p['W2'].shape == (1536, 320) and net.p['b2'].shape == (320,) and net.p['R2'].shape == (320, net.V)
    assert not net.p['W2'][1024:1536].count_nonzero()
    assert net.p['W1'].shape == (1536, 256) and not net.p['W1'][1024:].count_nonzero()
    hs, zs, us, _, _ = net.forward(x); old = ref.forward(x)
    for l, width in enumerate(WIDTHS):
        assert hs[l].shape == (*x.shape, width) and zs[l].shape == (*x.shape, net.V)
        if l < 2:
            close(hs[l], old[0][l], atol=0, rtol=0); close(zs[l], old[1][l], atol=0, rtol=0)
    for l in (1, 2):
        assert us[l-1].shape == (*x.shape, 1536)
        close(us[l-1], old[2][l-1], atol=0, rtol=0)
        for lag, start in ((1, 1024), (2, 1280)):
            block = us[l-1][..., start:start+256]
            close(block[:, lag:], hs[l-1][:, :-lag], atol=0, rtol=0)
            assert not block[:, :lag].count_nonzero()
    # Check hidden states and each local readout, beyond C.checks' mixed copy outputs.
    for t in (0, 7, 46, 126):
        other = x.clone(); other[:, t+1:] = (other[:, t+1:]+1)%net.V
        changed = net.forward(other)
        for a, b in zip(hs+zs, changed[0]+changed[1]): close(a[:, :t+1], b[:, :t+1], atol=0, rtol=0)
    del hs, zs, us, old, changed

    # Tiny float64 local losses: frozen lower-layer inputs, all coordinates checked,
    # including the extra 64 units, both lag blocks, readout/bias and a masked target.
    probe = copy.deepcopy(net); pg = copy.deepcopy(gate)
    probe.p = {k: p.double() for k, p in probe.p.items()}; probe.M = probe.M.double()
    probe.state = {k: [s.double() for s in ss] for k, ss in probe.state.items()}
    for key in ('A', 'm', 'v'): setattr(pg, key, getattr(pg, key).double())
    fx, fy = x[:, :4], y[:, :4].clone(); fy[0, 1] = -1
    hs, zs, us, ids, em = probe.forward(fx); M = probe._prepare(fx)[0]; errors = {}; analytics = {}
    for l in (1, 2):
        u = us[l-1]; analytic = probe._local_grads(l, fx, fy, hs[l], zs[l], u, ids, em, M); analytics.update(analytic)
        keys = [f'{k}{l}' for k in ('W', 'b', 'R', 'c')]
        with torch.enable_grad():
            w, b, r, c = params = [probe.p[k].detach().clone().requires_grad_() for k in keys]
            a = u @ w+b; h = a.tanh(); z = h @ r+c
            loss = F.cross_entropy(z.flatten(0, 1), fy.flatten(), ignore_index=-1)
            grads = torch.autograd.grad(loss, [*params, a])
        for key, grad in zip(keys, grads):
            close(analytic[key], grad, atol=1e-12, rtol=1e-10)
            errors[key] = (analytic[key]-grad).abs().max().item()
        close(Q.delta(hs[l], Q.N.error(zs[l], fy), probe.p[f'R{l}']), grads[-1], atol=1e-12, rtol=1e-10)
    assert not torch.is_grad_enabled()
    probe.train(fx, fy, pg)
    for key, grad in analytics.items():
        grad = grad/grad.norm().clamp_min(1)
        close(probe.state[key][0], .1*grad, atol=1e-12, rtol=1e-10)
        close(probe.state[key][1], .001*grad.square(), atol=1e-12, rtol=1e-10)
    Q.plain((probe, pg)); del probe, pg, hs, zs, us, ids, em, analytics, params, grads, a, h, z, loss

    # Real 16x128 batch: upper-layer changes cannot enter the lower local updates.
    batches = list(R.epoch(train[:6141], 0, min(len(train), 6141), net.V)); assert len(batches) == 3
    a, ag = copy.deepcopy(net), copy.deepcopy(gate)
    # CUDA index_add_ uses unordered atomics by default. Make this paired check
    # deterministic, then restore the caller's settings before actual training.
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        assert math.isfinite(a.train(*batches[0], ag)) and math.isfinite(ref.train(*batches[0], rg))
    finally: torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    for key in net.p:
        if key == 'g' or key in ('W2', 'b2', 'R2', 'c2'): continue
        close(a.p[key], ref.p[key], atol=0, rtol=0)
        for s, t in zip(a.state[key], ref.state[key]): close(s, t, atol=0, rtol=0)
    assert a.step == ref.step == ag.step == rg.step == 1
    del a, ag, ref, rg

    # iter24 schedule checks, applied to copies of the widened model.
    assert S.schedule(1, 8) == S.schedule(6, 8) == S.TOP and S.schedule(8, 8) == S.BOTTOM
    assert abs(S.schedule(7, 8)-5.5e-4) < 1e-15
    n = net.total_steps; hold = math.floor(S.HOLD*n); lrs = [S.schedule(s, n) for s in range(1, n+1)]
    assert all(v == S.TOP for v in lrs[:hold]) and lrs[hold] < S.TOP and lrs[-1] == S.BOTTOM
    assert all(a > b for a, b in zip(lrs[hold:], lrs[hold+1:]))
    for s in sorted({1, hold+1, n}):
        a, b = copy.deepcopy(net), copy.deepcopy(net); ag, bg = copy.deepcopy(gate), copy.deepcopy(gate)
        a.step = b.step = s-1; b.lr = S.schedule(s, n)
        a.train(*batches[0], ag); S.S.Model.train(b, *batches[0], bg)
        assert a.lr == b.lr; S.same(a, b, ag, bg)
    del a, b, ag, bg
    # Observe actual activation/update calls, checking both writes and the old-weight values.
    trace, tg = copy.deepcopy(net), copy.deepcopy(gate); bx, by = batches[0]; expected = trace.forward(bx)
    initial = {k: p.clone() for k, p in trace.p.items()}; gate_before = I.stamp(tg); events = []
    groups = [[f'R{l}', f'c{l}', f'b{l}'] + ([f'W{l}'] if l else [*[f'A{j}' for j in range(6)], 'E', *[f'P{k}' for k in range(3)]]) for l in range(3)]
    activate, update, gate_train = trace._layer, trace._update, tg.train
    def recorded_layer(l, *args):
        assert trace.step == 1; events.append(f'activate {l}'); out = activate(l, *args)
        for a, b in zip(out[:2], (expected[0][l], expected[1][l])): torch.testing.assert_close(a, b, atol=0, rtol=0)
        if l: torch.testing.assert_close(out[2], expected[2][l-1], atol=0, rtol=0)
        return out
    def recorded_update(grads):
        assert trace.step == 1
        if 'g' in grads:
            assert list(grads) == ['g'] and events[-1] == 'gate' and tg.step == 1
            assert not torch.equal(tg.A, gate.A); update(grads); events.append('g')
            assert not torch.equal(trace.p['g'], initial['g'])
            return
        l = next(l for l in range(3) if f'b{l}' in grads); assert set(grads) == set(groups[l])
        update(grads); events.append(f'update {l}')
        assert all(not torch.equal(trace.p[k], initial[k]) for k in groups[l]), 'layer did not update'
        for keys in groups[l+1:]:
            for key in keys: torch.testing.assert_close(trace.p[key], initial[key], atol=0, rtol=0)
        torch.testing.assert_close(trace.p['g'], initial['g'], atol=0, rtol=0)
        assert I.stamp(tg) == gate_before
    def recorded_gate(gx, p, gy):
        assert events[-1] == 'update 2'; torch.testing.assert_close(trace.p['g'], initial['g'], atol=0, rtol=0)
        old_final = sum(w*z for w, z in zip(initial['g'], expected[1]))
        torch.testing.assert_close(p, old_final.softmax(-1), atol=0, rtol=0)
        gate_train(gx, p, gy); events.append('gate')
    trace._layer = recorded_layer; trace._update = recorded_update; tg.train = recorded_gate
    gate_before = I.stamp(tg)
    try: trace.train(bx, by, tg)
    finally: del trace._layer, trace._update, tg.train
    assert events == ['activate 0', 'update 0', 'activate 1', 'update 1', 'activate 2', 'update 2', 'gate', 'g']
    Q.plain((trace, tg))
    assert I.stamp((net, gate)) == before, 'checks wrote model state'
    close(torch.get_rng_state(), cpu_rng, atol=0, rtol=0); close(torch.cuda.get_rng_state(), cuda_rng, atol=0, rtol=0)
    Q.plain((net, gate)); assert not torch.is_grad_enabled()
    result.update(widths=WIDTHS, shapes_and_zero_states=True, lag_rows_zero=True, initial_non_layer2_equal_iter24=True,
                  initial_gate_equal_iter24=True, initial_rng=True, input_blocks=True, all_layer_causality=True,
                  local_autograd=True, local_autograd_max_abs_error=errors, train_gradient=True,
                  first_step_lower_layers_equal_iter24=True, schedule_values=True, scheduled_lr_applied=True, order_interleaved=True, order_events=events, state_untouched=True)
    return result


def run(smoke=None, seed=0):
    if smoke is not None and smoke < 1: raise ValueError('--smoke must be positive')
    if seed < 0: raise ValueError('--seed must be non-negative')
    temporary = tempfile.TemporaryDirectory(prefix='iter27_') if smoke is not None else None
    try:
        name = 'iter27_width320.json' if seed == 0 else f'iter27_width320_seed{seed}.json'
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
        if seed:  # The checks compare against seed-0 construction, so they run first; then rebuild for this seed.
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
            result['per_layer_test_loss'] = [I.evaluate(lambda x, l=l: net.logits(x, l), te) for l in range(net.L)]
            weights = torch.zeros(5, device='cuda'); x, _ = B.eval_windows(te)
            for xb in x.split(16):
                xb = xb.cuda(); zs = net.forward(xb)[1]; p = sum(w*z for w, z in zip(net.p['g'], zs)).softmax(-1)
                weights += gate.predict(xb, p)[1].sum((0, 1))
            result['mean_gate_weight_test'] = dict(zip(['net', *[f'L{L}' for L in C.LENS]], (weights/x.numel()).tolist()))
            assert I.stamp((net, gate)) == before, 'evaluation wrote model state'
        finally: B.N_EVAL_WINDOWS = old_windows
        settings = dict(seed=seed, device='cuda', mode='copy', variant='width320', base='iter24', bs=16, T=128, passes=1,
                        train_end=cut, val_start=cut, training_partition_chars=len(train), smoke=smoke, eval_windows=16 if smoke is not None else old_windows,
                        sampling=f'non-overlapping target windows, shuffled with CPU generator seed {seed}', renaming=info, evaluation_frozen=True,
                        D=256, widths=WIDTHS, layers=3, embedding_width=32, lags=list(range(6)), decays=[.5,.8,.95], layer_input_width=1536, state_lags=[1,2],
                        lr=S.TOP, lr_final=S.BOTTOM, lr_hold_fraction=S.HOLD, lr_schedule='constant for s <= .75*S, then cosine to lr_final at s = S (1-indexed)',
                        lr_applies_to='all network parameters including g', optimizer='manual Adam', betas=[.9,.999], eps=1e-8, gradient_clip_norm=1, weight_decay=0,
                        update_order='per-layer during forward sweep; gate and g after', suffix_lengths=C.LENS, gate_lr=1e-2, gate_lr_schedule='constant')
        result.update(test_loss=test, renamed_loss=ren, renamed_gap=ren-test, val_loss=val, g=net.p['g'].tolist(), train_seconds=seconds,
                      chars=chars, steps=net.step, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, settings=settings,
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
