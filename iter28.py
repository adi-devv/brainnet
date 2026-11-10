"""iter24 with h = .95*tanh(a) + .05*a in layers 1 and 2 only."""
import argparse, copy, json, math, tempfile, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import iter24 as S
Q, R, C, I, B = S.Q, S.R, S.C, S.I, S.B
torch.set_grad_enabled(False)
close = torch.testing.assert_close
LOWER = [*[f'A{j}' for j in range(6)], 'E', *[f'P{k}' for k in range(3)], 'b0', 'R0', 'c0']


def derivative(t): return .95*(1-t.square())+.05


class Model(S.Model):
    # Inherit construction verbatim: identical parameters, optimizer states and RNG draws.
    def _layer(self, l, lower, M, ids, em):
        if not l: return super()._layer(l, lower, M, ids, em)
        p = self.p; lag1 = torch.zeros_like(lower); lag2 = torch.zeros_like(lower)
        lag1[:, 1:] = lower[:, :-1]; lag2[:, 2:] = lower[:, :-2]
        u = torch.cat([lower, *(M @ lower[:, None]).unbind(1), lag1, lag2], -1)
        a = u @ p[f'W{l}']+p[f'b{l}']; t = a.tanh(); h = .95*t+.05*a
        # The inherited sweep passes this local payload through unchanged. Keep tanh(a),
        # not h, for the derivative without recomputation or persistent model caches.
        return h, h @ p[f'R{l}']+p[f'c{l}'], (u, t)

    def _local_grads(self, l, x, y, h, z, u, ids, em, M):
        if not l: return super()._local_grads(l, x, y, h, z, u, ids, em, M)
        u, t = u; p = self.p; e = Q.N.error(z, y)
        d = (e @ p[f'R{l}'].T)*derivative(t)
        return {f'R{l}': Q.N.readout_delta(h, e), f'c{l}': e.sum((0, 1)),
                f'b{l}': d.sum((0, 1)), f'W{l}': Q.N.readout_delta(u, d)}


def autograd_checks():
    # Tiny deterministic float64 cases include masked targets and strongly saturated units.
    # Only this independent oracle enables autograd; training remains manual and layer-local.
    probe = object.__new__(Model); errors = {}
    lower = torch.arange(40, dtype=torch.float64).reshape(2, 5, 4).sin()*2
    distance = torch.arange(5)[:, None]-torch.arange(5)[None, :]
    M = torch.stack([(1-d)*d**distance.clamp_min(0)*(distance >= 0) for d in (.5, .8, .95)]).double()
    y = torch.tensor([[0, 1, -1, 2, 0], [2, -1, 1, 0, 2]])
    for l in (1, 2):
        probe.p = {f'W{l}': torch.arange(96, dtype=torch.float64).reshape(24, 4).cos()*(.3*l),
                   f'b{l}': torch.tensor([-9., -.2, .4, 9.], dtype=torch.float64),
                   f'R{l}': torch.arange(12, dtype=torch.float64).reshape(4, 3).sin()*.4,
                   f'c{l}': torch.tensor([.2, -.1, .3], dtype=torch.float64)}
        h, z, cache = probe._layer(l, lower, M, None, None); u, t = cache
        assert (t.abs() > .99).any() and (h.abs() > 1).any()
        manual = probe._local_grads(l, None, y, h, z, cache, None, None, M)
        keys = list(manual)
        with torch.enable_grad():
            p = {k: v.detach().clone().requires_grad_() for k, v in probe.p.items()}
            a = u.detach() @ p[f'W{l}']+p[f'b{l}']
            expected_h = .95*torch.tanh(a)+.05*a
            expected_z = expected_h @ p[f'R{l}']+p[f'c{l}']
            loss = F.cross_entropy(expected_z.flatten(0, 1), y.flatten(), ignore_index=-1)
            grads = torch.autograd.grad(loss, [p[k] for k in keys])
        close(h, expected_h.detach(), atol=0, rtol=0)
        close(z, expected_z.detach(), atol=0, rtol=0)
        for key, grad in zip(keys, grads):
            close(manual[key], grad, atol=1e-12, rtol=1e-10)
            errors[key] = (manual[key]-grad).abs().max().item()
    assert not torch.is_grad_enabled()
    return errors


def checks(net, gate, x, y, train):
    before = I.stamp((net, gate)); assert net.lr == S.TOP and net.step == gate.step == 0
    result = C.checks(net, gate); Q.plain((net, gate))  # Unchanged copy-head and mixed-output checks.
    cpu_rng = torch.get_rng_state(); cuda_rng = torch.cuda.get_rng_state()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(0); ref = S.Model(net.V, net.total_steps)
        close(torch.get_rng_state(), cpu_rng, atol=0, rtol=0)
        close(torch.cuda.get_rng_state(), cuda_rng, atol=0, rtol=0)
    assert set(net.p) == set(ref.p) == set(net.state)
    for key in net.p:
        close(net.p[key], ref.p[key], atol=0, rtol=0)
        for a, b in zip(net.state[key], ref.state[key]): close(a, b, atol=0, rtol=0)
    hs, zs, us, ids, em = net.forward(x); old = ref.forward(x)
    close(hs[0], old[0][0], atol=0, rtol=0); close(zs[0], old[1][0], atol=0, rtol=0)
    M = net._prepare(x)[0]
    for l in (1, 2):
        u, t = us[l-1]; lower = hs[l-1]
        close(u[..., :256], lower, atol=0, rtol=0)
        close(u[..., 256:1024], torch.cat((M @ lower[:, None]).unbind(1), -1), atol=0, rtol=0)
        for lag, start in ((1, 1024), (2, 1280)):
            block = u[..., start:start+256]
            close(block[:, lag:], lower[:, :-lag], atol=0, rtol=0); assert not block[:, :lag].count_nonzero()
        a = u @ net.p[f'W{l}']+net.p[f'b{l}']
        close(t, a.tanh(), atol=0, rtol=0); close(hs[l], .95*t+.05*a, atol=0, rtol=0)
    for t in (0, 7, 46, 126):
        other = x.clone(); other[:, t+1:] = (other[:, t+1:]+17)%net.V
        altered = net.forward(other)
        for l in range(3):
            close(hs[l][:, :t+1], altered[0][l][:, :t+1], atol=0, rtol=0)
            close(zs[l][:, :t+1], altered[1][l][:, :t+1], atol=0, rtol=0)
    errors = autograd_checks()
    del old, hs, zs, us, ids, em, altered

    # Adapt iter22's batch-update oracle to the new local derivative. Stored inputs
    # must give the same updates as the inherited interleaved sweep, including masked tails.
    batches = list(R.epoch(train[:6141], 0, min(len(train), 6141), net.V)); assert len(batches) == 3
    inter, batch = copy.deepcopy(net), copy.deepcopy(net); ig, bg = copy.deepcopy(gate), copy.deepcopy(gate)
    baseline, baseline_gate = copy.deepcopy(ref), copy.deepcopy(gate)
    inter.total_steps = batch.total_steps = baseline.total_steps = max(8, net.total_steps)
    for bx, by in batches:
        loss = inter.train(bx, by, ig); baseline.train(bx, by, baseline_gate)
        hs, zs, us, ids, em = batch.forward(bx); M = batch._prepare(bx)[0]
        final = sum(w*z for w, z in zip(batch.p['g'], zs)); ef = Q.N.error(final, by)
        grads = {'g': torch.stack([(ef*z).sum() for z in zs])}
        for l in range(3): grads.update(batch._local_grads(l, bx, by, hs[l], zs[l], us[l-1] if l else None, ids, em, M))
        batch.step += 1; batch.lr = S.schedule(batch.step, batch.total_steps)
        bg.train(bx, final.softmax(-1), by); batch._update(grads)
        assert abs(loss-F.cross_entropy(final.flatten(0, 1), by.flatten(), ignore_index=-1).item()) <= 1e-6
        S.same(inter, batch, ig, bg)
        for key in LOWER:
            close(inter.p[key], baseline.p[key], atol=1e-6, rtol=0)
            for a, b in zip(inter.state[key], baseline.state[key]): close(a, b, atol=1e-6, rtol=0)
    assert not torch.equal(inter.p['W1'], baseline.p['W1']) and not torch.equal(inter.p['W2'], baseline.p['W2'])
    del inter, batch, ig, bg, baseline, baseline_gate, ref, hs, zs, us, ids, em, grads

    # Retain iter22's observation of actual activation/update order and old-weight values.
    trace, tg = copy.deepcopy(net), copy.deepcopy(gate); bx, by = batches[0]; expected = trace.forward(bx)
    initial = {k: p.clone() for k, p in trace.p.items()}; events = []
    groups = [LOWER, *[[f'{k}{l}' for k in 'RcbW'] for l in (1, 2)]]
    activate, update, gate_train = trace._layer, trace._update, tg.train
    final = sum(w*z for w, z in zip(initial['g'], expected[1]))
    def recorded_layer(l, *args):
        events.append(f'activate {l}'); out = activate(l, *args)
        close(out[0], expected[0][l], atol=0, rtol=0); close(out[1], expected[1][l], atol=0, rtol=0)
        if l: close(out[2], expected[2][l-1], atol=0, rtol=0)
        return out
    def recorded_update(grads):
        if 'g' in grads:
            assert set(grads) == {'g'} and events[-1] == 'gate'; update(grads); events.append('g')
            assert not torch.equal(trace.p['g'], initial['g']); return
        l = next(l for l in range(3) if f'b{l}' in grads); assert set(grads) == set(groups[l])
        update(grads); events.append(f'update {l}')
        assert all(not torch.equal(trace.p[k], initial[k]) for k in groups[l])
        for keys in groups[l+1:]:
            for key in keys: close(trace.p[key], initial[key], atol=0, rtol=0)
        close(trace.p['g'], initial['g'], atol=0, rtol=0); assert I.stamp(tg) == gate_before
    def recorded_gate(gx, p, gy):
        assert events[-1] == 'update 2'; close(p, final.softmax(-1), atol=0, rtol=0)
        gate_train(gx, p, gy); events.append('gate'); assert not torch.equal(tg.A, gate.A)
    trace._layer = recorded_layer; trace._update = recorded_update; tg.train = recorded_gate; gate_before = I.stamp(tg)
    try: trace.train(bx, by, tg)
    finally: del trace._layer, trace._update, tg.train
    assert events == ['activate 0', 'update 0', 'activate 1', 'update 1', 'activate 2', 'update 2', 'gate', 'g']
    Q.plain((trace, tg)); del trace, tg

    # Reuse iter24's schedule checks, bypassing only its plain-tanh-specific iter22 checks.
    assert S.schedule(1, 8) == S.schedule(6, 8) == S.TOP and S.schedule(8, 8) == S.BOTTOM
    assert abs(S.schedule(7, 8)-5.5e-4) < 1e-15
    n = net.total_steps; hold = math.floor(S.HOLD*n); lrs = [S.schedule(s, n) for s in range(1, n+1)]
    assert all(v == S.TOP for v in lrs[:hold]) and lrs[hold] < S.TOP and lrs[-1] == S.BOTTOM
    assert all(a > b for a, b in zip(lrs[hold:], lrs[hold+1:]))
    a, b = copy.deepcopy(net), copy.deepcopy(net); ag, bg = copy.deepcopy(gate), copy.deepcopy(gate)
    a.total_steps = b.total_steps = max(n, 8)
    for bx, by in batches:
        assert abs(a.train(bx, by, ag)-S.S.Model.train(b, bx, by, bg)) <= 1e-6 and a.lr == b.lr == S.TOP
        S.same(a, b, ag, bg)
    del a, b, ag, bg
    bx, by = batches[0]; applied = {}
    for s, want in ((hold+1, lrs[hold]), (n, S.BOTTOM)):
        a, b, c = (copy.deepcopy(net) for _ in range(3)); ag, bg, cg = (copy.deepcopy(gate) for _ in range(3))
        a.step = b.step = c.step = s-1; b.lr = want; c.lr = S.TOP
        a.train(bx, by, ag); S.S.Model.train(b, bx, by, bg); S.S.Model.train(c, bx, by, cg)
        assert a.lr == want and a.step == s; S.same(a, b, ag, bg)
        for key in ('A', 'm', 'v'): close(getattr(ag, key), getattr(cg, key), atol=1e-6, rtol=0)
        if s == n:
            for key in ('b0', 'E', 'W2', 'g'): assert not torch.equal(a.p[key], c.p[key]), key
        applied[s] = want
        Q.plain((a, b, c, ag, bg, cg)); del a, b, c, ag, bg, cg
    assert I.stamp((net, gate)) == before, 'checks wrote model state'
    close(torch.get_rng_state(), cpu_rng, atol=0, rtol=0); close(torch.cuda.get_rng_state(), cuda_rng, atol=0, rtol=0)
    assert not torch.is_grad_enabled(); Q.plain((net, gate))
    result.update(initial_params_equal_iter24=True, initial_rng=True, layer0_equal_iter24=True, input_blocks=True,
                  layer_causality=True, upper_local_autograd=True, local_autograd_max_abs_error=errors, manual_training=True,
                  order_equals_batch=True, order_batches=3, order_events=events, schedule_values=True, hold_steps=hold,
                  decay_steps=n-hold, equals_unscheduled_while_held=True, scheduled_lr_applied=applied,
                  gate_lr_independent=True, state_untouched=True)
    return result


def validation_activation_stats(net, data):
    x, _ = B.eval_windows(data)
    sums = {l: torch.zeros(3, dtype=torch.float64, device='cuda') for l in (1, 2)}
    maxima = {l: torch.zeros((), device='cuda') for l in (1, 2)}; counts = {l: 0 for l in (1, 2)}
    for xb in x.split(16):
        hs, _, us, _, _ = net.forward(xb.cuda())
        for l in (1, 2):
            t = us[l-1][1]; h = hs[l]
            sums[l] += torch.stack([(t.abs() > .99).sum(dtype=torch.float64),
                                    derivative(t).sum(dtype=torch.float64), h.abs().sum(dtype=torch.float64)])
            maxima[l] = torch.maximum(maxima[l], h.abs().max()); counts[l] += h.numel()
    return {f'layer{l}': dict(tanh_saturated_fraction=(sums[l][0]/counts[l]).item(),
                             mean_derivative=(sums[l][1]/counts[l]).item(), mean_abs_h=(sums[l][2]/counts[l]).item(),
                             max_abs_h=maxima[l].item()) for l in (1, 2)}


def run(smoke=None, seed=0):
    if smoke is not None and smoke < 1: raise ValueError('--smoke must be positive')
    if seed < 0: raise ValueError('--seed must be non-negative')
    temporary = tempfile.TemporaryDirectory(prefix='iter28_') if smoke is not None else None
    try:
        name = 'iter28_softtanh.json' if seed == 0 else f'iter28_softtanh_seed{seed}.json'
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
            result['upper_activation_val'] = validation_activation_stats(net, tr[cut:])
            assert I.stamp((net, gate)) == before, 'evaluation wrote model state'
        finally: B.N_EVAL_WINDOWS = old_windows
        settings = dict(seed=seed, device='cuda', mode='copy', variant='softtanh', base='iter24', bs=16, T=128, passes=1,
                        train_end=cut, val_start=cut, training_partition_chars=len(train), smoke=smoke, eval_windows=16 if smoke is not None else old_windows,
                        sampling=f'non-overlapping target windows, shuffled with CPU generator seed {seed}', renaming=info, evaluation_frozen=True,
                        D=256, layers=3, embedding_width=32, lags=list(range(6)), decays=[.5,.8,.95], layer_input_width=1536, state_lags=[1,2],
                        lr=S.TOP, lr_final=S.BOTTOM, lr_hold_fraction=S.HOLD, lr_schedule='constant for s <= .75*S, then cosine to lr_final at s = S (1-indexed)',
                        lr_applies_to='all network parameters including g', optimizer='manual Adam', betas=[.9,.999], eps=1e-8, gradient_clip_norm=1, weight_decay=0,
                        activation=dict(layer0='tanh(a)', layer1='.95*tanh(a)+.05*a', layer2='.95*tanh(a)+.05*a',
                                        upper_derivative='.95*(1-tanh(a)**2)+.05; cached tanh(a)', saturation_threshold=.99, statistics_split='validation'),
                        update_order='per-layer during forward sweep; gate and g after', suffix_lengths=C.LENS, gate_lr=1e-2, gate_lr_schedule='constant')
        result.update(test_loss=test, renamed_loss=ren, renamed_gap=ren-test, val_loss=val, g=net.p['g'].tolist(), train_seconds=seconds,
                      chars=chars, steps=net.step, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, settings=settings,
                      iter24_reference=dict(val_loss=1.46686, test_loss=1.3320, renamed_loss=1.6298),
                      iter22_reference=dict(test_loss=1.3568, renamed_loss=1.6462),
                      iter23_reference=dict(test_loss=1.3530, renamed_loss=1.6430),
                      same_sampler_backprop_reference=dict(test_loss=1.340673, renamed_loss=1.686344))
        result['pass'] = dict(train=seconds <= 90, test=test <= 1.329, renamed=ren <= 1.667)
        with path.open('x', encoding='utf-8') as f: json.dump(result, f, indent=2, allow_nan=False)
        return result
    finally:
        if temporary: temporary.cleanup()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', type=int, metavar='N', help='train on at most N prefix chars, evaluate 16 windows, use a cleaned temporary result outside results/')
    p.add_argument('--seed', type=int, default=0, help='init and sampler seed; startup checks always use a seed-0 model')
    a = p.parse_args(); print(json.dumps(run(a.smoke, a.seed), indent=2), flush=True)
