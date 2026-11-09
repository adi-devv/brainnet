"""iter22 with a two-ahead auxiliary local objective on layers 0 and 1."""
import argparse, copy, json, math, tempfile, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import iter22 as S
Q, R, C, I, B = S.Q, S.R, S.C, S.I, S.B
torch.set_grad_enabled(False)


def two_ahead(y):
    y2 = torch.cat([y[:, 1:], torch.full_like(y[:, :1], -1)], 1)
    return y2.masked_fill(y < 0, -1)


def aux_error(z, y):
    return Q.N.error(z, y) if (y >= 0).any() else torch.zeros_like(z)


class Model(S.Model):
    def __init__(self, V):
        super().__init__(V); self.aux_weights = (.75, .25)
        for l in (0, 1):
            for key, shape in ((f'Ra{l}', (256, V)), (f'ca{l}', (V,))):
                self.p[key] = self.p[f'R{l}'].new_zeros(shape)
                self.state[key] = [torch.zeros_like(self.p[key]), torch.zeros_like(self.p[key])]

    def _local_grads(self, l, x, y, h, z, u, ids, em, M):
        if l == 2: return super()._local_grads(l, x, y, h, z, u, ids, em, M)
        p = self.p; w, wa = self.aux_weights; e = Q.N.error(z, y)
        e2 = aux_error(h @ p[f'Ra{l}'] + p[f'ca{l}'], two_ahead(y))
        d = (w*(e @ p[f'R{l}'].T) + wa*(e2 @ p[f'Ra{l}'].T)) * (1-h.square())
        grads = {f'R{l}': w*Q.N.readout_delta(h, e), f'c{l}': w*e.sum((0, 1)), f'b{l}': d.sum((0, 1)),
                 f'Ra{l}': wa*Q.N.readout_delta(h, e2), f'ca{l}': wa*e2.sum((0, 1))}
        if l: grads[f'W{l}'] = Q.N.readout_delta(u, d)
        else:
            for j in range(6): grads[f'A{j}'] = torch.zeros_like(p[f'A{j}']).index_add_(0, ids[j].flatten(), d.flatten(0, 1))
            de = torch.zeros_like(p['E'][x])
            for k in range(3):
                grads[f'P{k}'] = Q.N.readout_delta(em[:, k], d); de += M[k].T @ (d @ p[f'P{k}'].T)
            grads['E'] = torch.zeros_like(p['E']).index_add_(0, x.flatten(), de.flatten(0, 1))
        return grads


def checks(net, gate, x, y, train):
    before = I.stamp((net, gate)); result = C.checks(net, gate); Q.plain((net, gate))
    assert not torch.is_grad_enabled() and net.step == gate.step == 0
    assert net.aux_weights == (.75, .25)
    for name in ('forward', '_layer', '_update', 'train'): assert getattr(Model, name) is getattr(S.Model, name)
    cpu_rng = torch.get_rng_state(); cuda_rng = torch.cuda.get_rng_state()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(0); ref = S.Model(net.V)
        torch.testing.assert_close(torch.get_rng_state(), cpu_rng, atol=0, rtol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, atol=0, rtol=0)
    aux_keys = {f'{key}{l}' for l in (0, 1) for key in ('Ra', 'ca')}
    assert set(net.p)-set(ref.p) == set(net.state)-set(ref.state) == aux_keys
    for key, p in ref.p.items():
        torch.testing.assert_close(net.p[key], p, atol=0, rtol=0)
        for a, b in zip(net.state[key], ref.state[key]): torch.testing.assert_close(a, b, atol=0, rtol=0)
    for key in aux_keys:
        shape = (256, net.V) if key.startswith('Ra') else (net.V,)
        for p in [net.p[key], *net.state[key]]: assert p.shape == shape and not p.count_nonzero()
    assert x.shape[1] == 128
    for z, oz in zip(net.forward(x)[1], ref.forward(x)[1]): torch.testing.assert_close(z, oz, atol=1e-6, rtol=0)
    # Real sampler batches include a masked final window; artificial tails also test holes.
    batches = list(R.epoch(train[:6141], 0, min(len(train), 6141), net.V))
    assert len(batches) == 3 and any((by < 0).any() for _, by in batches)
    tails = y.expand(4, -1).clone()
    for row, end in enumerate((0, 1, 63, 127)): tails[row, end:] = -1
    tails[2, 7] = -1
    for by in [y, *[by for _, by in batches], tails, torch.full_like(y, -1)]:
        y2 = two_ahead(by); expected = torch.full_like(by, -1)
        expected[:, :-1] = torch.where(by[:, :-1] >= 0, by[:, 1:], -1)
        torch.testing.assert_close(y2, expected, atol=0, rtol=0)
        assert (y2[:, -1] == -1).all() and (y2[by < 0] == -1).all()
    for by in (torch.full_like(y, -1), tails[1:2]):
        y2 = two_ahead(by); assert (y2 < 0).all()
        e2 = aux_error(net.forward(x)[1][0], y2)
        assert torch.isfinite(e2).all() and not e2.count_nonzero()
    # Nonzero auxiliary heads, local RNG, and numerical gradients at the actual train call site.
    probe = copy.deepcopy(net); pg = copy.deepcopy(gate)
    probe.p = {k: p.double() for k, p in probe.p.items()}; probe.M = probe.M.double()
    probe.state = {k: [s.double() for s in ss] for k, ss in probe.state.items()}
    for key in ('A', 'm', 'v'): setattr(pg, key, getattr(pg, key).double())
    rng = torch.Generator(device=x.device).manual_seed(23)
    for key in sorted(aux_keys):
        p = probe.p[key]; p.copy_(torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=rng)*.05)
        assert p.count_nonzero() == p.numel()
    fy = y.clone(); fy[:, -19:] = -1; fy[:, 7] = -1
    errors = {}; analytics = {}; eps = 1e-5; local_grads = probe._local_grads
    def checked_grads(l, bx, by, h, z, u, ids, em, M):
        grads = local_grads(l, bx, by, h, z, u, ids, em, M)
        if l == 2:
            old = S.Model._local_grads(probe, l, bx, by, h, z, u, ids, em, M)
            assert grads.keys() == old.keys()
            for key in grads: torch.testing.assert_close(grads[key], old[key], atol=0, rtol=0)
            return grads
        p = probe.p; y2 = two_ahead(by)
        def ce():
            if l:
                hidden = (u @ p[f'W{l}'] + p[f'b{l}']).tanh()  # Freeze every lower-layer input.
            else:
                mm, ii, ee = probe._prepare(bx); hidden = probe._layer(0, None, mm, ii, ee)[0]
            z1 = hidden @ p[f'R{l}'] + p[f'c{l}']; z2 = hidden @ p[f'Ra{l}'] + p[f'ca{l}']
            return (probe.aux_weights[0]*F.cross_entropy(z1.flatten(0, 1), by.flatten(), ignore_index=-1)
                    + probe.aux_weights[1]*F.cross_entropy(z2.flatten(0, 1), y2.flatten(), ignore_index=-1))
        entries = {f'R{l}': [(0, 0), (127, net.V//2), (255, net.V-1)], f'c{l}': [(0,), (net.V//2,), (net.V-1,)],
                   f'Ra{l}': [(0, 0), (127, net.V//2), (255, net.V-1)], f'ca{l}': [(0,), (net.V//2,), (net.V-1,)]}
        if l:
            entries['W1'] = [(i, j) for rows in ((0, 511, 1023), (1024, 1151, 1279), (1280, 1407, 1535)) for i, j in zip(rows, (0, 127, 255))]
        else:
            entries.update(A0=[(int(bx[0, t]), j) for t, j in ((0, 0), (15, 127), (31, 255))],
                           P0=[(0, 0), (15, 127), (31, 255)], E=[(int(bx[0, t]), j) for t, j in ((0, 0), (15, 15), (31, 31))],
                           b0=[(0,), (127,), (255,)])
        for key, indices in entries.items():
            worst = 0.
            for ix in indices:
                value = p[key][ix].item()
                try:
                    p[key][ix] = value+eps; plus = ce(); p[key][ix] = value-eps; minus = ce()
                finally: p[key][ix] = value
                numeric = (plus-minus)/(2*eps); analytic = grads[key][ix]
                torch.testing.assert_close(analytic, numeric, atol=1e-8, rtol=1e-5)
                worst = max(worst, abs((analytic-numeric).item()))
            errors[key] = worst
        for key in ('W1', 'Ra0', 'Ra1'):
            if key in grads: analytics[key] = grads[key].clone()
        return grads
    probe._local_grads = checked_grads
    try: probe.train(x, fy, pg)
    finally: del probe._local_grads
    assert set(errors) == {'W1', 'A0', 'P0', 'E', 'b0', 'R0', 'c0', 'Ra0', 'ca0', 'R1', 'c1', 'Ra1', 'ca1'}
    for key, analytic in analytics.items():
        clipped = analytic/analytic.norm().clamp_min(1)
        torch.testing.assert_close(probe.state[key][0], .1*clipped, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(probe.state[key][1], .001*clipped.square(), atol=1e-12, rtol=1e-10)
    Q.plain((probe, pg)); del probe, pg, analytics
    # Setting the objective weights to (1, 0) recovers iter22, including its gate and Adam.
    inter, old = copy.deepcopy(net), copy.deepcopy(ref); ig, rg = copy.deepcopy(gate), copy.deepcopy(gate)
    inter.aux_weights = (1., 0.)
    for bx, by in batches:
        loss = inter.train(bx, by, ig); ref_loss = old.train(bx, by, rg)
        assert abs(loss-ref_loss) <= 1e-6 and inter.step == old.step == ig.step == rg.step
        for key in old.p:
            torch.testing.assert_close(inter.p[key], old.p[key], atol=1e-6, rtol=0)
            for a, b in zip(inter.state[key], old.state[key]): torch.testing.assert_close(a, b, atol=1e-6, rtol=0)
        for key in aux_keys:
            for p in [inter.p[key], *inter.state[key]]: assert not p.count_nonzero()
        assert vars(ig).keys() == vars(rg).keys()
        for key, a in vars(ig).items():
            b = getattr(rg, key)
            if isinstance(a, torch.Tensor): torch.testing.assert_close(a, b, atol=1e-6, rtol=0)
            else: assert a == b
    del inter, old, ig, rg, ref
    # Changed auxiliary logits cannot enter stored main logits, the gate input, or g's gradient.
    base, changed = copy.deepcopy(net), copy.deepcopy(net)
    for key in sorted(aux_keys):
        p = changed.p[key]; p.copy_(torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=rng)*.1)
    for a, b in zip(base.forward(x)[1], changed.forward(x)[1]): torch.testing.assert_close(a, b, atol=0, rtol=0)
    captures = []
    for model in (base, changed):
        cg = copy.deepcopy(gate); capture = {}; update, activate, gate_train = model._update, model._layer, cg.train
        def recorded_layer(l, *args):
            out = activate(l, *args); capture.setdefault('zs', []).append(out[1].clone()); return out
        def recorded_update(grads):
            if 'g' in grads: capture['g'] = grads['g'].clone()
            update(grads)
        def recorded_gate(gx, p, gy):
            capture['final'] = sum(w*z for w, z in zip(model.p['g'], capture['zs']))
            capture['gate_input'] = p.clone(); gate_train(gx, p, gy)
        model._layer = recorded_layer; model._update = recorded_update; cg.train = recorded_gate
        try: capture['loss'] = model.train(x, y, cg)
        finally: del model._layer, model._update, cg.train
        captures.append(capture)
    for key in ('final', 'gate_input', 'g'): torch.testing.assert_close(captures[0][key], captures[1][key], atol=0, rtol=0)
    for a, b in zip(captures[0]['zs'], captures[1]['zs']): torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert captures[0]['loss'] == captures[1]['loss']
    del base, changed, cg, captures
    # Observe the inherited interleaving, including both auxiliary heads in their layer's update.
    trace, tg = copy.deepcopy(net), copy.deepcopy(gate); bx, by = batches[0]; expected = trace.forward(bx)
    initial = {k: p.clone() for k, p in trace.p.items()}; events = []
    groups = [[f'R{l}', f'c{l}', f'b{l}'] + ([f'W{l}'] if l else [*[f'A{j}' for j in range(6)], 'E', *[f'P{k}' for k in range(3)]])
              + ([f'Ra{l}', f'ca{l}'] if l < 2 else []) for l in range(3)]
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
        final = sum(w*z for w, z in zip(initial['g'], expected[1]))
        torch.testing.assert_close(p, final.softmax(-1), atol=0, rtol=0)
        gate_train(gx, p, gy); events.append('gate')
    trace._layer = recorded_layer; trace._update = recorded_update; tg.train = recorded_gate; gate_before = I.stamp(tg)
    try: trace.train(bx, by, tg)
    finally: del trace._layer, trace._update, tg.train
    assert events == ['activate 0', 'update 0', 'activate 1', 'update 1', 'activate 2', 'update 2', 'gate', 'g']
    Q.plain((trace, tg)); assert I.stamp((net, gate)) == before, 'checks wrote model state'
    torch.testing.assert_close(torch.get_rng_state(), cpu_rng, atol=0, rtol=0)
    torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, atol=0, rtol=0)
    result.update(initial_weights=True, initial_rng=True, initial_logits=True, aux_init=True, two_ahead_targets=True, empty_aux_error=True,
                  no_autograd=True, combined_finite_difference=True, local_fd_max_abs_error=errors, train_gradient=True,
                  layer2_unchanged=True, reduces_to_iter22=True, reduction_batches=3, aux_logits_isolated=True,
                  order_interleaved=True, order_events=events, state_untouched=True)
    return result


def run(smoke=None):
    if smoke is not None and smoke < 1: raise ValueError('--smoke must be positive')
    temporary = tempfile.TemporaryDirectory(prefix='iter23_') if smoke is not None else None
    try:
        path = Path(temporary.name)/'iter23_aux.json' if temporary else I.ROOT/'results/iter23_aux.json'
        if path.exists(): raise FileExistsError(path)
        assert torch.cuda.is_available() and not torch.is_grad_enabled()
        torch.manual_seed(0); np.random.seed(0); torch.cuda.reset_peak_memory_stats()
        tr, te, V = B.load(); cut = int(len(tr)*.95); train = tr[:cut]
        if smoke is not None: train = train[:smoke]
        renamed, info = I.rename(te); old_windows = B.N_EVAL_WINDOWS
        net = Model(V); gate = C.Gate(V)
        x = torch.as_tensor(tr[:128].astype(np.int64), device='cuda')[None]
        y = torch.as_tensor(tr[1:129].astype(np.int64), device='cuda')[None]
        result = dict(checks=checks(net, gate, x, y, tr[:cut])); chars = 0
        def predict(x, copy=True):
            zs = net.forward(x)[1]; logits = sum(w*z for w, z in zip(net.p['g'], zs))
            return gate.predict(x, logits.softmax(-1))[0] if copy else logits.log_softmax(-1)
        torch.cuda.synchronize(); start = time.perf_counter()
        for x, y in R.epoch(train, 0, len(train), V):
            net.train(x, y, gate); chars += int((y >= 0).sum())
        torch.cuda.synchronize(); seconds = time.perf_counter()-start
        assert chars == len(train) and net.step == gate.step == math.ceil(len(train)/(16*128))
        Q.plain((net, gate)); assert not torch.is_grad_enabled()
        try:
            if smoke is not None: B.N_EVAL_WINDOWS = 16
            before = I.stamp((net, gate))
            test = I.evaluate(predict, te); ren = I.evaluate(predict, renamed); val = I.evaluate(predict, tr[cut:])
            result.update(Q.novel(predict, te, renamed, json.loads((I.ROOT/'data/meta.json').read_text())['vocab']))
            result['test_loss_nocopy'] = I.evaluate(lambda x: predict(x, False), te)
            result['renamed_loss_nocopy'] = I.evaluate(lambda x: predict(x, False), renamed)
            result['test_loss_nocopy_difference'] = result['test_loss_nocopy']-1.4922
            result['per_layer_test_loss'] = [I.evaluate(lambda x, l=l: net.logits(x, l), te) for l in range(net.L)]
            weights = torch.zeros(5, device='cuda'); x, _ = B.eval_windows(te)
            for xb in x.split(16):
                xb = xb.cuda(); zs = net.forward(xb)[1]; p = sum(w*z for w, z in zip(net.p['g'], zs)).softmax(-1)
                weights += gate.predict(xb, p)[1].sum((0, 1))
            result['mean_gate_weight_test'] = dict(zip(['net', *[f'L{L}' for L in C.LENS]], (weights/x.numel()).tolist()))
            assert I.stamp((net, gate)) == before, 'evaluation wrote model state'
        finally: B.N_EVAL_WINDOWS = old_windows
        settings = dict(seed=0, device='cuda', mode='copy', variant='aux', bs=16, T=128, passes=1,
                        train_end=cut, val_start=cut, training_partition_chars=len(train), smoke=smoke, eval_windows=16 if smoke is not None else old_windows,
                        sampling='non-overlapping target windows, shuffled with CPU generator seed 0', renaming=info, evaluation_frozen=True,
                        novel_rule='Q.novel: aligned changed targets; maximal ASCII runs; earlier complete run required for repeat',
                        D=256, layers=3, embedding_width=32, lags=list(range(6)), decays=[.5,.8,.95], lr=1e-3, lr_schedule='constant',
                        optimizer='manual Adam', betas=[.9,.999], eps=1e-8, gradient_clip_norm=1, weight_decay=0,
                        local_rule='layers 0,1: weighted next and two-ahead CE with old readouts; layer 2: iter22; no cross-layer derivative', g_init=1/3, g_constraint='none',
                        layer1_input_width=1536, layer1_state_lags=[1,2], layer1_new_rows_init='zero; original rows unchanged; no RNG draws',
                        layer2_input_width=1536, layer2_state_lags=[1,2], layer2_new_rows_init='zero; original rows unchanged; no RNG draws',
                        layer_inputs='l=1,2: lower h, three EMAs, lag1, lag2; all from stored old-weight h',
                        update_order='per-layer during forward sweep; gate and g after',
                        aux_weights=list(net.aux_weights), aux_layers=[0, 1], aux_target='two ahead, y[:, 1:]; final position masked',
                        aux_init='zero; no RNG draws', aux_in_g_or_gate=False,
                        suffix_lengths=C.LENS, memory='window-local', gate_shape=[5,15], gate_init='zero', gate_lr=1e-2, gate_lr_schedule='constant',
                        gate_optimizer='manual Adam', gate_betas=[.9,.999], gate_eps=1e-8, gate_clip=None, gate_weight_decay=0,
                        gate_features=['bias','matched x4','log2(1+count)/7 x4','max proposal probability x4','net max probability','net entropy/log V'],
                        probability_floor=C.FLOOR, floor_scope='mixture log and target denominator; entropy log', gate_prediction='old-weight zs and g; after layer updates',
                        copy_enabled=True, novel_copy_enabled=True, net_reference_test_loss=1.4922)
        result.update(test_loss=test, renamed_loss=ren, renamed_gap=ren-test, val_loss=val, g=net.p['g'].tolist(), train_seconds=seconds,
                      chars=chars, steps=net.step, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, settings=settings,
                      iter18_copy_reference=dict(test_loss=1.4112, renamed_loss=1.6975),
                      iter21_reference=dict(test_loss=1.3799, renamed_loss=1.6685),
                      iter22_reference=dict(test_loss=1.3568, renamed_loss=1.6462),
                      same_sampler_backprop_reference=dict(test_loss=1.340673, renamed_loss=1.686344))
        result['pass'] = dict(train=seconds <= 90, test=test <= 1.329, renamed=ren <= 1.667)
        with path.open('x', encoding='utf-8') as f: json.dump(result, f, indent=2, allow_nan=False)
        return result
    finally:
        if temporary: temporary.cleanup()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', type=int, metavar='N', help='train on at most N prefix chars, evaluate 16 windows, use a cleaned temporary result outside results/')
    a = p.parse_args(); print(json.dumps(run(a.smoke), indent=2), flush=True)
