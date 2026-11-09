"""iter21 with state lags in both upper layers and interleaved layer-local updates."""
import argparse, copy, json, math, tempfile, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import iter15 as Q
import iter17 as R
import iter18 as C
I = Q.I; B = I.B
torch.set_grad_enabled(False)


class Model(Q.Model):
    def __init__(self, V):
        super().__init__(V, hybrid=False)
        for l in (1, 2):
            key = f'W{l}'; w = self.p[key]; self.p[key] = torch.cat([w, w.new_zeros(512, 256)], 0)
            self.state[key] = [torch.zeros_like(self.p[key]), torch.zeros_like(self.p[key])]

    def _prepare(self, x):
        T = x.shape[1]; M = self.M[:, :T, :T]
        ids = [torch.cat([torch.full((len(x), j), self.V, device=x.device), x], 1)[:, :T] for j in range(6)]
        return M, ids, M @ self.p['E'][x][:, None]

    def _layer(self, l, lower, M, ids, em):
        p = self.p; u = None
        if l:
            lag1 = torch.zeros_like(lower); lag2 = torch.zeros_like(lower)
            lag1[:, 1:] = lower[:, :-1]; lag2[:, 2:] = lower[:, :-2]
            u = torch.cat([lower, *(M @ lower[:, None]).unbind(1), lag1, lag2], -1)
            a = u @ p[f'W{l}'] + p[f'b{l}']
        else:
            a = sum(p[f'A{j}'][ids[j]] for j in range(6)) + sum(em[:, k] @ p[f'P{k}'] for k in range(3)) + p['b0']
        h = a.tanh()
        return h, h @ p[f'R{l}'] + p[f'c{l}'], u

    def forward(self, x):
        M, ids, em = self._prepare(x); hs, zs, us = [], [], []
        for l in range(self.L):
            h, z, u = self._layer(l, hs[-1] if l else None, M, ids, em)
            hs.append(h); zs.append(z)
            if l: us.append(u)
        return hs, zs, us, ids, em

    def _local_grads(self, l, x, y, h, z, u, ids, em, M):
        p = self.p; e = Q.N.error(z, y); d = Q.delta(h, e, p[f'R{l}'])
        grads = {f'R{l}': Q.N.readout_delta(h, e), f'c{l}': e.sum((0, 1)), f'b{l}': d.sum((0, 1))}
        if l: grads[f'W{l}'] = Q.N.readout_delta(u, d)
        else:
            for j in range(6): grads[f'A{j}'] = torch.zeros_like(p[f'A{j}']).index_add_(0, ids[j].flatten(), d.flatten(0, 1))
            de = torch.zeros_like(p['E'][x])
            for k in range(3):
                grads[f'P{k}'] = Q.N.readout_delta(em[:, k], d); de += M[k].T @ (d @ p[f'P{k}'].T)
            grads['E'] = torch.zeros_like(p['E']).index_add_(0, x.flatten(), de.flatten(0, 1))
        return grads

    def _update(self, grads):
        for key, grad in grads.items():
            assert torch.isfinite(grad).all(), key
            grad = grad/grad.norm().clamp_min(1); m, v = self.state[key]
            m.mul_(.9).add_(grad, alpha=.1); v.mul_(.999).addcmul_(grad, grad, value=.001)
            self.p[key].addcdiv_(m/(1-.9**self.step), (v/(1-.999**self.step)).sqrt().add_(1e-8), value=-self.lr)

    def train(self, x, y, gate):
        M, ids, em = self._prepare(x); lower = None; zs = []; self.step += 1
        for l in range(self.L):
            h, z, u = self._layer(l, lower, M, ids, em); zs.append(z)
            self._update(self._local_grads(l, x, y, h, z, u, ids, em, M))
            lower = h  # Stored activation from before this layer's update; never recompute it.
        final = sum(w*z for w, z in zip(self.p['g'], zs))
        gate.train(x, final.softmax(-1), y)
        ef = Q.N.error(final, y); self._update({'g': torch.stack([(ef*z).sum() for z in zs])})
        return F.cross_entropy(final.flatten(0, 1), y.flatten(), ignore_index=-1).item()


def checks(net, gate, x, y, train):
    before = I.stamp((net, gate)); result = C.checks(net, gate); Q.plain((net, gate))
    assert not torch.is_grad_enabled() and net.step == gate.step == 0
    # Seed-matched construction consumes exactly the same RNG as iter18; restore it afterwards.
    cpu_rng = torch.get_rng_state(); cuda_rng = torch.cuda.get_rng_state()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(0); ref = Q.Model(net.V, False)
        torch.testing.assert_close(torch.get_rng_state(), cpu_rng, atol=0, rtol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, atol=0, rtol=0)
    for key, p in ref.p.items():
        torch.testing.assert_close(net.p[key][:1024] if key in ('W1', 'W2') else net.p[key], p, atol=0, rtol=0)
    for l in (1, 2):
        w = net.p[f'W{l}']; assert w.shape == (1536, 256) and not w[1024:].count_nonzero()
        for s in net.state[f'W{l}']: assert s.shape == (1536, 256) and not s.count_nonzero()
    hs, zs, us, _, _ = net.forward(x); old = ref.forward(x)
    for z, oz in zip(zs, old[1]): torch.testing.assert_close(z, oz, atol=1e-6, rtol=0)
    for l in (1, 2):
        torch.testing.assert_close(us[l-1][..., :1024], old[2][l-1], atol=1e-6, rtol=0)
        for lag, start in ((1, 1024), (2, 1280)):
            block = us[l-1][..., start:start+256]
            torch.testing.assert_close(block[:, lag:], hs[l-1][:, :-lag], atol=0, rtol=0)
            assert not block[:, :lag].count_nonzero()
    del ref, hs, zs, us, old
    # Frozen inputs keep each finite difference strictly local to the selected W.
    probe = copy.deepcopy(net); pg = copy.deepcopy(gate)
    probe.p = {k: p.double() for k, p in probe.p.items()}; probe.M = probe.M.double()
    probe.state = {k: [s.double() for s in ss] for k, ss in probe.state.items()}
    for key in ('A', 'm', 'v'): setattr(pg, key, getattr(pg, key).double())
    fy = y.clone(); fy[0, -1] = -1
    hs, zs, us, ids, em = probe.forward(x); analytics = {}; errors = {}; eps = 1e-5
    for l in (1, 2):
        u = us[l-1]; w = probe.p[f'W{l}']; r = probe.p[f'R{l}']; c = probe.p[f'c{l}']; b = probe.p[f'b{l}']
        analytic = probe._local_grads(l, x, fy, hs[l], zs[l], u, ids, em, probe.M)[f'W{l}']; analytics[l] = analytic
        def ce(): return F.cross_entropy(((u @ w+b).tanh() @ r+c).flatten(0, 1), fy.flatten(), ignore_index=-1)
        errors[f'W{l}'] = {}
        for name, rows in (('old', (0, 511, 1023)), ('lag1', (1024, 1151, 1279)), ('lag2', (1280, 1407, 1535))):
            worst = 0.
            for i, j in zip(rows, (0, 127, 255)):
                value = w[i, j].item(); w[i, j] = value+eps; plus = ce(); w[i, j] = value-eps; minus = ce(); w[i, j] = value
                numeric = (plus-minus)/(2*eps); torch.testing.assert_close(analytic[i, j], numeric, atol=1e-8, rtol=1e-5)
                worst = max(worst, abs((analytic[i, j]-numeric).item()))
            errors[f'W{l}'][name] = worst
    probe.train(x, fy, pg)
    for l, analytic in analytics.items():
        clipped = analytic/analytic.norm().clamp_min(1)
        torch.testing.assert_close(probe.state[f'W{l}'][0], .1*clipped, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(probe.state[f'W{l}'][1], .001*clipped.square(), atol=1e-12, rtol=1e-10)
    Q.plain((probe, pg)); del probe, pg, hs, zs, us, ids, em, analytics
    # Three real batches, including a masked tail, against iter15's post-forward updates.
    batches = list(R.epoch(train[:6141], 0, min(len(train), 6141), net.V)); assert len(batches) == 3
    inter, ref = copy.deepcopy(net), copy.deepcopy(net); ig, rg = copy.deepcopy(gate), copy.deepcopy(gate)
    for bx, by in batches:
        loss = inter.train(bx, by, ig)
        out = ref.forward(bx); rg.train(bx, sum(w*z for w, z in zip(ref.p['g'], out[1])).softmax(-1), by)
        ref_loss = Q.Model.train(ref, bx, by, 0)
        assert abs(loss-ref_loss) <= 1e-6 and inter.step == ref.step == ig.step == rg.step
        for key in inter.p:
            torch.testing.assert_close(inter.p[key], ref.p[key], atol=1e-6, rtol=0)
            for a, b in zip(inter.state[key], ref.state[key]): torch.testing.assert_close(a, b, atol=1e-6, rtol=0)
        assert vars(ig).keys() == vars(rg).keys()
        for key, a in vars(ig).items():
            b = getattr(rg, key)
            if isinstance(a, torch.Tensor): torch.testing.assert_close(a, b, atol=1e-6, rtol=0)
            else: assert a == b
    del inter, ref, ig, rg, out
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
    Q.plain((trace, tg)); assert I.stamp((net, gate)) == before, 'checks wrote model state'
    torch.testing.assert_close(torch.get_rng_state(), cpu_rng, atol=0, rtol=0)
    torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, atol=0, rtol=0)
    result.update(initial_weights=True, initial_rng=True, initial_logits=True, input_blocks=True, no_autograd=True,
                  layer1_finite_difference=True, layer2_finite_difference=True, local_fd_max_abs_error=errors,
                  train_gradient=True, order_equals_batch=True, order_batches=3, order_interleaved=True, order_events=events, state_untouched=True)
    return result


def run(smoke=None):
    if smoke is not None and smoke < 1: raise ValueError('--smoke must be positive')
    temporary = tempfile.TemporaryDirectory(prefix='iter22_') if smoke is not None else None
    try:
        path = Path(temporary.name)/'iter22_lagged.json' if temporary else I.ROOT/'results/iter22_lagged.json'
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
        settings = dict(seed=0, device='cuda', mode='copy', variant='lagged', bs=16, T=128, passes=1,
                        train_end=cut, val_start=cut, training_partition_chars=len(train), smoke=smoke, eval_windows=16 if smoke is not None else old_windows,
                        sampling='non-overlapping target windows, shuffled with CPU generator seed 0', renaming=info, evaluation_frozen=True,
                        novel_rule='Q.novel: aligned changed targets; maximal ASCII runs; earlier complete run required for repeat',
                        D=256, layers=3, embedding_width=32, lags=list(range(6)), decays=[.5,.8,.95], lr=1e-3, lr_schedule='constant',
                        optimizer='manual Adam', betas=[.9,.999], eps=1e-8, gradient_clip_norm=1, weight_decay=0,
                        local_rule='own CE with old R.T; iter15 gradients; no cross-layer derivative', g_init=1/3, g_constraint='none',
                        layer1_input_width=1536, layer1_state_lags=[1,2], layer1_new_rows_init='zero; original rows unchanged; no RNG draws',
                        layer2_input_width=1536, layer2_state_lags=[1,2], layer2_new_rows_init='zero; original rows unchanged; no RNG draws',
                        layer_inputs='l=1,2: lower h, three EMAs, lag1, lag2; all from stored old-weight h',
                        update_order='per-layer during forward sweep; gate and g after',
                        suffix_lengths=C.LENS, memory='window-local', gate_shape=[5,15], gate_init='zero', gate_lr=1e-2, gate_lr_schedule='constant',
                        gate_optimizer='manual Adam', gate_betas=[.9,.999], gate_eps=1e-8, gate_clip=None, gate_weight_decay=0,
                        gate_features=['bias','matched x4','log2(1+count)/7 x4','max proposal probability x4','net max probability','net entropy/log V'],
                        probability_floor=C.FLOOR, floor_scope='mixture log and target denominator; entropy log', gate_prediction='old-weight zs and g; after layer updates',
                        copy_enabled=True, novel_copy_enabled=True, net_reference_test_loss=1.4922)
        result.update(test_loss=test, renamed_loss=ren, renamed_gap=ren-test, val_loss=val, g=net.p['g'].tolist(), train_seconds=seconds,
                      chars=chars, steps=net.step, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, settings=settings,
                      iter18_copy_reference=dict(test_loss=1.4112, renamed_loss=1.6975),
                      iter21_reference=dict(test_loss=1.3799, renamed_loss=1.6685),
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
