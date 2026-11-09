"""iter18 copy with two lagged second-layer states in the final layer's local input."""
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
        w = self.p['W2']; self.p['W2'] = torch.cat([w, w.new_zeros(512, 256)], 0)
        self.state['W2'] = [torch.zeros_like(self.p['W2']), torch.zeros_like(self.p['W2'])]

    def forward(self, x):
        p = self.p; T = x.shape[1]; M = self.M[:, :T, :T]
        ids = [torch.cat([torch.full((len(x), j), self.V, device=x.device), x], 1)[:, :T] for j in range(6)]
        em = M @ p['E'][x][:, None]
        a = sum(p[f'A{j}'][ids[j]] for j in range(6)) + sum(em[:, k] @ p[f'P{k}'] for k in range(3)) + p['b0']
        hs, zs, us = [], [], []
        for l in range(self.L):
            if l:
                u = torch.cat([hs[-1], *(M @ hs[-1][:, None]).unbind(1)], -1)
                if l == 2:
                    h = hs[1]; lag1 = torch.zeros_like(h); lag2 = torch.zeros_like(h)
                    lag1[:, 1:] = h[:, :-1]; lag2[:, 2:] = h[:, :-2]
                    u = torch.cat([u, lag1, lag2], -1)
                us.append(u); a = u @ p[f'W{l}'] + p[f'b{l}']
            h = a.tanh(); hs.append(h); zs.append(h @ p[f'R{l}'] + p[f'c{l}'])
        return hs, zs, us, ids, em


def checks(net, gate, x, y):
    result = C.checks(net, gate); before = I.stamp((net, gate)); Q.plain((net, gate))
    assert not torch.is_grad_enabled() and net.step == gate.step == 0 and Model.train is Q.Model.train
    # Restore RNG after constructing the seed-matched reference; the extra rows draw nothing.
    cpu_rng = torch.get_rng_state(); cuda_rng = torch.cuda.get_rng_state()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(0); ref = Q.Model(net.V, False)
        torch.testing.assert_close(torch.get_rng_state(), cpu_rng, atol=0, rtol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, atol=0, rtol=0)
    for key, p in ref.p.items():
        torch.testing.assert_close(net.p[key][:1024] if key == 'W2' else net.p[key], p, atol=0, rtol=0)
    assert net.p['W2'].shape == (1536, 256) and not net.p['W2'][1024:].count_nonzero()
    for s in net.state['W2']: assert s.shape == (1536, 256) and not s.count_nonzero()
    hs, zs, us, _, _ = net.forward(x); old = ref.forward(x)
    for z, oz in zip(zs, old[1]): torch.testing.assert_close(z, oz, atol=1e-6, rtol=0)
    torch.testing.assert_close(us[0], old[2][0], atol=0, rtol=0)
    torch.testing.assert_close(us[1][..., :1024], old[2][1], atol=0, rtol=0)
    for lag, start in ((1, 1024), (2, 1280)):
        block = us[1][..., start:start+256]
        torch.testing.assert_close(block[:, lag:], hs[1][:, :-lag], atol=0, rtol=0)
        assert not block[:, :lag].count_nonzero()
    del ref, hs, zs, us, old
    # Freeze the lower layers: the local CE differentiates only this layer's W2.
    probe = copy.deepcopy(net); probe.p = {k: p.double() for k, p in probe.p.items()}
    probe.state = {k: [s.double() for s in ss] for k, ss in probe.state.items()}; probe.M = probe.M.double()
    y = y.clone(); y[0, -1] = -1
    hs, zs, us, _, _ = probe.forward(x); u = us[1]; w = probe.p['W2']; r = probe.p['R2']; c = probe.p['c2']; b = probe.p['b2']
    analytic = Q.N.readout_delta(u, Q.delta(hs[2], Q.N.error(zs[2], y), r))
    def ce(): return F.cross_entropy(((u @ w+b).tanh() @ r+c).flatten(0, 1), y.flatten(), ignore_index=-1)
    errors = {}; eps = 1e-5
    for name, rows in (('old', (0, 511, 1023)), ('lag1', (1024, 1151, 1279)), ('lag2', (1280, 1407, 1535))):
        worst = 0.
        for i, j in zip(rows, (0, 127, 255)):
            value = w[i, j].item(); w[i, j] = value+eps; plus = ce(); w[i, j] = value-eps; minus = ce(); w[i, j] = value
            numeric = (plus-minus)/(2*eps); torch.testing.assert_close(analytic[i, j], numeric, atol=1e-8, rtol=1e-5)
            worst = max(worst, abs((analytic[i, j]-numeric).item()))
        errors[name] = worst
    probe.train(x, y, 0); clipped = analytic/analytic.norm().clamp_min(1)
    torch.testing.assert_close(probe.state['W2'][0], .1*clipped, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(probe.state['W2'][1], .001*clipped.square(), atol=1e-12, rtol=1e-10)
    Q.plain(probe); assert I.stamp((net, gate)) == before, 'checks wrote model state'
    result.update(initial_weights=True, initial_rng=True, initial_logits=True, input_blocks=True, no_autograd=True,
                  layer2_finite_difference=True, layer2_fd_max_abs_error=errors, inherited_train_gradient=True)
    return result


def run(smoke=None):
    if smoke is not None and smoke < 1: raise ValueError('--smoke must be positive')
    temporary = tempfile.TemporaryDirectory(prefix='iter21_') if smoke is not None else None
    try:
        path = Path(temporary.name)/'iter21_lagged.json' if temporary else I.ROOT/'results/iter21_lagged.json'
        if path.exists(): raise FileExistsError(path)
        assert torch.cuda.is_available() and not torch.is_grad_enabled()
        torch.manual_seed(0); np.random.seed(0); torch.cuda.reset_peak_memory_stats()
        tr, te, V = B.load(); cut = int(len(tr)*.95); train = tr[:cut]
        if smoke is not None: train = train[:smoke]
        renamed, info = I.rename(te); old_windows = B.N_EVAL_WINDOWS
        net = Model(V); gate = C.Gate(V)
        x = torch.as_tensor(tr[:128].astype(np.int64), device='cuda')[None]
        y = torch.as_tensor(tr[1:129].astype(np.int64), device='cuda')[None]
        result = dict(checks=checks(net, gate, x, y)); chars = 0
        def predict(x, copy=True):
            zs = net.forward(x)[1]; logits = sum(w*z for w,z in zip(net.p['g'], zs))
            return gate.predict(x, logits.softmax(-1))[0] if copy else logits.log_softmax(-1)
        forward = net.forward
        def training_forward(x):
            out = forward(x)
            gate.train(x, sum(w*z for w,z in zip(net.p['g'], out[1])).softmax(-1), y)
            return out
        net.forward = training_forward
        torch.cuda.synchronize(); start = time.perf_counter()
        try:
            for x, y in R.epoch(train, 0, len(train), V):
                net.train(x, y, 0); chars += int((y >= 0).sum())
        finally: del net.forward
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
                xb = xb.cuda(); zs = net.forward(xb)[1]; p = sum(w*z for w,z in zip(net.p['g'], zs)).softmax(-1)
                weights += gate.predict(xb, p)[1].sum((0,1))
            result['mean_gate_weight_test'] = dict(zip(['net', *[f'L{L}' for L in C.LENS]], (weights/x.numel()).tolist()))
            assert I.stamp((net, gate)) == before, 'evaluation wrote model state'
        finally: B.N_EVAL_WINDOWS = old_windows
        settings = dict(seed=0, device='cuda', mode='copy', variant='lagged', bs=16, T=128, passes=1,
                        train_end=cut, val_start=cut, training_partition_chars=len(train), smoke=smoke, eval_windows=16 if smoke is not None else old_windows,
                        sampling='non-overlapping target windows, shuffled with CPU generator seed 0', renaming=info, evaluation_frozen=True,
                        novel_rule='Q.novel: aligned changed targets; maximal ASCII runs; earlier complete run required for repeat',
                        D=256, layers=3, embedding_width=32, lags=list(range(6)), decays=[.5,.8,.95], lr=1e-3, lr_schedule='constant',
                        optimizer='manual Adam', betas=[.9,.999], eps=1e-8, gradient_clip_norm=1, weight_decay=0,
                        local_rule='unchanged iter15.Model(V, False).train(x,y,0); no cross-layer derivative', g_init=1/3, g_constraint='none',
                        layer2_input_width=1536, layer2_state_lags=[1,2], layer2_new_rows_init='zero; original rows unchanged; no RNG draws',
                        suffix_lengths=C.LENS, memory='window-local', gate_shape=[5,15], gate_init='zero', gate_lr=1e-2,
                        gate_optimizer='manual Adam', gate_betas=[.9,.999], gate_eps=1e-8, gate_clip=None, gate_weight_decay=0,
                        gate_features=['bias','matched x4','log2(1+count)/7 x4','max proposal probability x4','net max probability','net entropy/log V'],
                        probability_floor=C.FLOOR, floor_scope='mixture log and target denominator; entropy log', gate_prediction='same forward, before network update',
                        copy_enabled=True, novel_copy_enabled=True, net_reference_test_loss=1.4922)
        result.update(test_loss=test, renamed_loss=ren, renamed_gap=ren-test, val_loss=val, g=net.p['g'].tolist(), train_seconds=seconds,
                      chars=chars, steps=net.step, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, settings=settings,
                      iter18_copy_reference=dict(test_loss=1.4112, renamed_loss=1.6975))
        result['pass'] = dict(train=seconds <= 90, test=test <= 1.329, renamed=ren <= 1.667)
        with path.open('x', encoding='utf-8') as f: json.dump(result, f, indent=2, allow_nan=False)
        return result
    finally:
        if temporary: temporary.cleanup()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', type=int, metavar='N', help='train on at most N prefix chars, evaluate 16 windows, use a cleaned temporary result outside results/')
    a = p.parse_args(); print(json.dumps(run(a.smoke), indent=2), flush=True)
