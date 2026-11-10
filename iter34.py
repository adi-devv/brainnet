"""Training-only nonlinear lower-layer heads; iter27 inference and Adam.

Normal results refuse overwrites; smoke JSON is temporary (iter30 convention).
--save-ckpt PATH saves a resumable model, gate, Adam and sampler checkpoint.
"""
import argparse
import copy
import hashlib
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import iter30 as P
import iter33 as D

W, S, Q, R, C, I, B = P.W, P.S, P.Q, P.R, P.C, P.I, P.B
torch.set_grad_enabled(False)
close = torch.testing.assert_close
MODES = ('auxhead', 'auxhead_bp')
AUX_SEED_XOR = 0x34A0C


def aux_keys(l):
    return tuple(f'aux{l}_{name}' for name in ('W', 'b', 'R', 'c'))


def backbone_keys(l):
    return D.layer_keys(l)-{f'R{l}', f'c{l}'}


def aux_logits(p, l, h):
    w, b, r, c = (p[k] for k in aux_keys(l))
    return F.gelu(h @ w+b, approximate='none') @ r+c


def backprop_grads(net, x, y):
    keys = [k for k in net.p if k != 'g']
    with torch.enable_grad():
        view = P.leaf_view(net, keys)
        hs, zs = view.forward(x)[:2]
        losses = [P.ce(aux_logits(view.p, l, hs[l]), y) for l in (0, 1)]
        sum([*(P.ce(z, y) for z in zs), *losses]).backward()
        assert all(view.p[k].grad is not None for k in keys)
        grads = {k: view.p[k].grad.detach() for k in keys}
        aux = torch.stack(losses).detach()
    return grads, [z.detach() for z in zs], aux


class Model(P.Model):
    def __init__(self, V, total_steps, mode='auxhead', hid=64):
        if mode not in MODES:
            raise ValueError(mode)
        if hid < 1:
            raise ValueError('--hid must be positive')
        super().__init__(V, total_steps, mode='local')
        self.mode, self.hid = mode, hid
        self.aux_step, self.last_aux = 0, None
        gen = torch.Generator(device=self.p['E'].device)
        gen.manual_seed(torch.initial_seed() ^ AUX_SEED_XOR)
        for l in (0, 1):
            w, b, r, c = aux_keys(l)
            # Fan-in normal first map, zero biases, zero output map. The first
            # backbone gradient is consequently zero, then learns after step 1.
            self.p[w] = torch.randn(256, hid, device=self.p['E'].device,
                                    generator=gen)/math.sqrt(256)
            self.p[b] = self.p[w].new_zeros(hid)
            self.p[r] = self.p[w].new_zeros(hid, V)
            self.p[c] = self.p[w].new_zeros(V)
            for key in aux_keys(l):
                self.state[key] = [torch.zeros_like(self.p[key]), torch.zeros_like(self.p[key])]

    def _layer(self, l, lower, M, ids, em):
        if l and self.mode == 'auxhead':
            lower = lower.detach()
        h, z, u = super()._layer(l, lower, M, ids, em)
        if l < 2 and self.mode == 'auxhead' and torch.is_grad_enabled():
            z = h.detach() @ self.p[f'R{l}']+self.p[f'c{l}']
        return h, z, u

    def _aux_grads(self, l, h, y):
        """Exact GELU chain rule: returns head gradients and d CE / d h."""
        w, b, r, c = aux_keys(l)
        a = h @ self.p[w]+self.p[b]
        v = F.gelu(a, approximate='none')
        z = v @ self.p[r]+self.p[c]
        e = Q.N.error(z, y)
        derivative = .5*(1+torch.erf(a/math.sqrt(2)))+a*torch.exp(-.5*a.square())/math.sqrt(2*math.pi)
        d = (e @ self.p[r].T)*derivative
        grads = {w: Q.N.readout_delta(h, d), b: d.sum((0, 1)),
                 r: Q.N.readout_delta(v, e), c: e.sum((0, 1))}
        return grads, d @ self.p[w].T, P.ce(z, y)

    def _backbone_grads(self, l, x, h, dh, u, ids, em, M):
        d = dh*(1-h.square())
        grads = {f'b{l}': d.sum((0, 1))}
        if l:
            grads[f'W{l}'] = Q.N.readout_delta(u, d)
        else:
            for j in range(6):
                grads[f'A{j}'] = torch.zeros_like(self.p[f'A{j}']).index_add_(
                    0, ids[j].flatten(), d.flatten(0, 1))
            de = torch.zeros_like(self.p['E'][x])
            for k in range(3):
                grads[f'P{k}'] = Q.N.readout_delta(em[:, k], d)
                de += M[k].T @ (d @ self.p[f'P{k}'].T)
            grads['E'] = torch.zeros_like(self.p['E']).index_add_(
                0, x.flatten(), de.flatten(0, 1))
        return grads

    def _candidate_grads(self, l, x, y, h, z, u, ids, em, M):
        if l == 2:
            return self._local_grads(l, x, y, h, z, u, ids, em, M), None
        grads, dh, loss = self._aux_grads(l, h, y)
        grads.update(self._backbone_grads(l, x, h, dh, u, ids, em, M))
        # Readout learns its CE on detached features; it contributes no dh.
        e = Q.N.error(z, y)
        grads.update({f'R{l}': Q.N.readout_delta(h, e), f'c{l}': e.sum((0, 1))})
        return grads, loss

    def _update(self, grads):
        base = {k: v for k, v in grads.items() if not k.startswith('aux')}
        aux = {k: v for k, v in grads.items() if k.startswith('aux')}
        if base:
            super()._update(base)
        if aux:
            step = self.step
            try:
                self.step = self.aux_step  # Fresh heads get fresh Adam bias correction.
                super()._update(aux)
            finally:
                self.step = step

    def warmup_heads(self, x, y):
        """Consume an extra batch without moving the backbone schedule or gate."""
        self.aux_step += 1
        self.lr = S.schedule(self.step+1, self.total_steps)
        M, ids, em = self._prepare(x)
        lower, hs = None, []
        for l in (0, 1):
            lower = self._layer(l, lower, M, ids, em)[0]
            hs.append(lower)
        for l in (0, 1):
            self._update(self._aux_grads(l, hs[l].detach(), y)[0])

    def train(self, x, y, gate):
        self.lr = S.schedule(self.step+1, self.total_steps)
        self.step += 1
        self.aux_step += 1
        if self.mode == 'auxhead_bp':
            grads, zs, self.last_aux = backprop_grads(self, x, y)
            final, grads['g'] = self._mixed(zs, y)
            gate.train(x, final.softmax(-1), y)
            self._update(grads)
        else:
            M, ids, em = self._prepare(x)
            lower, zs, losses = None, [], []
            for l in range(self.L):
                h, z, u = self._layer(l, lower, M, ids, em)
                grads, loss = self._candidate_grads(l, x, y, h, z, u, ids, em, M)
                zs.append(z)
                if loss is not None:
                    losses.append(loss)
                self._update(grads)
                lower = h  # iter27's stored pre-update activation.
            self.last_aux = torch.stack(losses)
            final, gg = self._mixed(zs, y)
            gate.train(x, final.softmax(-1), y)
            self._update({'g': gg})
        return P.ce(final, y).item()


class Sampler:
    """Resumable iter17.epoch, including its CPU permutation generator state.

    The permutation is drawn once up front, so RNG state alone cannot resume;
    starts, offset and remaining valid targets are saved too.
    """
    def __init__(self, train, seed, V, state=None):
        self.data = torch.as_tensor(train.astype(np.int64), device='cuda')
        self.V = V
        self.pos = torch.arange(128, device='cuda')
        self.generator = torch.Generator().manual_seed(seed)
        if state is None:
            self.starts = torch.randperm(math.ceil(len(train)/128), generator=self.generator)*128
            self.offset, self.left = 0, len(train)
        else:
            self.generator.set_state(state['rng'].cpu())
            self.starts = state['starts'].cpu().clone()
            self.offset, self.left = state['offset'], state['left']

    def state_dict(self):
        return dict(rng=self.generator.get_state(), starts=self.starts.clone(),
                    offset=self.offset, left=self.left)

    def __iter__(self):
        return self

    def __next__(self):
        if self.left <= 0 or self.offset >= len(self.starts):
            raise StopIteration
        st = self.starts[self.offset:self.offset+min(16, math.ceil(self.left/128))]
        ix = st.cuda()[:, None]+self.pos
        valid = ((ix < len(self.data)) &
                 (torch.arange(ix.numel(), device='cuda').reshape_as(ix) < self.left))
        x = torch.where(ix > 0, self.data[(ix-1).clamp(0, len(self.data)-1)], self.V)
        y = torch.where(valid, self.data[ix.clamp_max(len(self.data)-1)], -1)
        self.offset += 16
        self.left -= int(valid.sum())
        return x, y


def save_checkpoint(path, net, gate, sampler, train, seed):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(format='iter34_v1', model=vars(net), gate=vars(gate),
                   sampler=sampler.state_dict(), seed=seed, train_chars=len(train),
                   train_sha256=hashlib.sha256(train.tobytes()).hexdigest(),
                   torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
    with path.open('xb') as f:
        torch.save(payload, f)


def reference_aux(p, l, h):
    w, b, r, c = (p[k] for k in aux_keys(l))
    a = h @ w+b
    return (.5*a*(1+torch.erf(a/math.sqrt(2)))) @ r+c


def checks(net, gate, x, y, seed):
    before = I.stamp((net, gate))
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    baseline = copy.copy(net)
    baseline.__class__, baseline.mode = P.Model, 'backprop' if net.mode == 'auxhead_bp' else 'local'
    baseline.p = {k: v for k, v in net.p.items() if not k.startswith('aux')}
    baseline.state = {k: net.state[k] for k in baseline.p}
    result = P.checks(baseline, gate, x, y, seed)
    for l in (0, 1):
        assert not net.p[aux_keys(l)[2]].count_nonzero()
        assert not net.p[aux_keys(l)[3]].count_nonzero()
    fx, fy = x[:, :4], y[:, :4].clone()
    fy[0, 1] = -1
    # Verify resumability, masks and sentinel against the actual iter17 sampler.
    tiny = np.arange(6141, dtype=np.int64) % net.V
    sampler = Sampler(tiny, seed, net.V)
    expected_batches = list(R.epoch(tiny, seed, len(tiny), net.V))
    for index, expected in enumerate(expected_batches):
        if index == 1:
            sampler = Sampler(tiny, seed, net.V, sampler.state_dict())
        for actual, want in zip(next(sampler), expected):
            close(actual, want, atol=0, rtol=0)
    assert sampler.left <= 0 or sampler.offset >= len(sampler.starts)
    assert next(iter(sampler), None) is None
    probe = P.double_probe(net)
    probe.mode = 'auxhead'
    # Activate output weights so the isolation and chain-rule tests are nonzero.
    for l in (0, 1):
        r = probe.p[aux_keys(l)[2]]
        r.copy_(torch.arange(r.numel(), device=r.device, dtype=r.dtype).reshape_as(r).sin()*.03)
    hs, zs, us, ids, em = probe.forward(fx)
    M = probe._prepare(fx)[0]
    errors = {}
    analytic = {}
    for l in range(3):
        grad, _ = probe._candidate_grads(l, fx, fy, hs[l], zs[l],
                                       us[l-1] if l else None, ids, em, M)
        analytic.update(grad)
    with torch.enable_grad():
        view = P.leaf_view(probe, list(probe.p))
        hh, zz = view.forward(fx)[:2]
        for l in range(3):
            # Lower inference readout CE must touch only its R,c. Its aux CE
            # must touch only this layer's backbone and its own head.
            losses = [(P.ce(zz[l], fy), {f'R{l}', f'c{l}'} if l < 2 else D.layer_keys(l))]
            if l < 2:
                losses.append((P.ce(reference_aux(view.p, l, hh[l]), fy),
                               backbone_keys(l) | set(aux_keys(l))))
            for loss, allowed in losses:
                grads = torch.autograd.grad(loss, list(view.p.values()), retain_graph=True, allow_unused=True)
                for key, grad in zip(view.p, grads):
                    if key not in allowed:
                        assert grad is None, ('gradient leak', l, key)
                    else:
                        assert grad is not None
                        close(analytic[key], grad, atol=1e-12, rtol=1e-10)
                        errors[key] = (analytic[key]-grad).abs().max().item()
                        assert grad.norm() > 0, ('vacuous active-head check', key)
    # Zero-output initialization means exactly zero initial backbone gradient.
    zero = P.double_probe(net)
    for l in (0, 1):
        assert not zero._aux_grads(l, hs[l], fy)[1].count_nonzero()
    bp = P.double_probe(probe)
    bp.mode = 'auxhead_bp'
    actual_bp, _, _ = backprop_grads(bp, fx, fy)
    keys = [k for k in bp.p if k != 'g']
    with torch.enable_grad():
        view = P.leaf_view(bp, keys)
        mm, ii, ee = view._prepare(fx)
        lower, logits, hidden = None, [], []
        for l in range(3):
            # Bypass the production override for an independent BP reference.
            lower, z, _ = W.Model._layer(view, l, lower, mm, ii, ee)
            hidden.append(lower)
            logits.append(z)
        loss = sum(P.ce(z, fy) for z in logits)
        loss += sum(P.ce(reference_aux(view.p, l, hidden[l]), fy) for l in (0, 1))
        expected = torch.autograd.grad(loss, [view.p[k] for k in keys])
    for key, grad in zip(keys, expected):
        close(actual_bp[key], grad, atol=1e-12, rtol=1e-10)
    cross = {l: sum((actual_bp[k]-analytic[k]).square().sum().item()
                    for k in backbone_keys(l))**.5 for l in (0, 1)}
    assert all(v > 1e-10 for v in cross.values())
    applied = {}
    for mode in MODES:
        trial = P.double_probe(probe)
        trial.mode = mode
        expected, old_zs, want_aux = (backprop_grads(trial, fx, fy) if mode == 'auxhead_bp'
                                      else (dict(analytic), zs, torch.stack([
                                          P.ce(aux_logits(trial.p, l, hs[l]), fy) for l in (0, 1)])))
        _, expected['g'] = trial._mixed(old_zs, fy)
        tg = D.double_gate(gate)
        trial.train(fx, fy, tg)
        close(trial.last_aux, want_aux, atol=1e-12, rtol=1e-10)
        for key, grad in expected.items():
            clipped = grad/grad.norm().clamp_min(1)
            close(trial.state[key][0], .1*clipped, atol=1e-12, rtol=1e-10)
            close(trial.state[key][1], .001*clipped.square(), atol=1e-12, rtol=1e-10)
        applied[mode] = True
    warm = P.double_probe(probe)
    frozen = I.stamp({k: (warm.p[k], warm.state[k]) for k in baseline.p})
    warm.warmup_heads(fx, fy)
    assert I.stamp({k: (warm.p[k], warm.state[k]) for k in baseline.p}) == frozen
    assert warm.step == 0 and warm.aux_step == 1
    assert I.stamp((net, gate)) == before, 'startup checks wrote run state'
    close(torch.get_rng_state(), cpu_rng, atol=0, rtol=0)
    close(torch.cuda.get_rng_state(), cuda_rng, atol=0, rtol=0)
    Q.plain((net, gate))
    assert not torch.is_grad_enabled()
    result.update(aux_float64_autograd=True, aux_max_abs_error=errors,
                  local_losses_no_cross_layer_gradients=True, readouts_detach_backbone=True,
                  backbone_only_aux_ce=True, zero_output_initial_backbone_gradient=True,
                  bp_summed_loss_float64_autograd=True, bp_cross_layer_norm=cross,
                  production_gradients_applied=applied, sampler_equals_iter17_and_resumes=True,
                  head_warmup_freezes_backbone=True)
    return result


evaluate_subset = P.evaluate_subset
PhasePeaks = D.PhasePeaks


def run(mode='auxhead', smoke=None, seed=0, hid=64, save_ckpt=None):
    if smoke is not None and smoke < 1:
        raise ValueError('--smoke must be positive')
    if seed < 0:
        raise ValueError('--seed must be non-negative')
    if hid < 1:
        raise ValueError('--hid must be positive')
    if mode not in MODES:
        raise ValueError(mode)
    if save_ckpt is not None and Path(save_ckpt).exists():
        raise FileExistsError(save_ckpt)
    temporary = tempfile.TemporaryDirectory(prefix='iter34_') if smoke is not None else None
    old_windows = B.N_EVAL_WINDOWS
    try:
        name = f'iter34_{mode}' + (f'_seed{seed}' if seed else '') + '.json'
        path = Path(temporary.name)/name if temporary else I.ROOT/'results'/name
        if path.exists():
            raise FileExistsError(path)
        assert torch.cuda.is_available() and not torch.is_grad_enabled()
        torch.manual_seed(seed)
        np.random.seed(seed)
        tr, te, V = B.load()
        cut = int(len(tr)*.95)
        train = tr[:cut] if smoke is None else tr[:cut][:smoke]
        total = math.ceil(len(train)/(16*128))
        renamed, info = I.rename(te)
        net, gate = Model(V, total, mode, hid), C.Gate(V)
        x = torch.as_tensor(tr[:128].astype(np.int64), device='cuda')[None]
        y = torch.as_tensor(tr[1:129].astype(np.int64), device='cuda')[None]
        result = dict(checks=checks(net, gate, x, y, seed))
        sampler = Sampler(train, seed, V)
        peaks = PhasePeaks()  # Reset AFTER all startup probes have gone away.
        if smoke is not None:
            B.N_EVAL_WINDOWS = 16

        def predict(x, copy=True):
            zs = net.forward(x)[1]
            logits = sum(w*z for w, z in zip(net.p['g'], zs))
            return gate.predict(x, logits.softmax(-1))[0] if copy else logits.log_softmax(-1)

        # Materialize the FIRST quarter of the normal fixed windows, rather than
        # regenerating evenly spaced windows with a smaller N_EVAL_WINDOWS.
        torch.cuda.synchronize()
        eval_start = time.perf_counter()
        vx, vy = B.eval_windows(tr[cut:])
        subset_n = max(1, len(vx)//4)
        subset = vx[:subset_n].clone(), vy[:subset_n].clone()
        del vx, vy
        torch.cuda.synchronize()
        eval_seconds = time.perf_counter()-eval_start
        peaks.finish('eval')
        chars, seconds, curve = 0, 0., []
        tail_steps = max(1, math.ceil(total*.1))
        aux_sum = torch.zeros(2, device='cuda', dtype=torch.float64)
        aux_count = 0
        train_loss_sum = 0.
        milestones = {math.ceil(total*k/10) for k in range(1, 11)}
        torch.cuda.synchronize()
        start = time.perf_counter()
        for x, y in sampler:
            train_loss_sum += net.train(x, y, gate)
            if net.step > total-tail_steps:
                aux_sum += net.last_aux.double()
                aux_count += 1
            chars += int((y >= 0).sum())
            if net.step in milestones:
                torch.cuda.synchronize()
                seconds += time.perf_counter()-start
                peaks.finish('train')
                eval_start = time.perf_counter()
                before = I.stamp((net, gate))
                val_subset = evaluate_subset(predict, subset)
                assert I.stamp((net, gate)) == before, 'curve evaluation wrote state'
                curve.append((chars, val_subset))
                torch.cuda.synchronize()
                eval_seconds += time.perf_counter()-eval_start
                peaks.finish('eval')
                start = time.perf_counter()
        torch.cuda.synchronize()
        seconds += time.perf_counter()-start
        peaks.finish('train')
        assert chars == len(train) and net.step == gate.step == total and net.lr == S.BOTTOM
        assert aux_count == tail_steps
        if save_ckpt is not None:
            save_checkpoint(save_ckpt, net, gate, sampler, train, seed)
        del sampler  # Match iter17.epoch releasing its GPU dataset at exhaustion.
        Q.plain((net, gate))
        assert not torch.is_grad_enabled()
        torch.cuda.synchronize()
        eval_start = time.perf_counter()
        before = I.stamp((net, gate))
        # Final evaluations, novelty definition, copy heads and per-layer logits
        # are deliberately the same calls and fields as iter27.
        test = I.evaluate(predict, te)
        ren = I.evaluate(predict, renamed)
        val = I.evaluate(predict, tr[cut:])
        result.update(Q.novel(predict, te, renamed, json.loads((I.ROOT/'data/meta.json').read_text())['vocab']))
        result['test_loss_nocopy'] = I.evaluate(lambda x: predict(x, False), te)
        result['renamed_loss_nocopy'] = I.evaluate(lambda x: predict(x, False), renamed)
        result['per_layer_test_loss'] = [I.evaluate(lambda x, l=l: net.logits(x, l), te) for l in range(net.L)]
        weights = torch.zeros(5, device='cuda')
        x, _ = B.eval_windows(te)
        for xb in x.split(16):
            xb = xb.cuda()
            zs = net.forward(xb)[1]
            p = sum(w*z for w, z in zip(net.p['g'], zs)).softmax(-1)
            weights += gate.predict(xb, p)[1].sum((0, 1))
        result['mean_gate_weight_test'] = dict(zip(['net', *[f'L{L}' for L in C.LENS]], (weights/x.numel()).tolist()))
        assert I.stamp((net, gate)) == before, 'evaluation wrote model state'
        torch.cuda.synchronize()
        eval_seconds += time.perf_counter()-eval_start
        peaks.finish('eval')
        settings = dict(seed=seed, device='cuda', mode='copy', control_mode=mode, variant='width320', base='iter27',
                        bs=16, T=128, passes=1, train_end=cut, val_start=cut,
                        training_partition_chars=len(train), smoke=smoke, eval_windows=B.N_EVAL_WINDOWS,
                        sampling=f'non-overlapping target windows, shuffled with CPU generator seed {seed}',
                        renaming=info, evaluation_frozen=True, D=256, widths=W.WIDTHS, layers=3,
                        embedding_width=32, lags=list(range(6)), decays=[.5,.8,.95],
                        layer_input_width=1536, layer2_input_width=1536, state_lags=[1,2],
                        lr=S.TOP, lr_final=S.BOTTOM, lr_hold_fraction=S.HOLD,
                        lr_schedule='constant for s <= .75*S, then cosine to lr_final at s = S (1-indexed)',
                        lr_applies_to='all network parameters including g', optimizer='manual Adam',
                        betas=[.9,.999], eps=1e-8, gradient_clip_norm=1,
                        gradient_clip_scope='each parameter tensor separately, as iter27', weight_decay=0,
                        update_order=('full forward; summed readout and aux CE backward; gate; Adam sweep'
                                      if mode == 'auxhead_bp' else 'per-layer during forward sweep; gate and g after'),
                        network_losses=('three inference CE plus two aux CE; mixed CE drives only g'
                                        if mode == 'auxhead_bp' else
                                        'lower backbone: aux CE only; detached lower readouts: own CE; top: own CE; mixed CE: g only'),
                        hid=hid, aux_shape=[256, hid, V], aux_training_only=True,
                        aux_init='first W normal / sqrt(256), first bias zero; second W and bias zero',
                        aux_generator_seed=seed ^ AUX_SEED_XOR,
                        aux_loss_last10_scope=f'mean per-step aux CE over final {tail_steps} steps; layers 0,1',
                        memory_peak_scope='after startup checks; train and eval phases measured separately; allocated includes live baseline',
                        cross_layer_gradients=mode == 'auxhead_bp', g_init=1/3, g_constraint='none',
                        suffix_lengths=C.LENS, gate_lr=1e-2, gate_lr_schedule='constant',
                        curve_steps=sorted(milestones), curve_val_windows=subset_n,
                        curve_subset='first quarter of fixed val windows; no respacing',
                        eval_seconds_scope='curve subset preparation, curve evaluations and all final evaluations')
        counts = {key: p.numel() for key, p in net.p.items() if not key.startswith('aux')}
        aux_counts = {key: p.numel() for key, p in net.p.items() if key.startswith('aux')}
        aux_params = sum(aux_counts.values())
        network_params = sum(counts.values())
        attention_params = 0
        extra_rows = 0
        result.update(mode=mode, hid=hid, aux_params=aux_params, curve=curve, eval_seconds=eval_seconds,
                      aux_loss_last10=(aux_sum/aux_count).tolist(),
                      train_loss_mean=train_loss_sum/net.step,
                      train_peak_mib=peaks.peaks['train'][0],
                      eval_peak_mib=peaks.peaks['eval'][0],
                      train_peak_reserved_mib=peaks.peaks['train'][1],
                      eval_peak_reserved_mib=peaks.peaks['eval'][1],
                      references=dict(local_val=1.45627, bp_val=1.38687, topdown_val=1.55959),
                      test_loss=test, renamed_loss=ren, renamed_gap=ren-test, val_loss=val,
                      g=net.p['g'].tolist(), train_seconds=seconds, chars=chars, steps=net.step,
                      params=network_params+gate.A.numel(),
                      param_counts=dict(network=network_params, gate=gate.A.numel(),
                                        total=network_params+gate.A.numel(), attention=attention_params,
                                        W2_extra_rows=extra_rows, iter27_network=network_params-attention_params-extra_rows,
                                        by_tensor=counts, aux_by_tensor=aux_counts, aux=aux_params,
                                        training_total=network_params+gate.A.numel()+aux_params),
                      peak_allocated_mib=max(v[0] for v in peaks.peaks.values()),
                      peak_reserved_mib=max(v[1] for v in peaks.peaks.values()), settings=settings,
                      iter27_reference=dict(val_loss=1.45627, test_loss=1.3232, renamed_loss=1.6232),
                      iter24_reference=dict(val_loss=1.46686, test_loss=1.3320, renamed_loss=1.6298),
                      iter22_reference=dict(test_loss=1.3568, renamed_loss=1.6462),
                      iter23_reference=dict(test_loss=1.3530, renamed_loss=1.6430),
                      same_sampler_backprop_reference=dict(test_loss=1.340673, renamed_loss=1.686344))
        result['pass'] = dict(train=seconds < 90, val=val < result['iter24_reference']['val_loss'],
                              test=test <= 1.329, renamed=ren <= 1.667)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2, allow_nan=False)
        return result
    finally:
        B.N_EVAL_WINDOWS = old_windows
        if temporary:
            temporary.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=MODES, default='auxhead')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--save-ckpt', type=Path, metavar='PATH', help='save resumable final training state; refuse overwrite')
    parser.add_argument('--seed', type=int, default=0, help='initialization and sampler seed')
    parser.add_argument('--smoke', type=int, metavar='N',
                        help='train at most N prefix chars, evaluate 16 windows, clean temporary JSON (iter27 convention)')
    args = parser.parse_args()
    print(json.dumps(run(args.mode, args.smoke, args.seed, args.hid, args.save_ckpt), indent=2), flush=True)


