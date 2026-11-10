"""Same-step, detached top-readout teaching on iter27, with a backprop control.

Normal runs write results/iter33_MODE[_seedN].json without overwriting.
Smoke runs use iter30's 16-window evaluations and cleaned temporary JSON.
"""
import argparse
import copy
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import iter30 as P

W, S, Q, R, C, I, B = P.W, P.S, P.Q, P.R, P.C, P.I, P.B
torch.set_grad_enabled(False)
close = torch.testing.assert_close
MODES = ('topdown', 'topdown_bp')


def teaching_weight(step, total, alpha):
    """1-indexed: zero through 5%, linear to alpha at 20%, then held."""
    return alpha * min(1., max(0., (step/total-.05)/.15))


def teacher(z):
    # Only the top network readout; neither g nor the copy mixture participates.
    return z.detach().log_softmax(-1)


def kl_loss(z, top_logp, y):
    """KL(top || layer), averaged over the same valid positions as CE."""
    top_logp = top_logp.detach()
    per_position = (top_logp.exp()*(top_logp-z.log_softmax(-1))).sum(-1)
    return (per_position*(y >= 0)).sum()/(y >= 0).sum().clamp_min(1)


def kl_error(z, top_logp, y):
    valid = y >= 0
    return ((z.softmax(-1)-top_logp.detach().exp())*valid[..., None]
            /valid.sum().clamp_min(1))


def objectives(zs, y, a):
    top = teacher(zs[-1])
    kls = [kl_loss(z, top, y) for z in zs[:-1]]
    losses = [P.ce(z, y) if a == 0 else (1-a)*P.ce(z, y)+a*k
              for z, k in zip(zs[:-1], kls)]
    return [*losses, P.ce(zs[-1], y)], kls


def backprop_grads(net, x, y, a):
    keys = [key for key in net.p if key != 'g']
    with torch.enable_grad():
        view = P.leaf_view(net, keys)
        zs = view.forward(x)[1]
        losses, kls = objectives(zs, y, a)
        sum(losses).backward()
        assert all(view.p[key].grad is not None for key in keys)
        grads = {key: view.p[key].grad.detach() for key in keys}
    return grads, [z.detach() for z in zs], torch.stack(kls).detach()


class Model(P.Model):
    def __init__(self, V, total_steps, mode='topdown', alpha=.5):
        if mode not in MODES:
            raise ValueError(mode)
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('--alpha must be finite and in [0, 1]')
        super().__init__(V, total_steps, mode='local')
        self.mode, self.alpha, self.last_kl = mode, alpha, None

    def _layer(self, l, lower, M, ids, em):
        # Enforce the local boundary even when startup checks enable autograd.
        if l and self.mode == 'topdown':
            lower = lower.detach()
        return super()._layer(l, lower, M, ids, em)

    def _grads_from_error(self, l, x, h, e, u, ids, em, M):
        # iter22/27's local chain rule, with a supplied own-readout error.
        # It ends at this layer's inputs; no hidden-state error is sent down.
        p = self.p
        d = Q.delta(h, e, p[f'R{l}'])
        grads = {f'R{l}': Q.N.readout_delta(h, e),
                 f'c{l}': e.sum((0, 1)), f'b{l}': d.sum((0, 1))}
        if l:
            grads[f'W{l}'] = Q.N.readout_delta(u, d)
        else:
            for j in range(6):
                grads[f'A{j}'] = torch.zeros_like(p[f'A{j}']).index_add_(
                    0, ids[j].flatten(), d.flatten(0, 1))
            de = torch.zeros_like(p['E'][x])
            for k in range(3):
                grads[f'P{k}'] = Q.N.readout_delta(em[:, k], d)
                de += M[k].T @ (d @ p[f'P{k}'].T)
            grads['E'] = torch.zeros_like(p['E']).index_add_(
                0, x.flatten(), de.flatten(0, 1))
        return grads

    def _teaching_grads(self, l, x, y, h, z, u, ids, em, M, top, a):
        if l == self.L-1 or a == 0:
            # Preserve the original arithmetic exactly when teaching is off.
            return self._local_grads(l, x, y, h, z, u, ids, em, M)
        e = (1-a)*Q.N.error(z, y)+a*kl_error(z, top, y)
        return self._grads_from_error(l, x, h, e, u, ids, em, M)

    def train(self, x, y, gate):
        self.lr = S.schedule(self.step+1, self.total_steps)
        self.step += 1
        a = teaching_weight(self.step, self.total_steps, self.alpha)
        if self.mode == 'topdown_bp':
            grads, zs, self.last_kl = backprop_grads(self, x, y, a)
            final, grads['g'] = self._mixed(zs, y)
            gate.train(x, final.softmax(-1), y)
            self._update(grads)
        else:
            assert not torch.is_grad_enabled()
            # One full forward, retaining the local caches until the same-step
            # teacher exists. Disjoint updates still use the old activations.
            hs, zs, us, ids, em = self.forward(x)
            M = self.M[:, :x.shape[1], :x.shape[1]]
            top = teacher(zs[-1])
            self.last_kl = torch.stack([kl_loss(z, top, y) for z in zs[:-1]])
            for l in range(self.L):
                grads = self._teaching_grads(l, x, y, hs[l], zs[l],
                                             us[l-1] if l else None,
                                             ids, em, M, top, a)
                self._update(grads)
            final, g_grad = self._mixed(zs, y)
            gate.train(x, final.softmax(-1), y)
            self._update({'g': g_grad})
        return P.ce(final, y).item()


def double_gate(gate):
    out = copy.deepcopy(gate)
    for key in ('A', 'm', 'v'):
        setattr(out, key, getattr(out, key).double())
    return out


def layer_keys(l):
    return ({f'W{l}', f'b{l}', f'R{l}', f'c{l}'} if l else
            {'E', 'b0', 'R0', 'c0', *[f'A{j}' for j in range(6)],
             *[f'P{k}' for k in range(3)]})


def checks(net, gate, x, y, seed):
    """Float64 tiny cases on copies, with production updates also checked."""
    before = I.stamp((net, gate))
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    # Reuse iter30's init, RNG, causality, optimizer and baseline gradient checks
    # through its own class, without changing the actual run model or its mode.
    baseline = copy.copy(net)
    baseline.__class__ = P.Model
    baseline.mode = 'backprop' if net.mode == 'topdown_bp' else 'local'
    result = P.checks(baseline, gate, x, y, seed)
    fx, fy = x[:, :4], y[:, :4].clone()
    fy[0, 1] = -1
    probe = P.double_probe(net)
    probe.mode = 'topdown'
    hs, zs, us, ids, em = probe.forward(fx)
    M = probe.M[:, :fx.shape[1], :fx.shape[1]]
    top = teacher(zs[-1])
    errors, kl_errors = {}, {}
    for a in (0., .37, 1.):
        with torch.enable_grad():
            view = P.leaf_view(probe, list(probe.p))
            vzs = view.forward(fx)[1]
            losses, kls = objectives(vzs, fy, a)
            assert not teacher(vzs[-1]).requires_grad
            for l, loss in enumerate(losses):
                grads = torch.autograd.grad(loss, list(view.p.values()),
                                            retain_graph=True, allow_unused=True)
                analytic = probe._teaching_grads(l, fx, fy, hs[l], zs[l],
                                                  us[l-1] if l else None,
                                                  ids, em, M, top, a)
                for key, grad in zip(view.p, grads):
                    if key not in layer_keys(l):
                        assert grad is None, ('cross-layer gradient', a, l, key)
                    else:
                        close(analytic[key], grad, atol=1e-12, rtol=1e-10)
                        errors[f'{a}:{key}'] = (analytic[key]-grad).abs().max().item()
            for l, loss in enumerate(kls):
                expected = torch.autograd.grad(loss, vzs[l], retain_graph=True)[0]
                close(kl_error(zs[l], top, fy), expected, atol=1e-12, rtol=1e-10)
                kl_errors[str(l)] = (kl_error(zs[l], top, fy)-expected).abs().max().item()
    # Independent summed-loss reference for the fair BP control, including
    # active teaching and explicit confirmation that its layer boundaries flow.
    bp = P.double_probe(net)
    bp.mode = 'topdown_bp'
    bp_errors, cross = {}, {}
    for a in (0., .37):
        actual, _, _ = backprop_grads(bp, fx, fy, a)
        keys = [key for key in bp.p if key != 'g']
        with torch.enable_grad():
            view = P.leaf_view(bp, keys)
            logits = view.forward(fx)[1]
            target = logits[-1].detach().softmax(-1)
            loss = P.ce(logits[-1], fy)
            for z in logits[:-1]:
                divergence = F.kl_div(z.log_softmax(-1), target, reduction='none').sum(-1)
                divergence = divergence[fy >= 0].mean()
                loss = loss+(1-a)*P.ce(z, fy)+a*divergence
            expected = torch.autograd.grad(loss, [view.p[key] for key in keys])
        for key, grad in zip(keys, expected):
            close(actual[key], grad, atol=1e-12, rtol=1e-10)
            bp_errors[f'{a}:{key}'] = (actual[key]-grad).abs().max().item()
        if a == 0:
            baseline_grads, _ = P.backprop_grads(bp, fx, fy)
            for key in keys:
                close(actual[key], baseline_grads[key], atol=1e-12, rtol=1e-10)
        else:
            for l in (0, 1):
                local = probe._teaching_grads(l, fx, fy, hs[l], zs[l],
                                                us[l-1] if l else None,
                                                ids, em, M, top, a)
                cross[l] = sum((actual[k]-local[k]).square().sum().item()
                               for k in layer_keys(l))**.5
                assert cross[l] > 1e-10, ('missing BP cross-layer derivative', l)
    # Delayed updates at alpha=0 must equal interleaved iter30 local updates,
    # including Adam states and gate, across multiple steps with a masked target.
    lp, rp = P.double_probe(net), P.double_probe(baseline)
    lp.mode, lp.alpha, lp.total_steps = 'topdown', 0., 3
    rp.mode, rp.total_steps = 'local', 3
    lg, rg = double_gate(gate), double_gate(gate)
    for _ in range(3):
        close(torch.tensor(lp.train(fx, fy, lg)), torch.tensor(rp.train(fx, fy, rg)),
              atol=0, rtol=0)
        S.same(lp, rp, lg, rg)
        for key in lp.p:
            close(lp.p[key], rp.p[key], atol=1e-12, rtol=1e-10)
            for actual, expected in zip(lp.state[key], rp.state[key]):
                close(actual, expected, atol=1e-12, rtol=1e-10)
        for key in ('A', 'm', 'v'):
            close(getattr(lg, key), getattr(rg, key), atol=1e-12, rtol=1e-10)
    # Active teaching: verify the actual train path uses the same-step target,
    # exactly one forward per layer, and applies all expected clipped gradients.
    applied = {}
    for mode in MODES:
        trial = P.double_probe(net)
        trial.mode, trial.alpha, trial.total_steps, trial.step = mode, .5, 100, 19
        tg = double_gate(gate)
        a = teaching_weight(20, 100, .5)
        if mode == 'topdown_bp':
            expected, old_zs, want_kl = backprop_grads(trial, fx, fy, a)
        else:
            hh, old_zs, uu, ii, ee = trial.forward(fx)
            tt = teacher(old_zs[-1])
            want_kl = torch.stack([kl_loss(z, tt, fy) for z in old_zs[:-1]])
            expected = {}
            for l in range(3):
                expected.update(trial._teaching_grads(l, fx, fy, hh[l], old_zs[l],
                                                       uu[l-1] if l else None,
                                                       ii, ee, M, tt, a))
        _, expected['g'] = trial._mixed(old_zs, fy)
        layers, updates = [], []
        original_class, update = type(trial), trial._update
        # Trace at the class boundary: leaf_view copies the model for BP, so
        # an instance closure bound to trial would bypass its autograd leaves.
        class TracedModel(original_class):
            def _layer(self, l, *args):
                layers.append(l)
                return super()._layer(l, *args)
        def recorded_update(grads):
            updates.append(tuple(grads))
            update(grads)
        trial.__class__, trial._update = TracedModel, recorded_update
        try:
            trial.train(fx, fy, tg)
        finally:
            trial.__class__ = original_class
            del trial._update
        assert layers == [0, 1, 2], ('extra forward', mode, layers)
        assert len(updates) == (1 if mode == 'topdown_bp' else 4)
        close(trial.last_kl, want_kl, atol=1e-12, rtol=1e-10)
        for key, grad in expected.items():
            clipped = grad/grad.norm().clamp_min(1)
            close(trial.state[key][0], .1*clipped, atol=1e-12, rtol=1e-10)
            close(trial.state[key][1], .001*clipped.square(), atol=1e-12, rtol=1e-10)
        applied[mode] = True
    for step, expected in ((1, 0.), (5, 0.), (12.5, .25), (20, .5), (100, .5)):
        assert abs(teaching_weight(step, 100, .5)-expected) < 1e-15
    assert I.stamp((net, gate)) == before, 'startup checks wrote run state'
    close(torch.get_rng_state(), cpu_rng, atol=0, rtol=0)
    close(torch.cuda.get_rng_state(), cuda_rng, atol=0, rtol=0)
    Q.plain((net, gate))
    assert not torch.is_grad_enabled()
    result.update(local_losses_no_cross_layer_gradients=True, kl_teacher_detached=True,
                  teaching_float64_autograd=True, teaching_max_abs_error=errors,
                  kl_logit_gradient_float64_autograd=True, kl_max_abs_error=kl_errors,
                  bp_teaching_float64_autograd=True, bp_teaching_max_abs_error=bp_errors,
                  bp_alpha_zero_equals_iter30_backprop=True, bp_teaching_cross_layer_norm=cross,
                  alpha_zero_three_steps_equal_iter30_local=True,
                  same_step_one_forward_gradients_applied=applied, teaching_ramp=True)
    return result


evaluate_subset = P.evaluate_subset


class PhasePeaks:
    """Allocated/reserved maxima by phase, excluding startup checks.

    Reserved memory remains allocator-cache dependent; allocated peaks are the
    phase measurements to compare. Resetting a counter does not empty the cache.
    """
    def __init__(self):
        self.peaks = {'train': [0., 0.], 'eval': [0., 0.]}
        torch.cuda.reset_peak_memory_stats()

    def finish(self, phase):
        torch.cuda.synchronize()
        values = (torch.cuda.max_memory_allocated()/2**20,
                  torch.cuda.max_memory_reserved()/2**20)
        self.peaks[phase] = [max(a, b) for a, b in zip(self.peaks[phase], values)]
        torch.cuda.reset_peak_memory_stats()


def run(mode='topdown', smoke=None, seed=0, alpha=.5):
    if smoke is not None and smoke < 1:
        raise ValueError('--smoke must be positive')
    if seed < 0:
        raise ValueError('--seed must be non-negative')
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('--alpha must be finite and in [0, 1]')
    if mode not in MODES:
        raise ValueError(mode)
    temporary = tempfile.TemporaryDirectory(prefix='iter33_') if smoke is not None else None
    old_windows = B.N_EVAL_WINDOWS
    try:
        name = f'iter33_{mode}' + (f'_seed{seed}' if seed else '') + '.json'
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
        net, gate = Model(V, total, mode, alpha), C.Gate(V)
        x = torch.as_tensor(tr[:128].astype(np.int64), device='cuda')[None]
        y = torch.as_tensor(tr[1:129].astype(np.int64), device='cuda')[None]
        result = dict(checks=checks(net, gate, x, y, seed))
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
        kl_sum = torch.zeros(2, device='cuda', dtype=torch.float64)
        kl_count = 0
        train_loss_sum = 0.
        milestones = {math.ceil(total*k/10) for k in range(1, 11)}
        torch.cuda.synchronize()
        start = time.perf_counter()
        for x, y in R.epoch(train, seed, len(train), V):
            train_loss_sum += net.train(x, y, gate)
            if net.step > total-tail_steps:
                kl_sum += net.last_kl.double()
                kl_count += 1
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
        assert kl_count == tail_steps
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
                        update_order=('full forward; summed teaching-objective backward; gate; one network Adam sweep'
                                      if mode == 'topdown_bp' else 'full forward; local manual updates 0,1,2; gate and g after'),
                        network_losses='layers 0,1: (1-a)*CE+a*KL(top||layer); top: CE; mixed CE drives only g',
                        alpha=alpha,
                        teaching_schedule='a=alpha*clamp((s/S-.05)/.15, 0, 1); s is 1-indexed',
                        teaching_target='same-step detached top own-readout softmax; network only',
                        kl_reduction='mean over valid (y >= 0) positions, same as CE',
                        mean_kl_last10_scope=f'unweighted mean of per-step KL over final {tail_steps} steps; layers 0,1',
                        memory_peak_scope='after startup checks; train and eval phases measured separately; allocated includes live baseline',
                        cross_layer_gradients=mode == 'topdown_bp', g_init=1/3, g_constraint='none',
                        suffix_lengths=C.LENS, gate_lr=1e-2, gate_lr_schedule='constant',
                        curve_steps=sorted(milestones), curve_val_windows=subset_n,
                        curve_subset='first quarter of fixed val windows; no respacing',
                        eval_seconds_scope='curve subset preparation, curve evaluations and all final evaluations')
        counts = {key: p.numel() for key, p in net.p.items()}
        network_params = sum(counts.values())
        attention_params = 0
        extra_rows = 0
        result.update(mode=mode, alpha=alpha, curve=curve, eval_seconds=eval_seconds,
                      mean_kl_last10=(kl_sum/kl_count).tolist(),
                      train_loss_mean=train_loss_sum/net.step,
                      train_peak_mib=peaks.peaks['train'][0],
                      eval_peak_mib=peaks.peaks['eval'][0],
                      train_peak_reserved_mib=peaks.peaks['train'][1],
                      eval_peak_reserved_mib=peaks.peaks['eval'][1],
                      references=dict(local_val=1.45627, bp_val=1.38687, attn_local_val=1.41043),
                      test_loss=test, renamed_loss=ren, renamed_gap=ren-test, val_loss=val,
                      g=net.p['g'].tolist(), train_seconds=seconds, chars=chars, steps=net.step,
                      params=network_params+gate.A.numel(),
                      param_counts=dict(network=network_params, gate=gate.A.numel(),
                                        total=network_params+gate.A.numel(), attention=attention_params,
                                        W2_extra_rows=extra_rows, iter27_network=network_params-attention_params-extra_rows,
                                        by_tensor=counts),
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
    parser.add_argument('--mode', choices=MODES, default='topdown')
    parser.add_argument('--alpha', type=float, default=.5, help='maximum lower-layer teaching weight in [0, 1]')
    parser.add_argument('--seed', type=int, default=0, help='initialization and sampler seed')
    parser.add_argument('--smoke', type=int, metavar='N',
                        help='train at most N prefix chars, evaluate 16 windows, clean temporary JSON (iter27 convention)')
    args = parser.parse_args()
    print(json.dumps(run(args.mode, args.smoke, args.seed, args.alpha), indent=2), flush=True)

