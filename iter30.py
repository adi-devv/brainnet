"""iter27 controls: unchanged local learning, summed-loss backprop, or local attention.

Smoke runs retain iter27's 16-window evaluation and cleaned temporary JSON output.
Normal runs write results/iter30_MODE[_seedN].json, refusing to overwrite a result.
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
import iter27 as W

S, Q, R, C, I, B = W.S, W.Q, W.R, W.C, W.I, W.B
torch.set_grad_enabled(False)
close = torch.testing.assert_close
ATTENTION_KEYS = ('attn_Q', 'attn_K', 'attn_V', 'attn_O')
TOP_KEYS = ('W2', 'b2', 'R2', 'c2', *ATTENTION_KEYS)


def ce(z, y):
    return F.cross_entropy(z.flatten(0, 1), y.flatten(), ignore_index=-1)


def upper_input(lower, M):
    lag1, lag2 = torch.zeros_like(lower), torch.zeros_like(lower)
    lag1[:, 1:] = lower[:, :-1]
    lag2[:, 2:] = lower[:, :-2]
    return torch.cat([lower, *(M @ lower[:, None]).unbind(1), lag1, lag2], -1)


def rope(z):
    """Interleaved pairs, base 10000, positions restarted at each window."""
    dim, T = z.shape[-1], z.shape[-2]
    freq = 10000. ** (-torch.arange(0, dim, 2, device=z.device, dtype=z.dtype)/dim)
    angle = torch.arange(T, device=z.device, dtype=z.dtype)[:, None]*freq
    a, b = z[..., 0::2], z[..., 1::2]
    return torch.stack([a*angle.cos()-b*angle.sin(),
                        a*angle.sin()+b*angle.cos()], -1).flatten(-2)


def attention(lower, p):
    b, T, d = lower.shape
    q, k, v = [(lower @ p[key]).reshape(b, T, 4, 64).transpose(1, 2)
               for key in ATTENTION_KEYS[:3]]
    q, k = rope(q), rope(k)
    future = torch.ones(T, T, dtype=torch.bool, device=lower.device).triu(1)
    weights = ((q @ k.transpose(-1, -2))/math.sqrt(64)).masked_fill(future, -torch.inf).softmax(-1)
    return (weights @ v).transpose(1, 2).reshape(b, T, d) @ p['attn_O']


def leaf_view(net, keys):
    """Ephemeral autograd leaves sharing storage; persistent tensors stay plain."""
    view = copy.copy(net)
    view.p = dict(net.p)
    for key in keys:
        view.p[key] = net.p[key].detach().requires_grad_()
    return view


def backprop_grads(net, x, y, detach_between=False):
    keys = [key for key in net.p if key != 'g']
    with torch.enable_grad():
        view = leaf_view(net, keys)
        M, ids, em = view._prepare(x)
        lower, zs = None, []
        for l in range(view.L):
            if l and detach_between:
                lower = lower.detach()
            lower, z, _ = view._layer(l, lower, M, ids, em)
            zs.append(z)
        # Only the three own-readout losses drive the network. The mixed loss
        # still drives g separately, exactly as in iter27.
        sum(ce(z, y) for z in zs).backward()
        assert all(view.p[key].grad is not None for key in keys)
        grads = {key: view.p[key].grad.detach() for key in keys}
    return grads, [z.detach() for z in zs]


class Model(W.Model):
    def __init__(self, V, total_steps, mode='local'):
        super().__init__(V, total_steps)
        if mode not in ('local', 'backprop', 'attn'):
            raise ValueError(mode)
        self.mode = mode
        if mode == 'attn':
            # No global RNG draws, including no change to the original W2 rows.
            gen = torch.Generator(device=self.p['W2'].device)
            gen.manual_seed(torch.initial_seed() ^ 0x30A77E)
            def rand(shape):
                return torch.randn(shape, device=self.p['W2'].device, generator=gen)
            extra = rand((256, 320))/math.sqrt(1024)  # iter27's random W2-row rule
            self.p['W2'] = torch.cat([self.p['W2'], extra], 0)
            for key in ATTENTION_KEYS[:3]:
                self.p[key] = rand((256, 256))/math.sqrt(256)
            self.p['attn_O'] = self.p['W2'].new_zeros(256, 256)
            for key in ('W2', *ATTENTION_KEYS):
                self.state[key] = [torch.zeros_like(self.p[key]), torch.zeros_like(self.p[key])]

    def _layer(self, l, lower, M, ids, em):
        if self.mode != 'attn' or l != 2:
            return super()._layer(l, lower, M, ids, em)
        # This boundary is enforced inside the production forward, even when
        # callers enable autograd on all lower-layer parameters.
        lower = lower.detach()
        base = upper_input(lower, M)
        extra = attention(lower, self.p)
        u = torch.cat([base, extra], -1)
        # Same linear map as u @ W2. Keeping the old product's shape also keeps
        # the O=0 forward identical without changing its GEMM reduction order.
        a = base @ self.p['W2'][:1536] + extra @ self.p['W2'][1536:] + self.p['b2']
        h = a.tanh()
        return h, h @ self.p['R2'] + self.p['c2'], u

    def _top_grads(self, lower, M, ids, em, y):
        with torch.enable_grad():
            view = leaf_view(self, TOP_KEYS)
            h, z, _ = view._layer(2, lower.detach(), M, ids, em)
            ce(z, y).backward()
            assert all(view.p[key].grad is not None for key in TOP_KEYS)
            grads = {key: view.p[key].grad.detach() for key in TOP_KEYS}
        return grads, h.detach(), z.detach()

    def _mixed(self, zs, y):
        final = sum(w*z for w, z in zip(self.p['g'], zs))
        ef = Q.N.error(final, y)
        return final, torch.stack([(ef*z).sum() for z in zs])

    def train(self, x, y, gate):
        if self.mode == 'local':
            # No rewrite of iter27's forward, derivatives, update order or Adam.
            return super().train(x, y, gate)
        self.lr = S.schedule(self.step+1, self.total_steps)
        self.step += 1
        if self.mode == 'backprop':
            grads, zs = backprop_grads(self, x, y)
            final, grads['g'] = self._mixed(zs, y)
            gate.train(x, final.softmax(-1), y)
            # One full forward, one backward, one Adam sweep over every tensor.
            self._update(grads)
        else:
            M, ids, em = self._prepare(x)
            lower, zs = None, []
            for l in range(self.L):
                if l == 2:
                    grads, h, z = self._top_grads(lower, M, ids, em, y)
                else:
                    h, z, u = self._layer(l, lower, M, ids, em)
                    grads = self._local_grads(l, x, y, h, z, u, ids, em, M)
                zs.append(z)
                self._update(grads)
                lower = h  # Stored pre-update activation, as in iter27.
            final, g_grad = self._mixed(zs, y)
            gate.train(x, final.softmax(-1), y)
            self._update({'g': g_grad})
        return ce(final, y).item()


def double_probe(net):
    probe = copy.copy(net)
    probe.p = {key: p.double().clone() for key, p in net.p.items()}
    probe.M = net.M.double().clone()
    probe.state = {key: [torch.zeros_like(p), torch.zeros_like(p)] for key, p in probe.p.items()}
    return probe


def reference_top(lower, M, p):
    """Independent, explicit float64 reference (no production RoPE/attention)."""
    b, T, _ = lower.shape
    pos = torch.arange(T, device=lower.device, dtype=lower.dtype)
    freq = 1/(10000. ** (torch.arange(32, device=lower.device, dtype=lower.dtype)/32))
    angle = pos[:, None]*freq
    projections = []
    for name in ATTENTION_KEYS[:3]:
        z = (lower @ p[name]).view(b, T, 4, 32, 2)
        if name != 'attn_V':
            even, odd = z[..., 0], z[..., 1]
            cosine, sine = angle.cos()[None, :, None], angle.sin()[None, :, None]
            z = torch.stack([even*cosine-odd*sine, even*sine+odd*cosine], -1)
        projections.append(z.flatten(-2).permute(0, 2, 1, 3))
    q, k, v = projections
    rows = []
    for t in range(T):
        score = (q[:, :, t:t+1] * k[:, :, :t+1]).sum(-1)/8
        rows.append((score.softmax(-1)[..., None]*v[:, :, :t+1]).sum(-2))
    extra = torch.stack(rows, 1).reshape(b, T, 256) @ p['attn_O']
    # Full concatenated matmul independently checks both original and extra W2 rows.
    u = torch.cat([upper_input(lower, M), extra], -1)
    return (u @ p['W2'] + p['b2']).tanh() @ p['R2'] + p['c2']


def check_causality(net, x):
    original = net.forward(x)
    raw = attention(original[0][1], net.p) if net.mode == 'attn' else None
    for t in (0, 7, 46, 126):
        if t+1 >= x.shape[1]:
            continue
        other = x.clone()
        other[:, t+1:] = (other[:, t+1:]+1)%net.V
        changed = net.forward(other)
        for a, b in zip(original[0]+original[1], changed[0]+changed[1]):
            close(a[:, :t+1], b[:, :t+1], atol=1e-6, rtol=0)
        if raw is not None:
            close(raw[:, :t+1], attention(changed[0][1], net.p)[:, :t+1], atol=1e-6, rtol=0)


def checks(net, gate, x, y, seed):
    """Small, RNG-preserving checks on copies; never train the actual run model."""
    before = I.stamp((net, gate))
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    result = {}
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        ref, rg = W.Model(net.V, net.total_steps), C.Gate(net.V)
        close(torch.get_rng_state(), cpu_rng, atol=0, rtol=0)
        close(torch.cuda.get_rng_state(), cuda_rng, atol=0, rtol=0)
        for key, p in ref.p.items():
            actual = net.p[key][:1536] if net.mode == 'attn' and key == 'W2' else net.p[key]
            close(actual, p, atol=0, rtol=0)
        for key, p in net.p.items():
            assert len(net.state[key]) == 2
            assert all(s.shape == p.shape and not s.count_nonzero() for s in net.state[key])
        for key in ('A', 'm', 'v'):
            close(getattr(gate, key), getattr(rg, key), atol=0, rtol=0)
        a, b = net.forward(x), ref.forward(x)
        for actual, expected in zip(a[0]+a[1], b[0]+b[1]):
            close(actual, expected, atol=1e-6, rtol=0)
        check_causality(net, x)
        result.update(initial_iter27_parameters_exact=True, initial_rng_exact=True,
                      initial_gate_exact=True, zero_adam_states=True,
                      initial_forward_equal_iter27=True, all_layer_causality=True)

        fx, fy = x[:, :4], y[:, :4].clone()
        fy[0, 1] = -1
        if net.mode in ('local', 'backprop'):
            probe = double_probe(net)
            hs, zs, us, ids, em = probe.forward(fx)
            M = probe._prepare(fx)[0]
            local = {}
            for l in range(3):
                local.update(probe._local_grads(l, fx, fy, hs[l], zs[l],
                                               us[l-1] if l else None, ids, em, M))
            blocked, _ = backprop_grads(probe, fx, fy, detach_between=True)
            errors = {}
            for key, grad in local.items():
                close(blocked[key], grad, atol=1e-12, rtol=1e-10)
                errors[key] = (blocked[key]-grad).abs().max().item()
            result.update(detached_autograd_equals_iter27_local=True,
                          local_autograd_max_abs_error=errors)
            if net.mode == 'backprop':
                flowing, _ = backprop_grads(probe, fx, fy)
                cross = {}
                for l, keys in ((0, ['E', *[f'A{j}' for j in range(6)],
                                    *[f'P{k}' for k in range(3)], 'b0']),
                                (1, ['W1', 'b1'])):
                    cross[l] = sum((flowing[k]-blocked[k]).square().sum().item() for k in keys)**.5
                    assert cross[l] > 1e-10, ('missing cross-layer derivative', l)
                for key in ('W2', 'b2', 'R2', 'c2', 'R1', 'c1', 'R0', 'c0'):
                    close(flowing[key], blocked[key], atol=1e-12, rtol=1e-10)
                # Check the actual train method applies the summed, clipped
                # gradients in one sweep, including g's unchanged mixed loss.
                expected = dict(flowing)
                pzs = probe.forward(fx)[1]
                _, expected['g'] = probe._mixed(pzs, fy)
                pg = copy.deepcopy(gate)
                for key in ('A', 'm', 'v'):
                    setattr(pg, key, getattr(pg, key).double())
                events = []
                update = probe._update
                def recorded(grads):
                    events.append(tuple(grads))
                    update(grads)
                probe._update = recorded
                try:
                    probe.train(fx, fy, pg)
                finally:
                    del probe._update
                assert len(events) == 1 and set(events[0]) == set(probe.p)
                for key, grad in expected.items():
                    clipped = grad/grad.norm().clamp_min(1)
                    close(probe.state[key][0], .1*clipped, atol=1e-12, rtol=1e-10)
                    close(probe.state[key][1], .001*clipped.square(), atol=1e-12, rtol=1e-10)
                result.update(cross_layer_gradient_norm=cross, one_network_adam_sweep=True,
                              summed_gradient_applied=True)
            else:
                # Exercise the inherited production path against fresh iter27.
                lp, rp = double_probe(net), double_probe(ref)
                lg, rpg = copy.deepcopy(gate), copy.deepcopy(rg)
                for g in (lg, rpg):
                    for key in ('A', 'm', 'v'):
                        setattr(g, key, getattr(g, key).double())
                close(torch.tensor(lp.train(fx, fy, lg)), torch.tensor(rp.train(fx, fy, rpg)), atol=0, rtol=0)
                for key in lp.p:
                    close(lp.p[key], rp.p[key], atol=1e-12, rtol=1e-10)
                    for s, t in zip(lp.state[key], rp.state[key]):
                        close(s, t, atol=1e-12, rtol=1e-10)
                for key in ('A', 'm', 'v'):
                    close(getattr(lg, key), getattr(rpg, key), atol=1e-12, rtol=1e-10)
                result['local_step_equals_iter27'] = True
        else:
            assert not net.p['attn_O'].count_nonzero()
            assert net.p['W2'].shape == (1792, 320)
            probe = double_probe(net)
            # Nonzero O makes the q/k/v derivative and causality checks nonvacuous.
            probe.p['attn_O'].copy_(torch.arange(256*256, device=x.device,
                                               dtype=torch.float64).reshape(256, 256).sin()*.02)
            active = copy.copy(net)
            active.p = dict(net.p, attn_O=probe.p['attn_O'].float())
            check_causality(active, x)
            hs = probe.forward(fx)[0]
            M, ids, em = probe._prepare(fx)
            analytic, _, _ = probe._top_grads(hs[1], M, ids, em, fy)
            with torch.enable_grad():
                leaves = {k: probe.p[k].detach().clone().requires_grad_() for k in TOP_KEYS}
                z = reference_top(hs[1].detach(), M, leaves)
                expected = torch.autograd.grad(ce(z, fy), list(leaves.values()))
            errors = {}
            for key, grad in zip(TOP_KEYS, expected):
                close(analytic[key], grad, atol=1e-12, rtol=1e-10)
                errors[key] = (analytic[key]-grad).abs().max().item()
                assert grad.norm() > 0, ('vacuous top gradient check', key)
            with torch.enable_grad():
                view = leaf_view(probe, list(probe.p))
                top_loss = ce(view.forward(fx)[1][2], fy)
                grads = torch.autograd.grad(top_loss, list(view.p.values()), allow_unused=True)
            for key, grad in zip(view.p, grads):
                if key not in TOP_KEYS:
                    assert grad is None, ('top loss leaked to lower layer or g', key)
                else:
                    close(grad, analytic[key], atol=1e-12, rtol=1e-10)
            # Production interleaved training must use exactly these top gradients.
            pg = copy.deepcopy(gate)
            for key in ('A', 'm', 'v'):
                setattr(pg, key, getattr(pg, key).double())
            probe.train(fx, fy, pg)
            for key, grad in analytic.items():
                clipped = grad/grad.norm().clamp_min(1)
                close(probe.state[key][0], .1*clipped, atol=1e-12, rtol=1e-10)
                close(probe.state[key][1], .001*clipped.square(), atol=1e-12, rtol=1e-10)
            result.update(zero_O_initial_equivalence=True, active_attention_causality=True,
                          top_local_float64_autograd=True, top_autograd_max_abs_error=errors,
                          top_loss_no_lower_gradients=True, top_gradient_applied=True)

        assert S.schedule(1, 8) == S.schedule(6, 8) == S.TOP
        assert S.schedule(8, 8) == S.BOTTOM
        assert abs(S.schedule(7, 8)-5.5e-4) < 1e-15
    assert I.stamp((net, gate)) == before, 'startup checks wrote run state'
    close(torch.get_rng_state(), cpu_rng, atol=0, rtol=0)
    close(torch.cuda.get_rng_state(), cuda_rng, atol=0, rtol=0)
    Q.plain((net, gate))
    assert not torch.is_grad_enabled()
    result.update(schedule_values=True, state_and_rng_untouched=True)
    return result


def evaluate_subset(predict, windows):
    x, y = windows
    total = 0.
    for a in range(0, len(x), 16):
        logits = predict(x[a:a+16].cuda()).float()
        total += F.cross_entropy(logits.flatten(0, 1), y[a:a+16].cuda().flatten(), reduction='sum').item()
    return total/y.numel()


def run(mode='local', smoke=None, seed=0):
    if smoke is not None and smoke < 1:
        raise ValueError('--smoke must be positive')
    if seed < 0:
        raise ValueError('--seed must be non-negative')
    if mode not in ('local', 'backprop', 'attn'):
        raise ValueError(mode)
    temporary = tempfile.TemporaryDirectory(prefix='iter30_') if smoke is not None else None
    old_windows = B.N_EVAL_WINDOWS
    try:
        name = f'iter30_{mode}' + (f'_seed{seed}' if seed else '') + '.json'
        path = Path(temporary.name)/name if temporary else I.ROOT/'results'/name
        if path.exists():
            raise FileExistsError(path)
        assert torch.cuda.is_available() and not torch.is_grad_enabled()
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.reset_peak_memory_stats()
        tr, te, V = B.load()
        cut = int(len(tr)*.95)
        train = tr[:cut] if smoke is None else tr[:cut][:smoke]
        total = math.ceil(len(train)/(16*128))
        renamed, info = I.rename(te)
        net, gate = Model(V, total, mode), C.Gate(V)
        x = torch.as_tensor(tr[:128].astype(np.int64), device='cuda')[None]
        y = torch.as_tensor(tr[1:129].astype(np.int64), device='cuda')[None]
        result = dict(checks=checks(net, gate, x, y, seed))
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
        chars, seconds, curve = 0, 0., []
        milestones = {math.ceil(total*k/10) for k in range(1, 11)}
        torch.cuda.synchronize()
        start = time.perf_counter()
        for x, y in R.epoch(train, seed, len(train), V):
            net.train(x, y, gate)
            chars += int((y >= 0).sum())
            if net.step in milestones:
                torch.cuda.synchronize()
                seconds += time.perf_counter()-start
                eval_start = time.perf_counter()
                before = I.stamp((net, gate))
                val_subset = evaluate_subset(predict, subset)
                assert I.stamp((net, gate)) == before, 'curve evaluation wrote state'
                curve.append((chars, val_subset))
                torch.cuda.synchronize()
                eval_seconds += time.perf_counter()-eval_start
                start = time.perf_counter()
        torch.cuda.synchronize()
        seconds += time.perf_counter()-start
        assert chars == len(train) and net.step == gate.step == total and net.lr == S.BOTTOM
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
        settings = dict(seed=seed, device='cuda', mode='copy', control_mode=mode, variant='width320', base='iter27',
                        bs=16, T=128, passes=1, train_end=cut, val_start=cut,
                        training_partition_chars=len(train), smoke=smoke, eval_windows=B.N_EVAL_WINDOWS,
                        sampling=f'non-overlapping target windows, shuffled with CPU generator seed {seed}',
                        renaming=info, evaluation_frozen=True, D=256, widths=W.WIDTHS, layers=3,
                        embedding_width=32, lags=list(range(6)), decays=[.5,.8,.95],
                        layer_input_width=1536, layer2_input_width=1792 if mode == 'attn' else 1536, state_lags=[1,2],
                        lr=S.TOP, lr_final=S.BOTTOM, lr_hold_fraction=S.HOLD,
                        lr_schedule='constant for s <= .75*S, then cosine to lr_final at s = S (1-indexed)',
                        lr_applies_to='all network parameters including g', optimizer='manual Adam',
                        betas=[.9,.999], eps=1e-8, gradient_clip_norm=1,
                        gradient_clip_scope='each parameter tensor separately, as iter27', weight_decay=0,
                        update_order=('full forward; summed own-readout backward; gate; one network Adam sweep'
                                      if mode == 'backprop' else 'per-layer during forward sweep; gate and g after'),
                        network_losses='sum of three per-layer mean CE losses; mixed CE drives only g',
                        cross_layer_gradients=mode == 'backprop', g_init=1/3, g_constraint='none',
                        suffix_lengths=C.LENS, gate_lr=1e-2, gate_lr_schedule='constant',
                        curve_steps=sorted(milestones), curve_val_windows=subset_n,
                        curve_subset='first quarter of fixed val windows; no respacing',
                        eval_seconds_scope='curve subset preparation, curve evaluations and all final evaluations')
        if mode == 'attn':
            settings['attention'] = dict(source_layer=1, target_layer=2, heads=4, head_dim=64,
                                         rope_base=10000, rope_pairs='interleaved', causal=True, window=128,
                                         biases=False, dropout=0, output_init='zero', qkv_init='normal / sqrt(256)',
                                         W2_extra_init='normal / sqrt(1024)',
                                         separate_generator_seed=seed ^ 0x30A77E,
                                         local_loss='layer 2 own CE only; layer 1 input detached',
                                         original_EMAs_and_lags_retained=True)
        counts = {key: p.numel() for key, p in net.p.items()}
        network_params = sum(counts.values())
        attention_params = sum(counts.get(key, 0) for key in ATTENTION_KEYS)
        extra_rows = 256*320 if mode == 'attn' else 0
        result.update(mode=mode, curve=curve, eval_seconds=eval_seconds,
                      test_loss=test, renamed_loss=ren, renamed_gap=ren-test, val_loss=val,
                      g=net.p['g'].tolist(), train_seconds=seconds, chars=chars, steps=net.step,
                      params=network_params+gate.A.numel(),
                      param_counts=dict(network=network_params, gate=gate.A.numel(),
                                        total=network_params+gate.A.numel(), attention=attention_params,
                                        W2_extra_rows=extra_rows, iter27_network=network_params-attention_params-extra_rows,
                                        by_tensor=counts),
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20, settings=settings,
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
    parser.add_argument('--mode', choices=('local', 'backprop', 'attn'), default='local')
    parser.add_argument('--seed', type=int, default=0, help='initialization and sampler seed')
    parser.add_argument('--smoke', type=int, metavar='N',
                        help='train at most N prefix chars, evaluate 16 windows, clean temporary JSON (iter27 convention)')
    args = parser.parse_args()
    print(json.dumps(run(args.mode, args.smoke, args.seed), indent=2), flush=True)
