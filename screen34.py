"""Paired 64-step screens from an uncompressed iter30 local 15% checkpoint.

Examples (checkpoint creation is a separate, one-time cost):
  python screen34.py --make-ckpt
  python screen34.py --candidate auxhead

Aux heads warm up on 32 extra batches with the backbone and gate frozen. The
paired local branch consumes the identical 32 idle batches. These extra batches
do not advance the original backbone Adam clock or schedule; both 64-step
branches resume at checkpoint.step+1. Warmup uses that next-step learning rate
and fresh head Adam clocks. All evaluations exclude the training-only heads.
"""
import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import iter30 as P
import iter33 as D
import iter34 as A

W, S, Q, R, C, I, B = P.W, P.S, P.Q, P.R, P.C, P.I, P.B
torch.set_grad_enabled(False)
DEFAULT_CKPT = I.ROOT/'ckpt/local_s0_p15.pt'
BRANCH_STEPS, WARMUP_STEPS, DIAG_BATCHES, VAL_WINDOWS = 64, 32, 4, 32


def device_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device).clone()
    if isinstance(value, dict):
        return {k: device_tree(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [device_tree(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(device_tree(v, device) for v in value)
    return copy.deepcopy(value)


def make_checkpoint(path=DEFAULT_CKPT):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    assert torch.cuda.is_available() and not torch.is_grad_enabled()
    torch.manual_seed(0)
    np.random.seed(0)
    tr, _, V = B.load()
    train = tr[:int(len(tr)*.95)]
    total = math.ceil(len(train)/(16*128))
    target = math.ceil(total*.15)
    net, gate = P.Model(V, total, 'local'), C.Gate(V)
    x = torch.as_tensor(tr[:128].astype(np.int64), device='cuda')[None]
    y = torch.as_tensor(tr[1:129].astype(np.int64), device='cuda')[None]
    checks = P.checks(net, gate, x, y, 0)
    sampler = A.Sampler(train, 0, V)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(target):
        net.train(*next(sampler), gate)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    assert net.step == gate.step == target and net.total_steps == total
    assert net.lr == S.schedule(target, total)
    A.save_checkpoint(path, net, gate, sampler, train, 0)
    result = dict(checkpoint=str(path), steps=target, total_steps=total,
                  actual_fraction=target/total, train_seconds=seconds,
                  schedule='full training partition and total; no compression', checks=checks)
    print(json.dumps(result, indent=2), flush=True)
    return result


def branch_model(payload, candidate, V, hid, alpha):
    # Fresh construction is RNG-isolated and cannot affect batches or branches.
    # The original iter27 parameters and moments are then restored exactly.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(payload['seed'])
        total = payload['model']['total_steps']
        if candidate == 'auxhead':
            net = A.Model(V, total, 'auxhead', hid)
        elif candidate == 'topdown':
            net = D.Model(V, total, 'topdown', alpha)
        else:
            net = P.Model(V, total, 'local')
        gate = C.Gate(V)
    original = device_tree(payload['model'], 'cuda')
    # Preserve fresh aux tensors, their zero moments and their independent clock.
    net.p.update(original.pop('p'))
    net.state.update(original.pop('state'))
    original.pop('mode', None)
    for key, value in original.items():
        setattr(net, key, value)
    vars(gate).update(device_tree(payload['gate'], 'cuda'))
    return net, gate


def local_gradients(net, candidate, x, y):
    hs, zs, us, ids, em = net.forward(x)
    M = net._prepare(x)[0]
    grads = {}
    for l in (0, 1):
        args = (l, x, y, hs[l], zs[l], us[l-1] if l else None, ids, em, M)
        if candidate == 'auxhead':
            layer, _ = net._candidate_grads(*args)
        elif candidate == 'topdown':
            a = D.teaching_weight(net.step+1, net.total_steps, net.alpha)
            layer = net._teaching_grads(*args, D.teacher(zs[-1]), a)
        else:
            layer = net._local_grads(*args)
        grads.update(layer)
    return grads


def diagnostics(net, candidate, batches, label):
    """Backbone coordinates only, raw negative gradient vs raw negative BP.

    The BP reference explicitly uses iter30.Model: no local detach boundaries,
    auxiliary CE, copy loss or mixed CE enter its summed three-readout loss.
    A fresh zero-output head has zero backbone direction: cosine is undefined.
    """
    baseline = copy.copy(net)
    baseline.__class__, baseline.mode = P.Model, 'backprop'
    baseline.p = {k: v for k, v in net.p.items() if not k.startswith('aux')}
    baseline.state = {k: net.state[k] for k in baseline.p}
    before = I.stamp(net)
    records = []
    for index, (x, y) in enumerate(batches[:DIAG_BATCHES]):
        bp, _ = P.backprop_grads(baseline, x, y)
        local = local_gradients(net, candidate, x, y)
        layers = {}
        for l in (0, 1):
            keys = sorted(A.backbone_keys(l))
            # Both update directions are negative gradients, so signs cancel.
            dot = sum((local[k].double()*bp[k].double()).sum() for k in keys)
            cn = sum(local[k].double().square().sum() for k in keys).sqrt().item()
            bn = sum(bp[k].double().square().sum() for k in keys).sqrt().item()
            layers[str(l)] = dict(cosine=dot.item()/(cn*bn) if cn > 0 and bn > 0 else None,
                                  norm_ratio=cn/bn if bn > 0 else None,
                                  candidate_norm=cn, bp_norm=bn)
        record = dict(batch=index, layers=layers)
        records.append(record)
        print(json.dumps(dict(diagnostic=label, **record)), flush=True)
    assert I.stamp(net) == before, 'diagnostic modified branch state'
    return records


def evaluate(net, gate, windows):
    before = I.stamp((net, gate))
    totals = dict(top=0., mixed=0., mixed_nocopy=0.)
    x, y = windows
    for a in range(0, len(x), 16):
        xb, yb = x[a:a+16], y[a:a+16]
        zs = net.forward(xb)[1]
        mix = sum(w*z for w, z in zip(net.p['g'], zs))
        logcopy = gate.predict(xb, mix.softmax(-1))[0]
        count = yb.numel()
        totals['top'] += P.ce(zs[-1], yb).item()*count
        totals['mixed'] += P.ce(logcopy, yb).item()*count
        totals['mixed_nocopy'] += P.ce(mix, yb).item()*count
    assert I.stamp((net, gate)) == before, 'screen evaluation modified state'
    return {key: value/y.numel() for key, value in totals.items()}


def screen(candidate, path=DEFAULT_CKPT, hid=64, alpha=.5):
    if candidate not in ('local', 'auxhead', 'topdown'):
        raise ValueError(candidate)
    if hid < 1 or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('positive hid and alpha in [0,1] required')
    output = I.ROOT/f'results/screen34_{candidate}.json'
    if output.exists():
        raise FileExistsError(output)
    assert torch.cuda.is_available() and not torch.is_grad_enabled()
    torch.cuda.synchronize()
    start = time.perf_counter()
    # Checkpoints contain tensors and primitive containers, no pickled models.
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    if payload['format'] != 'iter34_v1' or payload['seed'] != 0:
        raise ValueError('expected iter34-format local seed-0 checkpoint')
    tr, _, V = B.load()
    cut = int(len(tr)*.95)
    train = tr[:cut]
    if (payload['train_chars'] != len(train) or
            payload['train_sha256'] != hashlib.sha256(train.tobytes()).hexdigest()):
        raise ValueError('checkpoint training partition differs from current data')
    total = math.ceil(len(train)/(16*128))
    step = payload['model']['step']
    if (payload['model']['mode'] != 'local' or
            payload['model']['total_steps'] != total or step != math.ceil(total*.15)):
        raise ValueError('expected original-schedule iter30 local 15% checkpoint')
    sampler = A.Sampler(train, 0, V, payload['sampler'])
    idle = WARMUP_STEPS if candidate == 'auxhead' else 0
    # One materialization: warmup, diagnostics and both branches share tensors.
    batches = [next(sampler) for _ in range(idle+BRANCH_STEPS)]
    # Take the first 32 of the normal fixed validation windows, without respacing.
    vx, vy = B.eval_windows(tr[cut:])
    windows = vx[:VAL_WINDOWS].cuda(), vy[:VAL_WINDOWS].cuda()
    del vx, vy, sampler
    net, gate = branch_model(payload, candidate, V, hid, alpha)
    base, bg = branch_model(payload, 'local', V, hid, alpha)
    checkpoint_eval = evaluate(base, bg, windows)
    diag = diagnostics(net, candidate, batches, 'at_checkpoint_fresh_heads')
    warm_seconds = 0.
    post_warm = None
    if idle:
        frozen = I.stamp(({k: (net.p[k], net.state[k]) for k in base.p}, gate))
        torch.cuda.synchronize()
        warm_start = time.perf_counter()
        for x, y in batches[:idle]:
            net.warmup_heads(x, y)
        torch.cuda.synchronize()
        warm_seconds = time.perf_counter()-warm_start
        assert I.stamp(({k: (net.p[k], net.state[k]) for k in base.p}, gate)) == frozen
        assert net.step == base.step == step and gate.step == bg.step == step
        # Backbone still at the checkpoint; warmed heads give useful directions
        # in addition to the mandated (zero-direction) fresh-head diagnostic.
        post_warm = diagnostics(net, candidate, batches, 'at_checkpoint_warmed_heads')
    branch_results, branch_seconds = {}, {}
    for label, model, copy_gate in (('local', base, bg), ('candidate', net, gate)):
        # The baseline has already consumed the same idle batches via the shared
        # materialization; neither its parameters nor its Adam states were touched.
        torch.cuda.synchronize()
        branch_start = time.perf_counter()
        for x, y in batches[idle:]:
            model.train(x, y, copy_gate)
        torch.cuda.synchronize()
        branch_seconds[label] = time.perf_counter()-branch_start
        assert model.step == copy_gate.step == step+BRANCH_STEPS
        assert model.lr == S.schedule(step+BRANCH_STEPS, total)
        branch_results[label] = evaluate(model, copy_gate, windows)
    local, chosen = branch_results['local'], branch_results['candidate']
    delta = {key: chosen[key]-local[key] for key in chosen}
    improvement = {label: {key: values[key]-checkpoint_eval[key] for key in values}
                   for label, values in branch_results.items()}
    torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    result = dict(candidate=candidate, checkpoint=str(path), checkpoint_step=step,
                  total_steps=total, branch_steps=BRANCH_STEPS, head_warmup_steps=idle,
                  local_idle_batches=idle, val_windows=VAL_WINDOWS, hid=hid, alpha=alpha,
                  checkpoint_losses=checkpoint_eval, local=local, candidate_losses=chosen,
                  delta_candidate_minus_local=delta, change_from_checkpoint=improvement,
                  diagnostics_at_checkpoint=diag, diagnostics_after_head_warmup=post_warm,
                  wall_seconds=seconds, warmup_seconds=warm_seconds, branch_seconds=branch_seconds,
                  settings=dict(seed=0, bs=16, T=128, diagnostic_batches=DIAG_BATCHES,
                                diagnostic_parameters='each lower backbone, excludes inference readouts and auxiliary heads',
                                diagnostic_direction='negative raw gradients, before Adam and clipping',
                                bp_objective='summed three iter30 inference readout CE, fully flowing',
                                delta_sign='negative is better',
                                mixed_loss='final network g mixture plus suffix copy gate',
                                validation='first 32 original fixed val windows, no respacing',
                                warmup='head-only; frozen backbone and gate; next scheduled LR held; fresh head Adam',
                                schedule='64 original schedule steps starting at checkpoint.step+1; no compression',
                                data='identical branch batches; aux warmup/local idle consume identical extra 32 batches',
                                wall_scope='load, batch preparation, diagnostics, warmup, both branches and evals; excludes checkpoint creation'))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--make-ckpt', action='store_true')
    parser.add_argument('--candidate', choices=('local', 'auxhead', 'topdown'))
    parser.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT)
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=.5)
    args = parser.parse_args()
    if not args.make_ckpt and args.candidate is None:
        parser.error('specify --make-ckpt and/or --candidate')
    if args.make_ckpt:
        make_checkpoint(args.ckpt)
    if args.candidate is not None:
        screen(args.candidate, args.ckpt, args.hid, args.alpha)
