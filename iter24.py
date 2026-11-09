"""iter22 with the network LR held for 75% of steps, then cosine-decayed to 1e-4."""
import argparse, copy, json, math, tempfile, time
from pathlib import Path
import numpy as np
import torch
import iter22 as S
Q, R, C, I, B = S.Q, S.R, S.C, S.I, S.B
torch.set_grad_enabled(False)
TOP, BOTTOM, HOLD = 1e-3, 1e-4, .75


def schedule(s, total):
    """Network LR for 1-indexed step s of total."""
    assert 1 <= s <= total
    if s <= HOLD*total: return TOP
    return BOTTOM+(TOP-BOTTOM)*.5*(1+math.cos(math.pi*(s-HOLD*total)/((1-HOLD)*total)))


class Model(S.Model):
    def __init__(self, V, total_steps):
        super().__init__(V); self.total_steps = total_steps

    def train(self, x, y, gate):
        self.lr = schedule(self.step+1, self.total_steps)
        return super().train(x, y, gate)


def same(a, b, ag, bg):
    assert a.step == b.step and ag.step == bg.step
    for key in a.p:
        torch.testing.assert_close(a.p[key], b.p[key], atol=1e-6, rtol=0)
        for x, y in zip(a.state[key], b.state[key]): torch.testing.assert_close(x, y, atol=1e-6, rtol=0)
    for key in ('A', 'm', 'v'): torch.testing.assert_close(getattr(ag, key), getattr(bg, key), atol=1e-6, rtol=0)


def checks(net, gate, x, y, train):
    before = I.stamp((net, gate)); assert net.lr == TOP and net.step == gate.step == 0
    result = S.checks(net, gate, x, y, train)
    # Schedule shape: hand-computed cases, then the real step count.
    assert schedule(1, 8) == schedule(6, 8) == TOP and schedule(8, 8) == BOTTOM and abs(schedule(7, 8)-5.5e-4) < 1e-15
    n = net.total_steps; hold = math.floor(HOLD*n); assert 3 <= hold < n
    lrs = [schedule(s, n) for s in range(1, n+1)]
    assert all(v == TOP for v in lrs[:hold]) and lrs[hold] < TOP and lrs[-1] == BOTTOM
    assert all(a > b for a, b in zip(lrs[hold:], lrs[hold+1:]))
    # While held, training equals iter22 exactly.
    batches = list(R.epoch(train[:6141], 0, min(len(train), 6141), net.V)); assert len(batches) == 3
    a, b = copy.deepcopy(net), copy.deepcopy(net); ag, bg = copy.deepcopy(gate), copy.deepcopy(gate)
    for bx, by in batches:
        assert abs(a.train(bx, by, ag)-S.Model.train(b, bx, by, bg)) <= 1e-6 and a.lr == b.lr == TOP
        same(a, b, ag, bg)
    del a, b, ag, bg
    # In decay, the scheduled LR is what the update uses, g included; the gate does not see it.
    bx, by = batches[0]; applied = {}
    for s, want in ((hold+1, lrs[hold]), (n, BOTTOM)):
        a, b, c = (copy.deepcopy(net) for _ in range(3)); ag, bg, cg = (copy.deepcopy(gate) for _ in range(3))
        a.step = b.step = c.step = s-1; b.lr = want; c.lr = TOP
        a.train(bx, by, ag); S.Model.train(b, bx, by, bg); S.Model.train(c, bx, by, cg)
        assert a.lr == want and a.step == s
        same(a, b, ag, bg)
        for key in ('A', 'm', 'v'): torch.testing.assert_close(getattr(ag, key), getattr(cg, key), atol=1e-6, rtol=0)
        if s == n:
            for key in ('b0', 'E', 'W2', 'g'): assert not torch.equal(a.p[key], c.p[key]), key
        applied[s] = want
        Q.plain((a, b, c, ag, bg, cg)); del a, b, c, ag, bg, cg
    assert I.stamp((net, gate)) == before, 'checks wrote model state'
    result.update(schedule_values=True, hold_steps=hold, decay_steps=n-hold, equals_iter22_while_held=True,
                  scheduled_lr_applied=applied, gate_lr_independent=True, state_untouched_iter24=True)
    return result


def run(smoke=None, seed=0):
    if smoke is not None and smoke < 1: raise ValueError('--smoke must be positive')
    if seed < 0: raise ValueError('--seed must be non-negative')
    temporary = tempfile.TemporaryDirectory(prefix='iter24_') if smoke is not None else None
    try:
        name = 'iter24_decay.json' if seed == 0 else f'iter24_decay_seed{seed}.json'
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
        assert chars == len(train) and net.step == gate.step == total and net.lr == BOTTOM
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
        settings = dict(seed=seed, device='cuda', mode='copy', variant='lagged', base='iter22, no auxiliary', bs=16, T=128, passes=1,
                        train_end=cut, val_start=cut, training_partition_chars=len(train), smoke=smoke, eval_windows=16 if smoke is not None else old_windows,
                        sampling=f'non-overlapping target windows, shuffled with CPU generator seed {seed}', renaming=info, evaluation_frozen=True,
                        D=256, layers=3, embedding_width=32, lags=list(range(6)), decays=[.5,.8,.95], layer_input_width=1536, state_lags=[1,2],
                        lr=TOP, lr_final=BOTTOM, lr_hold_fraction=HOLD, lr_schedule='constant for s <= .75*S, then cosine to lr_final at s = S (1-indexed)',
                        lr_applies_to='all network parameters including g', optimizer='manual Adam', betas=[.9,.999], eps=1e-8, gradient_clip_norm=1, weight_decay=0,
                        update_order='per-layer during forward sweep; gate and g after', suffix_lengths=C.LENS, gate_lr=1e-2, gate_lr_schedule='constant')
        result.update(test_loss=test, renamed_loss=ren, renamed_gap=ren-test, val_loss=val, g=net.p['g'].tolist(), train_seconds=seconds,
                      chars=chars, steps=net.step, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, settings=settings,
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
