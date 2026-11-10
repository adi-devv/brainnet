"""Bounded one-shot memory, with optional independently learned structural keys.

Every chunk is scored with old parameters/counts, then learned/written once.
No autograd is used, including in the small keys module (its own exact CE
derivative is written explicitly). Evaluation uses iter30's actual functions.
Smoke results are retained, explicitly marked ineligible for the full-data bars.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import copy
import gc
import json
import math
import time
from collections import Counter

import numpy as np
import torch
import iter30 as CONTROL
import iter10 as CLOCK

I, B, Q, C = CONTROL.I, CONTROL.B, CONTROL.Q, CONTROL.C
torch.set_grad_enabled(False)
P = 1000003
FLOOR = 1e-6
RAW_ORDERS = (1, 2, 3, 4, 8, 16, 32)
KEY_BITS = (8, 12, 16)
TARGETS = dict(test=1.2562, renamed=1.5714, val=1.3869,
               train_seconds=79.4, peak_allocated_mib=151.)


def nbytes(x):
    return x.numel()*x.element_size()


def shift(x, n, pad):
    if not n:
        return x
    return torch.cat([torch.full((len(x), n), pad, dtype=x.dtype,
                                 device=x.device), x], 1)[:, :x.shape[1]]


def hashes(x, pad, orders):
    # Same reversed polynomial convention as iter5.Counts.codes.
    h = torch.zeros_like(x)
    out = {}
    for j in range(1, max(orders)+1):
        h = h*P+shift(x, j-1, pad)+1
        if j in orders:
            out[j] = h.clone()
    return out


class Dense:
    """Exact small context space; int32 cannot overflow in one <2^31-char pass."""
    def __init__(self, rows, V, device):
        self.V = V
        self.count = torch.zeros((rows, V), dtype=torch.int32, device=device)

    def read(self, code):
        return self.count[code].float()

    def write(self, code, y, scratch=None):
        self.count.view(-1).index_add_(0, code*self.V+y,
                                      torch.ones_like(y, dtype=torch.int32))

    def memory(self):
        return dict(counts=nbytes(self.count))


class Tagged:
    """One direct-mapped (context,next) cache: int32 count + 31-bit fingerprint.

    Reads ignore mismatched fingerprints. Chunk collisions elect the largest
    fingerprint, independent of scatter scheduling, then add only its writes.
    There are no conflicting-value advanced-index assignments. The shared
    scratch array is bounded by the largest table, and only touched cells reset.
    This is deliberately a lossy cache, not an allegedly exact count-min sketch.
    """
    def __init__(self, bits, V, device):
        self.V, self.mask = V, (1 << bits)-1
        self.count = torch.zeros(1 << bits, dtype=torch.int32, device=device)
        self.tag = torch.zeros_like(self.count)
        self.chars = torch.arange(V, device=device)

    def address(self, h, y):
        ix, tag = I.slot(h*P+y+1)
        return ix & self.mask, tag

    def read(self, code):
        ix, tag = self.address(code[:, None], self.chars)
        return (self.count[ix]*(self.tag[ix] == tag)).float()

    def commit(self, ix, tag, scratch):
        scratch.index_fill_(0, ix, 0)
        scratch.scatter_reduce_(0, ix, tag, reduce='amax', include_self=True)
        winner = scratch[ix]
        changed = self.tag[ix] != winner
        self.count.index_fill_(0, ix[changed], 0)
        self.tag.scatter_(0, ix, winner)  # duplicate indices have identical values
        keep = tag == winner
        self.count.index_add_(0, ix[keep], torch.ones_like(tag[keep]))

    def write(self, code, y, scratch):
        self.commit(*self.address(code, y), scratch)

    def memory(self):
        return dict(counts=nbytes(self.count), tags=nbytes(self.tag),
                    character_indices=nbytes(self.chars))


class Adam:
    """Plain tensors only. No optimizer or autograd graph."""
    def __init__(self, params):
        self.p = params
        self.m = {k: torch.zeros_like(v) for k, v in params.items()}
        self.v = {k: torch.zeros_like(v) for k, v in params.items()}
        self.step = 0

    def update(self, grads, lr):
        self.step += 1
        for k, g in grads.items():
            g = g/g.norm().clamp_min(1)
            self.m[k].mul_(.9).add_(g, alpha=.1)
            self.v[k].mul_(.999).addcmul_(g, g, value=.001)
            self.p[k].addcdiv_(self.m[k]/(1-.9**self.step),
                              (self.v[k]/(1-.999**self.step)).sqrt().add_(1e-8),
                              value=-lr)

    def memory(self):
        return {f'{kind}.{k}': nbytes(v) for kind, tensors in
                (('parameters', self.p), ('adam_m', self.m), ('adam_v', self.v))
                for k, v in tensors.items()}


class LocalMixer:
    """hybrid_mixer log-linear pool, with char/pair rows and confidence slopes.

    A local CE updates only the mixer. Experts and confidence features are
    constants to this rule; the keys net never receives this loss.
    """
    def __init__(self, V, experts, device):
        self.E = experts
        self.opt = Adam(dict(char=torch.full((V+1, experts), 1/experts, device=device),
                             pair=torch.zeros(512, experts, device=device),
                             confidence=torch.zeros(experts, 2, device=device)))

    def forward(self, lp, counts, selectors):
        ch, pair = selectors
        phi = torch.stack([counts.log1p().clamp_max(12)/12, lp.amax(-1).exp()], -1)
        p = self.opt.p
        weights = p['char'][ch]+p['pair'][pair]+(phi*p['confidence']).sum(-1)
        q = (weights[..., None]*lp).sum(1).log_softmax(-1)
        return q, phi

    def gradients(self, lp, q, phi, selectors, y):
        valid = y >= 0
        actual = lp.gather(-1, y.clamp_min(0)[:, None, None].expand(-1, self.E, 1))[..., 0]
        expected = (q.exp()[:, None, :]*lp).sum(-1)
        # Negative CE derivative wrt the independent log-linear weights.
        d = (expected-actual)*valid[:, None]/valid.sum().clamp_min(1)
        ch, pair = selectors
        return dict(char=torch.zeros_like(self.opt.p['char']).index_add_(0, ch, d),
                    pair=torch.zeros_like(self.opt.p['pair']).index_add_(0, pair, d),
                    confidence=(d[..., None]*phi).sum(0))


class KeysNet:
    """A 96-unit, one-module local learner, smaller than iter27 layer 0.

    Six additive class lag lookups + three causal EMA projections. Identifier
    chars share a canonical input symbol, including upper/lowercase/digits/_;
    length-preserving identifier renaming therefore preserves the input codes.
    Predicts the original 96-way next char with its OWN CE. Fixed random sign
    projections of its old h make 8-, 12- and 16-bit memory addresses. Learning
    changes codes over time; this representation drift is not hidden or replayed.
    """
    D, EMB, LAGS = 96, 16, 6

    def __init__(self, V, device):
        self.V = V
        p = {f'A{j}': torch.randn(V+2, self.D, device=device)/math.sqrt(6*self.D)
             for j in range(self.LAGS)}
        p.update({f'P{k}': torch.randn(self.EMB, self.D, device=device)/math.sqrt(3*self.EMB)
                  for k in range(3)})
        p.update(b=torch.zeros(self.D, device=device),
                 R=torch.randn(self.D, V, device=device)/math.sqrt(self.D),
                 c=torch.zeros(V, device=device))
        self.opt = Adam(p)
        self.embedding = torch.randn(V+2, self.EMB, device=device)/math.sqrt(self.EMB)
        self.embedding[V] = 0
        self.projection = torch.randn(self.D, max(KEY_BITS), device=device)/math.sqrt(self.D)
        t = torch.arange(128, device=device)
        distance = t[:, None]-t[None, :]
        self.M = torch.stack([(1-d)*d**distance.clamp_min(0)*(distance >= 0)
                              for d in (.5, .8, .95)])
        self.bitweights = 2**torch.arange(max(KEY_BITS), device=device)

    def forward(self, canonical):
        T = canonical.shape[1]
        ids = [shift(canonical, j, self.V) for j in range(self.LAGS)]
        em = self.M[:, :T, :T] @ self.embedding[canonical][:, None]
        p = self.opt.p
        h = (sum(p[f'A{j}'][idx] for j, idx in enumerate(ids))+
             sum(em[:, k] @ p[f'P{k}'] for k in range(3))+p['b']).tanh()
        z = h @ p['R']+p['c']
        bits = ((h @ self.projection) > 0).long()*self.bitweights
        codes = [bits[..., :n].sum(-1) for n in KEY_BITS]
        return h, z, ids, em, codes

    def gradients(self, cache, y):
        h, z, ids, em, _ = cache
        e = z.softmax(-1)
        e.scatter_add_(-1, y.clamp_min(0)[..., None], -torch.ones_like(e[..., :1]))
        e *= (y >= 0)[..., None]/(y >= 0).sum().clamp_min(1)
        d = (e @ self.opt.p['R'].T)*(1-h.square())
        hf, ef, df = (a.flatten(0, 1) for a in (h, e, d))
        grads = dict(R=hf.T @ ef, c=ef.sum(0), b=df.sum(0))
        grads.update({f'A{j}': torch.zeros_like(self.opt.p[f'A{j}']).index_add_(0, idx.flatten(), df)
                      for j, idx in enumerate(ids)})
        grads.update({f'P{k}': em[:, k].flatten(0, 1).T @ df for k in range(3)})
        return grads

    def memory(self):
        return dict(**self.opt.memory(), embedding=nbytes(self.embedding),
                    projection=nbytes(self.projection), ema_matrix=nbytes(self.M),
                    bitweights=nbytes(self.bitweights))


class Model:
    def __init__(self, vocab, mode, device='cuda', tiny=False):
        self.V, self.mode, self.device = len(vocab), mode, device
        V = self.V
        self.word = torch.tensor([c.isascii() and (c.isalnum() or c == '_')
                                  for c in vocab]+[False], device=device)
        self.tables = {'raw1': Dense(V+1, V, device),
                       'raw2': Dense((V+1)**2, V, device)}
        for n in RAW_ORDERS[2:]:
            self.tables[f'raw{n}'] = Tagged(16 if tiny else (19 if n == 3 else 20), V, device)
        for name in ('wordshape', 'clock1', 'clock3'):
            self.tables[name] = Tagged(16 if tiny else 18, V, device)
        self.net = KeysNet(V, device) if mode == 'keys' else None
        if self.net:
            for n in KEY_BITS:
                self.tables[f'key{n}'] = (Dense(1 << n, V, device) if n <= 12
                                          else Tagged(16 if tiny else 18, V, device))
        max_slots = max(t.count.numel() for t in self.tables.values() if isinstance(t, Tagged))
        self.scratch = torch.zeros(max_slots, dtype=torch.int32, device=device)
        self.uni = torch.zeros(V, dtype=torch.int32, device=device)
        self.expert_names = ['unigram', *self.tables, *(['local_net'] if self.net else []),
                             *[f'copy{n}' for n in C.LENS]]
        self.mixer = LocalMixer(V, len(self.expert_names), device)

    def contexts(self, x):
        V = self.V
        prev = shift(x, 1, V)
        raw = hashes(x, V, RAW_ORDERS[2:])
        codes = dict(raw1=x, raw2=x+(V+1)*prev)
        codes.update({f'raw{n}': h for n, h in raw.items()})
        canonical = torch.where(self.word[x], V+1, x)
        shape = hashes(canonical, V, (8,))[8]
        pos = torch.arange(x.shape[1], device=x.device)[None].expand_as(x)
        start = torch.where(~self.word[x] | ~self.word[prev], pos, 0).cummax(-1).values
        length = (pos-start+1).clamp_max(15)
        # Current normalized identifier shape, run length and preceding delimiter.
        delimiter = x.gather(1, (start-1).clamp_min(0))
        delimiter = torch.where(start > 0, delimiter, V)
        codes['wordshape'] = (shape*P+length)*P+delimiter+1
        # Surprise of CURRENT observed x from the OLD exact bigram table.
        # Never use y[t], or a score at a later input position, to form t's key.
        context = prev+(V+1)*shift(x, 2, V)
        hist = self.tables['raw2'].count
        flat = context.flatten()
        denom = hist[flat].sum(-1).reshape_as(x).float()
        num = hist[flat, x.clamp_max(V-1).flatten()].reshape_as(x).float()
        surprise = -((num+.5/V)/(denom+.5)).log()
        fire = torch.stack([surprise > 1., surprise > 3.], 1)
        clock, _, _ = CLOCK.clocked(x, fire, V)
        codes.update(clock1=clock[2], clock3=clock[5])
        return codes, canonical, (x.flatten(), ((prev*P+x) % 512).flatten())

    def score(self, x, copy_enabled=True):
        codes, canonical, selectors = self.contexts(x)
        cache = self.net.forward(canonical) if self.net else None
        if cache:
            codes.update({f'key{n}': code for n, code in zip(KEY_BITS, cache[-1])})
        codes = {k: v.flatten() for k, v in codes.items()}
        N = x.numel()
        p = ((self.uni.float()+1)/(self.uni.sum()+self.V)).expand(N, -1)
        ps, totals = [p], [self.uni.sum().float().expand(N)]
        high = None
        for name, table in self.tables.items():
            c = table.read(codes[name])
            total = c.sum(-1)
            if name.startswith('raw'):
                p = (c+.5*p)/(total[:, None]+.5)
                high = p
                candidate = p
            else:
                # Extra contexts are independent proposals, backed by the same
                # lexical expert, not accidentally chained through other extras.
                candidate = (c+.5*high)/(total[:, None]+.5)
            ps.append(candidate)
            totals.append(total)
        if cache:
            ps.append(cache[1].flatten(0, 1).softmax(-1))
            totals.append(torch.zeros(N, device=x.device))
        heads, ns = C.heads(x, self.V)
        for i in range(len(C.LENS)):
            n = ns[..., i].flatten()
            hp = heads[..., i, :].flatten(0, 1)
            proposal = (hp*n[:, None]+.5*high)/(n[:, None]+.5)
            ps.append(proposal if copy_enabled else high)
            totals.append(n if copy_enabled else torch.zeros_like(n))
        lp = torch.stack(ps, 1)
        del ps, heads, hp, proposal, candidate, c, p
        lp.clamp_min_(FLOOR).log_()
        counts = torch.stack(totals, 1)
        q, phi = self.mixer.forward(lp, counts, selectors)
        return q.reshape(*x.shape, self.V), (codes, cache, lp, phi, selectors)

    def logprobs(self, x, copy=True):
        return self.score(x, copy)[0]

    def learn_write(self, x, y, q, state, lr):
        codes, cache, lp, phi, selectors = state
        # All scores, clock decisions, h and codes were materialized before ANY
        # learner/count write. The mixer cannot send a derivative into the net.
        self.mixer.opt.update(self.mixer.gradients(lp, q.flatten(0, 1), phi,
                                                  selectors, y.flatten()), .005)
        if cache:
            self.net.opt.update(self.net.gradients(cache, y), lr)
        keep = y.flatten() >= 0
        target = y.flatten()[keep]
        for name, table in self.tables.items():
            table.write(codes[name][keep], target, self.scratch)
        self.uni.index_add_(0, target, torch.ones_like(target, dtype=torch.int32))

    def memory(self):
        out = {name: table.memory() for name, table in self.tables.items()}
        out.update(unigram=dict(counts=nbytes(self.uni)),
                   shared_write_scratch=dict(indices=nbytes(self.scratch)),
                   character_classes=dict(word=nbytes(self.word)),
                   mixer=self.mixer.opt.memory())
        if self.net:
            out['keys_net'] = self.net.memory()
        out['total_persistent'] = sum(sum(structure.values()) for structure in out.values())
        return out


def epoch_cpu(tr, seed, chunk_size, V):
    """Same target partition/order as iter17.epoch, streamed from CPU.

    Grouping windows into chunks changes memory visibility, not the order or
    multiplicity of targets. Unlike iter17, no full int64 training copy lives on
    the GPU. Sub-window chunks still get their whole causal input window.
    """
    starts = (torch.randperm(math.ceil(len(tr)/128),
                            generator=torch.Generator().manual_seed(seed))*128).numpy()
    bs = max(1, chunk_size//128)
    offsets = np.arange(128)
    for a in range(0, len(starts), bs):
        ix = starts[a:a+bs, None]+offsets
        x = np.where(ix > 0, tr[np.clip(ix-1, 0, len(tr)-1)], V).astype(np.int64)
        y = np.where(ix < len(tr), tr[ix.clip(max=len(tr)-1)].astype(np.int64), -1)
        yield torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()


def stale_contexts(codes, y, accumulator):
    # Semantic staleness, not a guess from chunk size: count each later exact
    # context occurrence within the chunk. Hash eviction interference is separate.
    valid = y.flatten() >= 0
    N = valid.sum()
    any_repeat = torch.zeros_like(valid)
    for name, code in codes.items():
        selected = code[valid]
        _, inverse = selected.unique(return_inverse=True)
        pos = torch.arange(len(selected), device=y.device)
        first = torch.full_like(pos, len(selected))
        first.scatter_reduce_(0, inverse, pos, reduce='amin', include_self=True)
        repeated = pos > first[inverse]
        accumulator[name] += repeated.sum()
        any_repeat[valid] |= repeated
    accumulator['any_context'] += any_repeat.sum()
    # The unigram is also memory: every target except the first sees old counts.
    accumulator['including_unigram'] += (N-1).clamp_min(0)


def flops(V, mode):
    # Count a dense m*k by k*n product as 2*m*k*n. Include full causal matrices
    # because the reference actually executes dense, zero-masked products.
    T = 128
    bp_forward = dict(layer0_ema=2*3*T*32, layer0_projection=2*3*32*256,
                      layer0_readout=2*256*V,
                      layer1_ema=2*3*T*256, layer1_projection=2*1536*256,
                      layer1_readout=2*256*V,
                      layer2_ema=2*3*T*256, layer2_projection=2*1536*320,
                      layer2_readout=2*320*V)
    E = 15 if mode == 'tables' else 19
    ours = dict(logistic_pool_and_local_derivative=4*E*V,
                confidence_and_weight_derivatives=12*E)
    if mode == 'keys':
        # Fixed embedding EMA has no derivative; learned P/R have two products
        # in their local derivative in addition to their forward product.
        ours.update(fixed_class_ema=2*3*T*KeysNet.EMB,
                    local_projection_and_derivative=3*2*3*KeysNet.EMB*KeysNet.D,
                    local_readout_and_derivative=3*2*KeysNet.D*V,
                    sign_projection=2*KeysNet.D*max(KEY_BITS))
    matmul = sum(ours.values())
    # Conservative scalar floating arithmetic allowance; hashing, comparisons,
    # integer adds and memory traffic are explicitly not FLOPs.
    elementwise = 16*E*V + (24*KeysNet.D if mode == 'keys' else 0)
    return dict(iter35=matmul+elementwise, iter35_matmul=matmul,
                iter35_elementwise_allowance=elementwise, iter35_by_structure=ours,
                iter30_backprop=3*sum(bp_forward.values()),
                iter30_bp_forward_by_structure=bp_forward,
                convention='2 FLOPs per multiply-add; BP = 3x actual dense forward matmuls; '
                           'copy heads use comparison/scatter, no matmul; integer hashing, '
                           'atomics, exp/log, optimizer and CPU operations not counted; '
                           'iter35 adds a scalar arithmetic allowance; this is not wall time')


def finite_difference_checks(model, x):
    # Exercise the actual manual derivatives without enabling grad mode at all.
    xd = x[:, :6]
    y = (xd+1) % model.V
    y[0, -1] = -1
    probe = copy.deepcopy(model.mixer)
    probe.opt.p = {k: p.double() for k, p in probe.opt.p.items()}
    E = probe.E
    lp = torch.arange(xd.numel()*E*model.V, device=x.device, dtype=torch.float64)
    lp = lp.reshape(xd.numel(), E, model.V).sin().log_softmax(-1)
    counts = torch.ones(xd.numel(), E, device=x.device, dtype=torch.float64)
    selectors = xd.flatten(), torch.zeros(xd.numel(), dtype=torch.long, device=x.device)
    q, phi = probe.forward(lp, counts, selectors)
    analytic = probe.gradients(lp, q, phi, selectors, y.flatten())
    def ce():
        q = probe.forward(lp, counts, selectors)[0]
        return -q.gather(-1, y.flatten().clamp_min(0)[:, None])[y.flatten() >= 0].mean()
    max_error = 0.
    for k, idx in (('char', (int(xd[0, 0]), 0)), ('pair', (0, E-1)),
                   ('confidence', (1, 0)), ('confidence', (2, 1))):
        v = probe.opt.p[k]
        old, eps = v[idx].item(), 1e-5
        v[idx] = old+eps
        plus = ce()
        v[idx] = old-eps
        minus = ce()
        v[idx] = old
        error = abs(float(analytic[k][idx]-(plus-minus)/(2*eps)))
        assert error < 1e-7, ('mixer derivative', k, error)
        max_error = max(max_error, error)
    if model.net:
        probe_net = copy.deepcopy(model.net)
        probe_net.opt.p = {k: p.double() for k, p in probe_net.opt.p.items()}
        for k in ('embedding', 'projection', 'M'):
            setattr(probe_net, k, getattr(probe_net, k).double())
        canonical = torch.where(model.word[xd], model.V+1, xd)
        grads = probe_net.gradients(probe_net.forward(canonical), y)
        def local_ce():
            return torch.nn.functional.cross_entropy(probe_net.forward(canonical)[1].flatten(0, 1),
                                                     y.flatten(), ignore_index=-1)
        for k, idx in (('A0', (int(canonical[0, 0]), 3)), ('P1', (2, 3)),
                       ('R', (3, 2)), ('b', (3,)), ('c', (2,))):
            v = probe_net.opt.p[k]
            old, eps = v[idx].item(), 1e-5
            v[idx] = old+eps
            plus = local_ce()
            v[idx] = old-eps
            minus = local_ce()
            v[idx] = old
            error = abs(float(grads[k][idx]-(plus-minus)/(2*eps)))
            assert error < 1e-7, ('local CE derivative', k, error)
            max_error = max(max_error, error)
    return max_error


def checks(vocab, mode, device):
    # Small independent models; preserve both RNGs before constructing the run.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(35)
        m = Model(vocab, mode, device, tiny=True)
        V = m.V
        seq = [i % min(4, V) for i in (0, 1, 2, 0, 1, 2, 0, 1, 3, 0, 1, 2)]
        x = torch.tensor([seq], device=device)
        # Verify EACH read/write with chunk size 1 against exact context dicts,
        # then verify all live slot counts, not just the observed target count.
        refs = {name: Counter() for name in m.tables}
        occupied = {name: {} for name, t in m.tables.items() if isinstance(t, Tagged)}
        # Keep keys stable in this count test: only the memory under test learns.
        for t in range(len(seq)):
            q, state = m.score(x[:, :t+1])
            codes = state[0]
            target = (seq[t]+1) % V
            for name, table in m.tables.items():
                code = int(codes[name][-1])
                actual = table.read(codes[name][-1:])[0]
                want = torch.tensor([refs[name][code, c] for c in range(V)],
                                    device=device, dtype=torch.float32)
                torch.testing.assert_close(actual, want, atol=0, rtol=0)
                if isinstance(table, Tagged):
                    ix, tag = table.address(codes[name][-1:], torch.tensor([target], device=device))
                    cell, fingerprint = int(ix[0]), int(tag[0])
                    assert cell not in occupied[name] or occupied[name][cell] == fingerprint, 'tiny fixture collided'
                    occupied[name][cell] = fingerprint
                refs[name][code, target] += 1
                table.write(codes[name][-1:], torch.tensor([target], device=device), m.scratch)
            m.uni[target] += 1
        for name, table in m.tables.items():
            expected = torch.zeros_like(table.count)
            for (code, target), count in refs[name].items():
                if isinstance(table, Dense):
                    expected[code, target] += count
                else:
                    ix, _ = table.address(torch.tensor([code], device=device),
                                          torch.tensor([target], device=device))
                    expected[ix] += count
            assert torch.equal(expected, table.count), ('dict counts', name)
        assert int(m.uni.sum()) == len(seq)
        # Defined simultaneous-collision behavior and duplicate-tag accumulation.
        collision = Tagged(4, V, device)
        scratch = torch.zeros(16, dtype=torch.int32, device=device)
        collision.tag[3], collision.count[3] = 11, 5
        collision.commit(torch.tensor([3, 3, 3, 7], device=device),
                         torch.tensor([11, 22, 22, 9], dtype=torch.int32, device=device), scratch)
        assert int(collision.tag[3]) == 22 and int(collision.count[3]) == 2
        assert int(collision.count[7]) == 1
        memory_before = m.memory()
        derivative_error = finite_difference_checks(m, x)
        # Nonempty memory and nonzero learned local rules, before causal checks.
        y = (x+1) % V
        for _ in range(2):
            q, state = m.score(x)
            m.learn_write(x, y, q, state, .001)
        before = I.stamp(m)
        a = m.logprobs(x)
        for t in (0, 3, 7, len(seq)-2):
            other = x.clone()
            other[:, t+1:] = (other[:, t+1:]+3) % V
            b = m.logprobs(other)
            torch.testing.assert_close(a[:, :t+1], b[:, :t+1], atol=1e-6, rtol=0)
            # Inspect every expert, not only a possibly cancelling mixture.
            sa, sb = m.score(x)[1], m.score(other)[1]
            for name in sa[0]:
                assert torch.equal(sa[0][name][:t+1], sb[0][name][:t+1]), name
            torch.testing.assert_close(sa[2][:t+1], sb[2][:t+1], atol=1e-6, rtol=0)
        assert I.stamp(m) == before, 'evaluation wrote state'
        if m.net:
            # Change all word chars while preserving the word/nonword partition.
            ids = m.word[:V].nonzero().flatten()
            perm = torch.arange(V+1, device=device)
            perm[ids] = ids.roll(1)
            ca = torch.where(m.word[x], V+1, x)
            cb = torch.where(m.word[perm[x]], V+1, perm[x])
            assert torch.equal(ca, cb)
            assert all(torch.equal(u, v) for u, v in
                       zip(m.net.forward(ca)[-1], m.net.forward(cb)[-1]))
        assert m.memory() == memory_before, 'memory grew after writes'
        Q.plain(m)
        assert not torch.is_grad_enabled()
        return dict(causality=True, all_experts_causal=True, chunk1_exact_dict=True,
                    deterministic_collision_writes=True, bounded_structures=True,
                    no_autograd=True, local_finite_difference_max_error=derivative_error,
                    normalized_key_renaming_invariance=bool(m.net))


def run(mode, smoke=None, chunk_size=4096, seed=0):
    if smoke is not None and smoke < 1:
        raise ValueError('--smoke must be positive')
    if chunk_size < 1 or (chunk_size >= 128 and chunk_size % 128):
        raise ValueError('--chunk-size must be 1..127 or a positive multiple of 128')
    if chunk_size > 4096:
        raise ValueError('chunk sizes above 4096 are outside the 151 MiB working-set design')
    if seed < 0:
        raise ValueError('--seed must be nonnegative')
    path = I.ROOT/f'results/iter35_{mode}.json'
    if path.exists():
        raise FileExistsError(f'{path}; refusing to edit an existing result')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU/full-training fallback is run')
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, te, V = B.load()
    cut = int(len(tr)*.95)
    train = tr[:cut] if smoke is None else tr[:cut][:smoke]
    if len(train) >= 2**31:
        raise ValueError('int32 counts require fewer than 2^31 one-pass targets')
    vocab = json.loads((I.ROOT/'data/meta.json').read_text())['vocab']
    verified = checks(vocab, mode, 'cuda')
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = Model(vocab, mode)
    memory = model.memory()
    old_windows = B.N_EVAL_WINDOWS
    try:
        if smoke is not None:
            B.N_EVAL_WINDOWS = 16  # identical iter30 smoke convention
        torch.cuda.synchronize()
        start_eval = time.perf_counter()
        renamed, renaming = I.rename(te)
        vx, vy = B.eval_windows(tr[cut:])
        subset_n = max(1, len(vx)//4)
        subset = vx[:subset_n].clone(), vy[:subset_n].clone()
        del vx, vy
        torch.cuda.synchronize()
        eval_seconds = time.perf_counter()-start_eval
        eval_peak = torch.cuda.max_memory_allocated()/2**20
        torch.cuda.reset_peak_memory_stats()
        train_peak, seconds, chars, steps, curve = 0., 0., 0, 0, []
        stale = {k: torch.zeros((), dtype=torch.long, device='cuda')
                 for k in [*model.tables, 'any_context', 'including_unigram']}
        milestones = [math.ceil(len(train)*k/10) for k in range(1, 11)]
        next_milestone = 0
        total_steps = math.ceil(len(train)/chunk_size)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for x, y in epoch_cpu(train, seed, chunk_size, V):
            # Default: whole windows/chunks. Smaller chunks retain the prefix
            # input for causal copy, but score/write ONLY their selected targets.
            for left in range(0, x.numel(), chunk_size):
                steps += 1
                masked_y = y.clone()
                valid_range = torch.arange(x.numel(), device=x.device).reshape_as(x)
                masked_y[(valid_range < left) | (valid_range >= left+chunk_size)] = -1
                # For chunks shorter than one window, recomputing its prefix is
                # intentional. No future inputs may determine a memory address.
                q, state = model.score(x)
                stale_contexts(state[0], masked_y, stale)
                lr = .001 if steps <= .75*total_steps else .0001+.0009*.5*(
                    1+math.cos(math.pi*min(1., (steps-.75*total_steps)/(.25*total_steps))))
                model.learn_write(x, masked_y, q, state, lr)
                chars += int((masked_y >= 0).sum())
                del q, state, masked_y
                if next_milestone < 10 and chars >= milestones[next_milestone]:
                    torch.cuda.synchronize()
                    seconds += time.perf_counter()-start
                    train_peak = max(train_peak, torch.cuda.max_memory_allocated()/2**20)
                    torch.cuda.reset_peak_memory_stats()
                    eval_start = time.perf_counter()
                    before = I.stamp(model)
                    value = CONTROL.evaluate_subset(model.logprobs, subset)
                    assert I.stamp(model) == before, 'curve evaluation wrote state'
                    torch.cuda.synchronize()
                    eval_seconds += time.perf_counter()-eval_start
                    eval_peak = max(eval_peak, torch.cuda.max_memory_allocated()/2**20)
                    while next_milestone < 10 and chars >= milestones[next_milestone]:
                        curve.append(dict(fraction=(next_milestone+1)/10, chars=chars,
                                          train_seconds=seconds, val_subset=value))
                        next_milestone += 1
                    print(json.dumps(dict(progress=chars/len(train), val_subset=value,
                                          train_seconds=seconds)), flush=True)
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
        torch.cuda.synchronize()
        seconds += time.perf_counter()-start
        train_peak = max(train_peak, torch.cuda.max_memory_allocated()/2**20)
        assert chars == len(train) and len(curve) == 10
        assert memory == model.memory(), 'persistent memory grew'
        Q.plain(model)
        assert not torch.is_grad_enabled()
        torch.cuda.reset_peak_memory_stats()
        eval_start = time.perf_counter()
        before = I.stamp(model)
        test = I.evaluate(model.logprobs, te)
        renamed_loss = I.evaluate(model.logprobs, renamed)
        val = I.evaluate(model.logprobs, tr[cut:])
        nocopy = I.evaluate(lambda x: model.logprobs(x, False), te)
        novel = Q.novel(model.logprobs, te, renamed, vocab)
        assert I.stamp(model) == before, 'final evaluation wrote state'
        torch.cuda.synchronize()
        eval_seconds += time.perf_counter()-eval_start
        eval_peak = max(eval_peak, torch.cuda.max_memory_allocated()/2**20)
        estimate = flops(V, mode)
        observed = dict(test=test, renamed=renamed_loss, val=val, train_seconds=seconds,
                        peak_allocated_mib=max(train_peak, eval_peak),
                        flops_per_char_estimate=estimate['iter35'])
        limits = dict(TARGETS, flops_per_char_estimate=estimate['iter30_backprop'])
        # Validation is diagnostic: requested all-target conjunction covers five
        # resource/quality bars plus compute, not an additional val threshold.
        required = [k for k in limits if k != 'val']
        bars = {k: dict(value=observed[k], threshold=v, comparison='<',
                        pass_observed=observed[k] < v,
                        pass_full_run=(observed[k] < v) if smoke is None else None)
                for k, v in limits.items()}
        result = dict(mode=mode, smoke=smoke, comparable_full_run=smoke is None,
                      val=val, test=test, renamed=renamed_loss, test_nocopy=nocopy,
                      val_loss=val, test_loss=test, renamed_loss=renamed_loss,
                      test_loss_nocopy=nocopy, renamed_gap=renamed_loss-test,
                      train_seconds=seconds, eval_seconds=eval_seconds,
                      peak_allocated_mib=dict(train=train_peak, eval=eval_peak),
                      peak_allocated_mib_train=train_peak, peak_allocated_mib_eval=eval_peak,
                      memory_bytes=memory, flops_per_char_estimate=estimate,
                      stale_fraction=float(stale['including_unigram'])/chars,
                      stale_fraction_context=float(stale['any_context'])/chars,
                      stale_fraction_by_structure={k: float(v)/chars for k, v in stale.items()},
                      stale_definition='Fraction of scored positions with an earlier valid write in '
                                       'the same chunk to the same semantic context; primary fraction '
                                       'includes shared unigram. Per-structure values exclude incidental '
                                       'hash collision/eviction effects. Copy heads still see j<t in-window.',
                      val_curve=curve, curve=curve, targets=bars, checks=verified,
                      all_targets_pass=(all(bars[k]['pass_full_run'] for k in required)
                                        if smoke is None else None),
                      chars=chars, steps=steps, **novel,
                      settings=dict(seed=seed, device='cuda', mode=mode, passes=1,
                                    train_end=cut, val_start=cut, training_partition_chars=len(train),
                                    chunk_size=chunk_size, input_window=128, cpu_streamed_training=True,
                                    sampling='iter17/iter30 non-overlapping target windows; CPU '
                                             'randperm seed; row-major chunk reads then writes',
                                    evaluation_frozen=True, eval_windows=B.N_EVAL_WINDOWS,
                                    evaluation_functions=['iter5.evaluate (iter30 I.evaluate)',
                                                          'iter30.evaluate_subset', 'iter15.novel'],
                                    curve_subset='first quarter of fixed validation windows; no respacing',
                                    curve_milestones='first completed chunk at/after each 10% of targets',
                                    renaming=renaming, raw_orders=RAW_ORDERS, alpha=.5,
                                    experts=model.expert_names, key_bits=KEY_BITS if model.net else [],
                                    key_inputs='ASCII word chars share one symbol; six lags + three EMAs',
                                    key_loss='own raw-character CE only; manual exact derivative',
                                    key_net_width=KeysNet.D if model.net else 0,
                                    key_lr='.001 hold 75%, cosine to .0001', mixer_lr=.005,
                                    mixer='local log-linear pool; char and pair rows; log count / max prob features',
                                    collision='31-bit tag; max tag wins a chunk collision; replace then add',
                                    count_dtype='int32; bounded by single-pass char count <2^31',
                                    copy_lengths=C.LENS, probability_floor=FLOOR,
                                    nocopy='replace copy experts with highest lexical expert; frozen mixer',
                                    surprise='old exact bigram probability of observed input; thresholds 1/3; '
                                             'two iter10 three-event hashes',
                                    word_context='normalized 8-char shape + current run length (cap 15) + delimiter',
                                    metric_note='novel: iter15/iter30 aligned changed targets, maximal ASCII '
                                                'runs in a window; repeat requires an earlier complete run; '
                                                'README attributes earlier first/repeat observations to iter12',
                                    train_seconds_scope='CPU batch assembly/transfers, scoring, local updates, '
                                                        'memory writes and stale diagnostics; excludes eval/setup/checks',
                                    memory_scope='CUDA max allocated, separately sampled for train and eval; '
                                                 'persistent bytes separate from transient working set'))
        path.parent.mkdir(exist_ok=True)
        with path.open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2, allow_nan=False)
        return result
    finally:
        B.N_EVAL_WINDOWS = old_windows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('tables', 'keys'), default='tables')
    parser.add_argument('--smoke', type=int, metavar='N', help='prefix N targets; 16 eval windows; retain labeled JSON')
    parser.add_argument('--chunk-size', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args.mode, args.smoke, args.chunk_size, args.seed), indent=2), flush=True)
