"""GPU hashed counts, local mixers, PPM-C and the unchanged locked baseline."""
import sys
sys.dont_write_bytecode = True
import argparse, builtins, gc, io, json, keyword, math, random, time, tokenize
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import benchmark as B
from hybrid_mixer import Mixer as LocalMixer, Neuron
from hybrid_gpu import Hybrid, LENS
torch.set_grad_enabled(False)
ROOT = Path(__file__).resolve().parent
SIZE, MASK, MILLION = 1 << 25, (1 << 25) - 1, 999_424
NAMES = ('check', 'lr_grid', 'mix_ngram', 'mix_ngram_hebb', 'mix_full', 'ppm_full', 'baseline_full')
RULES = ['Drop Hebbian predictor unless mix_ngram_hebb beats mix_ngram by >=0.03 test nats.',
         'mix_full must beat ppm_full on test, or nothing beyond a compressor.',
         'renamed_loss - test_loss measures dependence on memorised names.']

def slot(h):
    # MurmurHash3 64-bit avalanche, with logical right shifts and signed constants.
    h = (h ^ ((h >> 33) & 0x7fffffff)) * -49064778989728563
    h = (h ^ ((h >> 33) & 0x7fffffff)) * -4265267296055464877
    h = h ^ ((h >> 33) & 0x7fffffff)
    # Low bits pick the cell; high bits are a nonzero tag, so collisions read as 0.
    return h & MASK, (((h >> 33) & 0x7fffffff) | 1).int()

class Counts:
    def __init__(self, V, ppm=False):
        self.V, self.ppm, self.lens = V, ppm, tuple(range(7)) if ppm else LENS
        self.C = [torch.zeros(SIZE, dtype=torch.int32, device='cuda') for _ in self.lens]
        self.T = [torch.zeros_like(c) for c in self.C]
        self.uni = torch.zeros(V, dtype=torch.int32, device='cuda')
        self.chars = torch.arange(V, device='cuda')

    def codes(self, x):
        b, t = x.shape; mx = max(self.lens)
        xp = torch.cat([torch.full((b, mx - 1), self.V, device=x.device), x], 1)
        h = torch.zeros_like(x); out = {0: h.clone()} if self.ppm else {}
        for j in range(1, mx + 1):
            h = h * 1000003 + xp[:, mx - j:mx - j + t] + 1
            if j in self.lens: out[j] = h.clone()
        return out

    def predict(self, codes, unseen=None):
        n = len(next(iter(codes.values())))
        p = (torch.full((self.V,), 1 / self.V, device='cuda') if self.ppm else
             (self.uni.float() + 1) / (self.uni.sum() + self.V)).expand(n, self.V)
        ps = []
        for i, order in enumerate(self.lens):
            h = codes[order]
            ix, tag = slot(h[:, None] * 1000003 + self.chars + 1)
            c = (self.C[i][ix] * (self.T[i][ix] == tag)).float()
            total = c.sum(1, keepdim=True)
            if unseen is not None: unseen[i] += (total == 0).sum()
            escape = (c > 0).sum(1, keepdim=True) if self.ppm else 0.5
            q = (c + escape * p) / (total + escape).clamp_min(0.5)
            p = torch.where(total > 0, q, p); ps.append(p)
        return ps

    def write(self, codes, y):
        one = torch.ones_like(y, dtype=torch.int32)
        self.uni.index_add_(0, y, one)
        for i, order in enumerate(self.lens):
            ix, tag = slot(codes[order] * 1000003 + y + 1)
            self.C[i][ix[self.T[i][ix] != tag]] = 0  # replace on collision
            self.T[i][ix] = tag
            ok = self.T[i][ix] == tag
            self.C[i].index_add_(0, ix[ok], one[ok])

class Model(LocalMixer):
    def __init__(self, V, lr, hebb=False, ppm=False):
        self.V, self.name, self.lr = V, 'mix_two_layer', lr
        self.base, self.ppm = Counts(V, ppm), ppm
        self.hebb = Hybrid(V, mode='hebb', surprise=False) if hebb else None
        self.first = [] if ppm else [Neuron(r, 10 if hebb else 5, lr) for r in (1, V + 1, 4096)]
        self.second = None if ppm else Neuron(V + 1, 3, lr)
        for n in self.first + ([self.second] if self.second else []): n.w = n.w.cuda()

    def output(self, cod, selectors, hc=None, y=None, unseen=None):
        ps = self.base.predict(cod, unseen)
        if self.ppm: return ps[-1].clamp_min(1e-30).log()
        if self.hebb: ps += self.hebb._predict(hc, False)[1]
        return self.mix(ps, selectors, y)

    def logprobs(self, x, unseen=None):
        shape = (*x.shape, self.V); out = []
        cod = {n: c.reshape(-1) for n, c in self.base.codes(x).items()}
        sel = [s.reshape(-1) for s in self.selectors(x)]
        hc = {n: c.reshape(-1, self.hebb.D) for n, c in self.hebb._codes(x).items()} if self.hebb else {}
        for a in range(0, x.numel(), 1024):
            out.append(self.output({n: c[a:a+1024] for n, c in cod.items()},
                       [s[a:a+1024] for s in sel], {n: c[a:a+1024] for n, c in hc.items()}, unseen=unseen))
        return torch.cat(out).reshape(shape)

def batches(tr, budget, bs, one_pass, V):
    g = torch.Generator().manual_seed(0); data = torch.as_tensor(tr.astype(np.int64), device='cuda')
    if one_pass:
        # Target blocks partition ALL training chars, including the final short block.
        starts = torch.randperm(math.ceil(len(tr) / 128), generator=g) * 128
        left = len(tr) if budget is None else min(budget, len(tr))
        for a in range(0, len(starts), bs):
            st = starts[a:a+min(bs, math.ceil(left / 128))]
            if not len(st): break
            ix = st.cuda()[:, None] + torch.arange(128, device='cuda')
            valid = (ix < len(tr)) & (torch.arange(ix.numel(), device='cuda').reshape_as(ix) < left)
            x = torch.where(ix > 0, data[(ix-1).clamp(0, len(tr)-1)], V)
            y = torch.where(valid, data[ix.clamp_max(len(tr)-1)], -1)
            yield x, y
            left -= int(valid.sum())
            if left <= 0: break
    else:
        left = MILLION if budget is None else budget
        while left >= 128:
            n = min(bs, left // 128)
            st = torch.randint(0, len(tr) - 129, (n,), generator=g).cuda()
            ix = st[:, None] + torch.arange(128, device='cuda')
            yield data[ix], data[ix + 1]
            left -= n * 128

def fit(m, tr, budget, bs, one_pass):
    torch.cuda.synchronize(); start = time.perf_counter(); chars = 0
    for x, y in batches(tr, budget, bs, one_pass, m.V):
        cod, sel = m.base.codes(x), m.selectors(x)
        hc = m.hebb._codes(x) if m.hebb else {}
        masked = bool((y < 0).any())
        for t in range(x.shape[1]):
            keep = y[:, t] >= 0 if masked else slice(None); target = y[:, t][keep]
            if not len(target): continue
            c = {n: v[:, t][keep] for n, v in cod.items()}; h = {n: v[:, t][keep] for n, v in hc.items()}
            if not m.ppm: m.output(c, [s[:, t][keep] for s in sel], h, target)
            m.base.write(c, target)
            if m.hebb: m.hebb.step(h, target)
        chars += int((y >= 0).sum())
    torch.cuda.synchronize()
    return time.perf_counter() - start, chars

def stamp(o):
    if isinstance(o, torch.Tensor): return (id(o), o._version)
    if isinstance(o, dict): return tuple((k, stamp(v)) for k, v in o.items())
    if isinstance(o, (list, tuple)): return tuple(map(stamp, o))
    return stamp(vars(o)) if hasattr(o, '__dict__') else repr(o)

def evaluate(fn, data):
    x, y = B.eval_windows(data); total = 0.0
    for a in range(0, len(x), 16):
        logits = fn(x[a:a+16].cuda()).float()
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y[a:a+16].cuda().reshape(-1), reduction='sum').item()
    return total / y.numel()

def rename(te):
    vocab = json.loads((ROOT / 'data/meta.json').read_text())['vocab']; enc = {c: i for i, c in enumerate(vocab)}
    text = ''.join(vocab[int(c)] for c in te); lines = text.splitlines(keepends=True)
    offsets = np.cumsum([0] + [len(s) for s in lines]); rng = random.Random(0); mapping = {}; edits = []; fallback = False
    def collect(source, row=0):
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
        for t in tokens:
            if t.type != tokenize.NAME or keyword.iskeyword(t.string) or t.string in vars(builtins): continue
            if t.string not in mapping:
                candidate = t.string
                while candidate == t.string or keyword.iskeyword(candidate) or candidate in vars(builtins):
                    candidate = ''.join(rng.choice('abcdefghijklmnopqrstuvwxyz') for _ in t.string)
                mapping[t.string] = candidate
            a = int(offsets[row + t.start[0] - 1]) + t.start[1]
            edits.append((a, a + len(t.string), mapping[t.string]))
    try: collect(text)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        fallback = True; edits.clear(); mapping.clear(); rng.seed(0)
        for row, line in enumerate(lines):
            try: collect(line, row)
            except (tokenize.TokenError, IndentationError, SyntaxError): pass
    result = list(text); changed = 0
    for a, b, replacement in edits:
        changed += sum(c != d for c, d in zip(text[a:b], replacement)); result[a:b] = replacement
    return np.array([enc[c] for c in result], dtype=np.uint8), dict(fallback=fallback, fraction_chars_renamed=changed / len(te), identifier_chars_fraction=sum(b-a for a,b,_ in edits)/len(te))

def run(name, lr=None, budget=None):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.cuda.set_per_process_memory_fraction(2.7 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(0); tr, te, V = B.load(); assert V == 96
    cut = int(len(tr) * .95); val = tr[cut:]; train = tr if name in ('check', 'baseline_full') else tr[:cut]
    if budget is not None and budget < (8192 if name == 'baseline_full' else 128): raise ValueError('budget too small')
    if name == 'check' and lr not in (None, 0.0003125): raise ValueError('check lr is locked')
    if lr is None:
        lr = 0.0003125 if name == 'check' else (None if name in ('lr_grid', 'ppm_full', 'baseline_full') else
             json.loads((ROOT / 'results/iter5_lr_grid.json').read_text())['best_lr'])
    settings = dict(seed=0, device='cuda', bs=64 if name == 'check' else 976, T=128, lr=lr, budget=budget,
                    sampling='one_pass' if name in ('mix_full', 'ppm_full') else 'seeded_random_windows',
                    val_start=cut, train_end=len(train), count_entries=SIZE, count_dtype='int32',
                    lens=list(range(7)) if name == 'ppm_full' else list(LENS), alpha=.5, batch_delta='sum',
                    mixer='mix_two_layer', probability_floor=1e-6, decision_rules=RULES, allocator_limit_gib=2.7,
                    hash='rolling h*1000003+tok+1; Murmur3 finalizer; low 25 bits', count_normalizer='sum of V read counts', collisions='31-bit tag, replace on collision')
    renamed, info = rename(te); settings['renaming'] = info
    result = dict(name=name, settings=settings)
    if name == 'baseline_full':
        import baseline
        original = B.evaluate; captured = []
        def capture(fn, *args, **kwargs):
            captured[:] = [fn]; return original(fn, *args, **kwargs)
        B.evaluate = capture
        try:
            with torch.enable_grad(): result.update(baseline.train(0, 16_000_000 if budget is None else budget))
        finally: B.evaluate = original; torch.set_grad_enabled(False)
        result.update(val_loss=evaluate(captured[0], val), renamed_loss=evaluate(captured[0], renamed), chars=result['chars_trained'])
        settings.update(baseline_cfg=baseline.CFG, bs=64, sampling='baseline locked sampler', budget=16_000_000 if budget is None else budget)
    else:
        grid = []; m = None; total_seconds = 0.0; total_chars = 0
        for rate in ([3e-5, 1e-4, 3e-4, 1e-3] if name == 'lr_grid' else [lr]):
            m = Model(V, rate, name == 'mix_ngram_hebb', name == 'ppm_full')
            seconds, chars = fit(m, train, budget, settings['bs'], name in ('mix_full', 'ppm_full'))
            before = stamp(m); loss = evaluate(m.logprobs, val); assert stamp(m) == before, 'evaluation wrote state'
            grid.append(dict(lr=rate, val_loss=loss, train_seconds=seconds, chars=chars))
            total_seconds += seconds; total_chars += chars
            if name == 'lr_grid': del m; m = None; gc.collect(); torch.cuda.empty_cache()
        if name == 'lr_grid':
            best = min(grid, key=lambda r: r['val_loss'])
            result.update(grid=grid, best_lr=best['lr'], val_loss=best['val_loss'], test_loss=None, chars=total_chars, train_seconds=total_seconds)
        else:
            unseen = {}
            def score(data, key):
                counts = torch.zeros(len(m.base.lens), dtype=torch.long, device='cuda') if m.ppm else None
                value = evaluate(lambda x: m.logprobs(x, counts), data)
                if counts is not None: unseen[key] = dict(zip(map(str, m.base.lens), counts.tolist()))
                return value
            result.update(val_loss=loss, test_loss=score(te, 'test'), renamed_loss=score(renamed, 'renamed'), chars=chars, train_seconds=seconds)
            if m.ppm: result.update(unseen_contexts=unseen, val_loss=score(val, 'val')); settings.update(escape='distinct read chars', exclusions=False)
            assert stamp(m) == before, 'evaluation wrote state'
    result['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
    settings['requested_chars'] = budget if budget is not None else (16_000_000 if name == 'baseline_full' else len(train) if settings['sampling'] == 'one_pass' else MILLION)
    with (ROOT / f'results/iter5_{name}.json').open('w', encoding='utf-8') as f: json.dump(result, f, indent=2)
    if name == 'check' and abs(result['test_loss'] - 1.3803) > .01: raise RuntimeError(f"check failed: {result['test_loss']:.6f}; expected 1.3803 +/- 0.01")
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('name', choices=NAMES)
    p.add_argument('--lr', type=float); p.add_argument('--budget', type=int)
    a = p.parse_args(); print(json.dumps(run(a.name, a.lr, a.budget), indent=2), flush=True)
