"""Causal role contexts and a small, bounded, gradient-free memory LM.

At position t, x[t] has been consumed and y[t] has NOT. Identifier identities
are committed only on the first non-word character, never at a window end.
Partial words are represented by length, not spelling. Quoted text/comments
stay character literals. Role ids are zero based; first occurrence and MTF
recency are separate namespaces. Historical MTF tokens keep their rank at
the time they were emitted (they are not retroactively relabelled).

Library: IdentifierTracker, role_features, literal_keys, decode_roles,
encode_targets, BoundedCounts, RoleMemoryLM. Importing does not load data,
initialize CUDA, alter grad mode, or import any experiment module.

Run --eval --smoke 204800 [--literal-only], or --checks alone. Full evaluation
is opt-in. JSON results refuse overwrites. See briefs/iter36_codex_out.md.
"""
from __future__ import annotations

import argparse
import builtins
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import keyword
import math
from pathlib import Path
import re
import sys
import time
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
BASE = 1000003
ROLE_BASE = 1 << 40
WORD_BASE = 1 << 50
PAD_TOKEN = 257
SCHEMES = ("first", "recency")
# Match I.rename's hard-keyword/builtin exclusions. Soft keywords such as
# `match` can also be ordinary identifiers, and I.rename renames them.
RESERVED = frozenset(keyword.kwlist) | frozenset(vars(builtins))
REFERENCES = dict(bp_test=1.2562, bp_renamed=1.5714,
                  tables_test=1.165, tables_renamed=1.606)
WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
# Greedy bodies cannot cross an unescaped closing quote. Unterminated quotes
# consume to newline (single) or EOF (triple), just as a causal lexer would.
# All characters in these spans are emitted individually, not as one token.
SPANS = re.compile(
    r"(?P<opaque>\#[^\n]*|'''(?:\\(?:[\s\S]|$)|(?!''')[^\\])*(?:'''|$)|"
    r'"""(?:\\(?:[\s\S]|$)|(?!""")[^\\])*(?:"""|$)|'
    r"'(?:\\(?:[\s\S]|$)|[^'\\\n])*(?:'|(?=\n)|$)|"
    r'"(?:\\(?:[\s\S]|$)|[^"\\\n])*(?:"|(?=\n)|$))'
    r"|(?P<word>[A-Za-z_][A-Za-z_0-9]*)"
    r"|(?P<number>[0-9][A-Za-z_0-9.]*)"
    r"|(?P<raw>[^A-Za-z_0-9#'\"]+)", re.DOTALL)


def word_token(word: str) -> int:
    """Stable literal-word namespace; independent of Python's hash seed."""
    return WORD_BASE + int.from_bytes(hashlib.blake2b(
        word.encode("ascii"), digest_size=6).digest(), "little")


class IdentifierTracker:
    """Per-window identity state; count statistics may be shared across windows.

    frequency_threshold <= 0 disables promotion. Otherwise classification at
    a boundary uses counts BEFORE that completed word's training write.
    Neither evaluation nor partial-word observation writes train counts.
    ``complete`` is also useful to a streaming lexer: call it only after its
    delimiter has arrived. ``partial`` holds the prefix consumed so far.
    ``scan`` exposes a strictly causal character-by-character interface.
    """
    def __init__(self, train_counts: Mapping[str, int] | None = None,
                 frequency_threshold: int = 100, update_counts: bool = False):
        self.train_counts = train_counts if train_counts is not None else Counter()
        self.threshold = frequency_threshold
        self.update_counts = update_counts
        self.first: dict[str, int] = {}
        self.recent: list[str] = []
        self.completed: Counter = Counter()
        self.partial = ""

    def is_literal(self, word: str) -> bool:
        return (word in RESERVED or (self.threshold > 0 and
                self.train_counts.get(word, 0) >= self.threshold))

    def role_names(self, scheme: str = "first") -> list[str]:
        if scheme == "first":
            return list(self.first)
        if scheme == "recency":
            return list(self.recent)
        raise ValueError(scheme)

    def complete(self, word: str) -> dict[str, int]:
        """Commit one observed identifier; return its two emitted tokens."""
        if not WORD.fullmatch(word):
            raise ValueError("not an ASCII Python identifier")
        literal = self.is_literal(word)
        if literal:
            tokens = dict.fromkeys(SCHEMES, word_token(word))
        else:
            if word not in self.first:
                self.first[word] = len(self.first)
            # Emit pre-touch MTF rank. A first use is the next unused rank;
            # repeats use their current rank, then move to front.
            rank = self.recent.index(word) if word in self.recent else len(self.recent)
            tokens = dict(first=ROLE_BASE+self.first[word], recency=ROLE_BASE+rank)
            if word in self.recent:
                self.recent.remove(word)
            self.recent.insert(0, word)
        self.completed[word] += 1
        if self.update_counts:
            self.train_counts[word] = self.train_counts.get(word, 0) + 1
        self.partial = ""
        return tokens

    def scan(self, text: str):
        """Yield (tokens_emitted, prefix, role_names) after each character.

        Segmentation uses a regex accelerator, but no word identity or literal
        classification is exposed until its delimiter. No EOF flush occurs.
        This processes one document/window; reset with a fresh tracker.
        """
        pending = ""
        for match in SPANS.finditer(text):
            emitted = {s: [] for s in SCHEMES}
            if pending:
                tok = self.complete(pending)
                emitted = {s: [tok[s]] for s in SCHEMES}
                pending = ""
            if match.lastgroup == "word":
                word = match.group()
                for j in range(len(word)):
                    self.partial = word[:j+1]
                    yield ({s: emitted[s] if j == 0 else [] for s in SCHEMES},
                           self.partial, {s: self.role_names(s) for s in SCHEMES})
                pending = word
            else:
                self.partial = ""
                for j, char in enumerate(match.group()):
                    tok = ord(char)+1
                    yield ({s: (emitted[s] if j == 0 else [])+[tok] for s in SCHEMES},
                           "", {s: self.role_names(s) for s in SCHEMES})


@dataclass
class RoleFeatures:
    # keys[scheme][order] are int64 [B,T], on the requested device.
    keys: dict[str, dict[int, torch.Tensor]]
    partial_length: torch.Tensor
    # Optional int16 [B,T,R]. >=0 is a vocabulary index; -1 means a known
    # identifier inconsistent with the prefix; -2 means an unbound/new role.
    role_next: dict[str, torch.Tensor]
    observed_counts: Counter
    # Final window states (not per-position inputs to the hashes/decoder).
    trackers: list[IdentifierTracker]


def _validate_orders(orders):
    orders = tuple(sorted(set(int(k) for k in orders)))
    if not orders or orders[0] < 1 or orders[-1] > 128:
        raise ValueError("orders must be in 1..128")
    return orders


def _input_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.int64)
    if x.ndim == 1:
        x = x[None]
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("x must be a nonempty [T] or [B,T] array")
    return x


@torch.no_grad()
def role_features(x, vocab: Sequence[str], orders=(1, 2, 4, 8), *,
                  schemes=SCHEMES, train_counts=None, frequency_threshold=100,
                  update_counts=False, max_roles=32, build_decode=False,
                  device=None, chunk_rows=512) -> RoleFeatures:
    """Extract name-free context hashes, including causal partial-word length.

    CPU regex/span bookkeeping; vectorized/chunked Torch token gathering and
    int64 rolling hashes on CPU or GPU. No Python loop over characters for
    hashing or identifier-prefix lengths. ``build_decode`` materializes only
    bounded role-to-next-char proposals and is best used on small batches.
    For 6M characters, call by windows/chunks to avoid retaining every hash.

    x may contain pad=len(vocab), represented as a non-word sentinel. Train
    counts must reflect only earlier training writes. For parallel training,
    use update_counts=False and commit observed_counts AFTER scoring/writing
    the batch; this is a causal, one-batch-delayed frequency policy.
    """
    orders = _validate_orders(orders)
    schemes = tuple(schemes)
    if not schemes or any(s not in SCHEMES for s in schemes):
        raise ValueError("schemes must contain first and/or recency")
    if max_roles < 1 or max_roles > 32700 or chunk_rows < 1:
        raise ValueError("invalid role capacity or chunk_rows")
    if len(vocab) > 32700 or any(len(c) != 1 or ord(c) > 127 for c in vocab):
        raise ValueError("expected the benchmark ASCII character vocabulary")
    if device is None:
        device = x.device if isinstance(x, torch.Tensor) else "cpu"
    data = _input_numpy(x)
    if train_counts is None:
        train_counts = Counter()
    if (data < 0).any() or (data > len(vocab)).any():
        raise ValueError("x contains an invalid character index")
    B, T = data.shape
    chars = list(vocab)+["\0"]
    enc = {c: i for i, c in enumerate(vocab)}
    ends = np.zeros((B, T), dtype=np.int64)
    partial = np.zeros_like(ends)
    events = {s: [] for s in schemes}
    proposals = {s: np.full((B, T, max_roles), -2, dtype=np.int16)
                 for s in schemes} if build_decode else {}
    trackers, observed = [], Counter()

    for b, row in enumerate(data):
        text = "".join(chars[c] for c in row)
        tracker = IdentifierTracker(train_counts, frequency_threshold, update_counts)
        history = {s: [] for s in schemes}
        pending = ""
        for match in SPANS.finditer(text):
            a, z = match.span()
            if pending:
                tokens = tracker.complete(pending)
                for s in schemes:
                    history[s].append(tokens[s])
                pending = ""
            isword = match.lastgroup == "word"
            span = match.group()
            if isword:
                ends[b, a:z] = len(history[schemes[0]])
                partial[b, a:z] = np.arange(1, z-a+1)
                pending = span
                tracker.partial = span  # never commit a word at EOF
            else:
                n = len(history[schemes[0]])
                literal_tokens = [ord(c)+1 for c in span]
                for s in schemes:
                    history[s].extend(literal_tokens)
                ends[b, a:z] = n+np.arange(1, z-a+1)
                tracker.partial = ""
            if build_decode:
                for s in schemes:
                    names = tracker.role_names(s)[:max_roles]
                    proposals[s][b, a:z, :len(names)] = -1
                    if match.lastgroup == "opaque":
                        continue
                    for r, name in enumerate(names):
                        if isword:
                            common = 0
                            for u, v in zip(span, name):
                                if u != v:
                                    break
                                common += 1
                            n = min(common, len(name)-1)
                            if n:
                                proposals[s][b, a:a+n, r] = [enc[c] for c in name[1:n+1]]
                        else:
                            proposals[s][b, a:z, r] = enc[name[0]]
        for s in schemes:
            events[s].append(history[s])
        observed.update(tracker.completed)
        trackers.append(tracker)

    lengths = partial*2+(partial > 0)
    keys = {s: {k: torch.empty((B, T), dtype=torch.int64, device=device)
                for k in orders} for s in schemes}
    for a in range(0, B, chunk_rows):
        z = min(B, a+chunk_rows)
        end = torch.as_tensor(ends[a:z], device=device)
        p = torch.as_tensor(lengths[a:z], device=device)
        for s in schemes:
            width = max(1, max(map(len, events[s][a:z])))
            ev = np.full((z-a, width), PAD_TOKEN, dtype=np.int64)
            for j, seq in enumerate(events[s][a:z]):
                ev[j, :len(seq)] = seq
            ev = torch.as_tensor(ev, device=device)
            # One hash recurrence per lag, reusing the same gathers across
            # orders. Short histories are left padded with PAD_TOKEN.
            for k in orders:
                h = torch.full_like(end, 17+k)
                for lag in range(k-1, -1, -1):
                    ix = end-1-lag
                    token = ev.gather(1, ix.clamp(0, width-1))
                    token = torch.where(ix >= 0, token, PAD_TOKEN)
                    h = h*BASE+token
                keys[s][k][a:z] = h*BASE+p
    return RoleFeatures(keys, torch.as_tensor(partial, device=device),
                        {s: torch.as_tensor(p, device=device) for s, p in proposals.items()},
                        observed, trackers)


@torch.no_grad()
def literal_keys(x: torch.Tensor, orders=(1, 2, 4, 8), pad=None):
    """Same int64 rolling hash convention, over literal character ids."""
    orders = _validate_orders(orders)
    if x.ndim != 2:
        raise ValueError("expected [B,T]")
    pad = PAD_TOKEN if pad is None else pad+1
    result = {}
    for k in orders:
        h = torch.full_like(x, 17+k, dtype=torch.int64)
        for lag in range(k-1, -1, -1):
            token = torch.full_like(x, pad, dtype=torch.int64)
            if lag < x.shape[1]:
                token[:, lag:] = x[:, :x.shape[1]-lag]+1
            h = h*BASE+token
        result[k] = h
    return result


@torch.no_grad()
def decode_roles(role_mass: torch.Tensor, literal_counts: torch.Tensor,
                 role_next: torch.Tensor, fallback: torch.Tensor | None = None):
    """Convert symbol masses to normalized next-character probabilities.

    Shapes [...,R] (or [...,R+1] including explicit NEW), [...,V], [...,R].
    Known inconsistent roles (-1) are discarded, then valid mass renormalizes.
    Unbound roles (-2) and NEW use the literal distribution. A role whose
    name ends exactly at the prefix is inconsistent: delimiter characters
    are supplied by literal counts. All-zero inputs use fallback/uniform.
    Neither a future complete identifier nor a target is needed to decode.
    """
    R, V = role_next.shape[-1], literal_counts.shape[-1]
    if role_mass.shape[-1] not in (R, R+1):
        raise ValueError("role mass must have R or R+1 columns")
    literal = literal_counts.float().clamp_min(0)
    if fallback is None:
        fallback = torch.full_like(literal, 1/V)
    else:
        fallback = fallback.float().clamp_min(0)
        fallback_total = fallback.sum(-1, keepdim=True)
        fallback = torch.where(fallback_total > 0,
                               fallback/fallback_total.clamp_min(1e-30),
                               torch.full_like(fallback, 1/V))
    base = torch.where(literal.sum(-1, keepdim=True) > 0,
                       literal/literal.sum(-1, keepdim=True).clamp_min(1e-30), fallback)
    mass = role_mass.float().clamp_min(0)
    out = literal.clone()
    valid = role_next >= 0
    out.scatter_add_(-1, role_next.long().clamp_min(0), mass[..., :R]*valid)
    new_mass = (mass[..., :R]*(role_next == -2)).sum(-1, keepdim=True)
    if mass.shape[-1] == R+1:
        new_mass = new_mass+mass[..., R:]
    out = out+new_mass*base
    total = out.sum(-1, keepdim=True)
    return torch.where(total > 0, out/total.clamp_min(1e-30), base)


@torch.no_grad()
def encode_targets(y: torch.Tensor, role_next: torch.Tensor, V: int):
    """Causal symbol write: first available role proposing observed y, else y.

    When multiple known names share the prefix/next character, choose the
    lowest role. This is deliberate next-character ambiguity; it never looks
    ahead to disambiguate the full target word. Unknown names write literals.
    Padded targets (-1) remain masked and must not be written to memory.
    """
    match = (role_next == y[..., None]) & (role_next >= 0)
    role = match.long().argmax(-1)
    return torch.where(match.any(-1) & (y >= 0), V+role, y)


class BoundedCounts:
    """Fixed direct-mapped tagged context rows, dense float32 symbol counts.

    Collisions replace whole rows. Concurrent writes choose the last input
    row's tag deterministically; only matching writers add. With batch=1 and
    collision-free slots this is exactly a dict of context -> Counter.
    Float32 counts halve before losing unit increments at 2**23.
    """
    def __init__(self, symbols: int, slots=4096, device="cpu"):
        if symbols < 1 or slots < 2 or slots & (slots-1):
            raise ValueError("slots must be a power of two >= 2")
        self.symbols, self.slots = symbols, slots
        self.tags = torch.zeros(slots, dtype=torch.int64, device=device)
        self.counts = torch.zeros((slots, symbols), dtype=torch.float32, device=device)

    def slot(self, h):
        # Signed overflow is intentional; mix upper bits into slot bits.
        h = (h ^ (h >> 29))*-49064778989728563
        return (h ^ (h >> 32)) & (self.slots-1)

    @torch.no_grad()
    def read(self, h):
        ix = self.slot(h)
        return self.counts[ix]*(self.tags[ix] == h)[..., None]

    @torch.no_grad()
    def write(self, h, y):
        keep = y >= 0
        h, y = h[keep], y[keep]
        if not h.numel():
            return
        if bool((y >= self.symbols).any()):
            raise ValueError("symbol outside table vocabulary")
        ix = self.slot(h)
        order = ix.argsort(stable=True)
        sorted_ix = ix[order]
        last = torch.ones_like(sorted_ix, dtype=torch.bool)
        last[:-1] = sorted_ix[:-1] != sorted_ix[1:]
        winners = order[last]
        cells, tags = ix[winners], h[winners]
        replace = self.tags[cells] != tags
        self.counts[cells[replace]] = 0
        self.tags[cells] = tags
        large = self.counts[cells].amax(-1) >= 2**23
        self.counts[cells[large]] *= .5
        ok = self.tags[ix] == h
        flat = ix[ok]*self.symbols+y[ok]
        self.counts.view(-1).index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))

    def memory_bytes(self):
        return sum(v.numel()*v.element_size() for v in (self.tags, self.counts))


class LocalMixer:
    """hybrid_mixer-style local logistic update, without autograd."""
    def __init__(self, V, experts, device, lr=.02):
        self.V, self.lr = V, lr
        # Last observed char plus a four-level count-confidence bucket.
        self.w = torch.full(((V+1)*4, experts), 1/experts, device=device)

    @torch.no_grad()
    def __call__(self, ps, x, confidence, y=None):
        lp = torch.stack(ps, 1).clamp_min(1e-6).log()
        bucket = ((confidence > 0).long()+(confidence > 3)+(confidence > 15))
        selector = x+bucket*(self.V+1)
        q = (self.w[selector, :, None]*lp).sum(1).log_softmax(-1)
        if y is not None:
            keep = y >= 0
            if bool(keep.any()):
                inp, out, target = lp[keep], q[keep].exp(), y[keep]
                observed = inp.gather(2, target[:, None, None].expand(-1, inp.shape[1], 1)).squeeze(-1)
                delta = observed-(out[:, None]*inp).sum(-1)
                self.w.index_add_(0, selector[keep], self.lr*delta/max(1, int(keep.sum())))
                self.w.clamp_(-4, 4)
        return q


class RoleMemoryLM:
    """Bounded literal+role memory, fixed model state during evaluation.

    train_batch predicts each timestep before writing it. Different windows
    in the batch are parallel independent examples; they see a pre-timestep
    table snapshot. There are no within-window stale count writes. Frequency
    promotion deliberately uses a pre-batch snapshot, then commits the batch.
    """
    def __init__(self, vocab, orders=(1, 2, 4, 8), slots=4096, max_roles=32,
                 scheme="first", frequency_threshold=100, literal_only=False,
                 device="cpu"):
        self.vocab, self.V = list(vocab), len(vocab)
        self.orders, self.scheme = _validate_orders(orders), scheme
        if scheme not in SCHEMES:
            raise ValueError(scheme)
        self.max_roles, self.threshold = max_roles, frequency_threshold
        self.literal_only, self.device = literal_only, torch.device(device)
        self.train_counts = Counter()
        self.literal = {k: BoundedCounts(self.V, slots, device) for k in self.orders}
        self.role = {} if literal_only else {
            k: BoundedCounts(self.V+max_roles+1, slots, device) for k in self.orders}
        self.unigram = torch.full((self.V,), .5, device=device)
        self.mixer = LocalMixer(self.V, 1+len(self.literal)+len(self.role), device)

    def prepare(self, x):
        lk = literal_keys(x, self.orders, self.V)
        rf = None if self.literal_only else role_features(
            x, self.vocab, self.orders, schemes=(self.scheme,),
            train_counts=self.train_counts, frequency_threshold=self.threshold,
            max_roles=self.max_roles, build_decode=True, device=self.device)
        return lk, rf

    @torch.no_grad()
    def output(self, chars, lk, rk=None, role_next=None, y=None):
        base = (self.unigram/self.unigram.sum()).expand(len(chars), -1)
        ps = [base]
        confidence = torch.zeros(len(chars), device=self.device)
        p = base
        for k, table in self.literal.items():
            c = table.read(lk[k])
            total = c.sum(-1, keepdim=True)
            p = (c+.5*p)/(total+.5)
            ps.append(p)
            confidence = torch.maximum(confidence, total[:, 0])
        literal_prior = p
        for k, table in self.role.items():
            c = table.read(rk[k])
            decoded = decode_roles(c[:, self.V:], c[:, :self.V]+.5*literal_prior,
                                   role_next, literal_prior)
            ps.append(decoded)
            confidence = torch.maximum(confidence, c.sum(-1))
        # The returned score is computed with the old mixer AND old tables.
        q = self.mixer(ps, chars, confidence, y)
        if y is not None:
            for k, table in self.literal.items():
                table.write(lk[k], y)
            if self.role:
                symbol = encode_targets(y, role_next, self.V)
                for k, table in self.role.items():
                    table.write(rk[k], symbol)
            valid = y[y >= 0]
            self.unigram.index_add_(0, valid, torch.ones_like(valid, dtype=torch.float32))
        return q

    @torch.no_grad()
    def train_batch(self, x, y):
        lk, rf = self.prepare(x)
        total = 0.
        for t in range(x.shape[1]):
            q = self.output(x[:, t], {k: v[:, t] for k, v in lk.items()},
                            {k: v[:, t] for k, v in rf.keys[self.scheme].items()} if rf else None,
                            rf.role_next[self.scheme][:, t] if rf else None, y[:, t])
            total += F.nll_loss(q, y[:, t], ignore_index=-1, reduction="sum").item()
        if rf:
            self.train_counts.update(rf.observed_counts)
        return total

    @torch.no_grad()
    def logprobs(self, x):
        lk, rf = self.prepare(x)
        flat = {k: v.flatten() for k, v in lk.items()}
        rk = {k: v.flatten() for k, v in rf.keys[self.scheme].items()} if rf else None
        nxt = rf.role_next[self.scheme].reshape(-1, self.max_roles) if rf else None
        xx, out = x.flatten(), []
        for a in range(0, len(xx), 1024):
            z = a+1024
            out.append(self.output(xx[a:z], {k: v[a:z] for k, v in flat.items()},
                                   {k: v[a:z] for k, v in rk.items()} if rk else None,
                                   nxt[a:z] if nxt is not None else None))
        return torch.cat(out).reshape(*x.shape, self.V)

    def memory_bytes(self):
        return dict(literal_tables=sum(t.memory_bytes() for t in self.literal.values()),
                    role_tables=sum(t.memory_bytes() for t in self.role.values()),
                    mixer=self.mixer.w.numel()*self.mixer.w.element_size(),
                    unigram=self.unigram.numel()*self.unigram.element_size(),
                    # Includes Python hash-table allocation, string keys and
                    # count values; tracks frequency metadata separately.
                    train_word_counts_cpu=sys.getsizeof(self.train_counts)+sum(
                        sys.getsizeof(k)+sys.getsizeof(v) for k, v in self.train_counts.items()))


def training_windows(train, V, bs=16, seed=0, T=128):
    """CPU equivalent of iter30.R.epoch: identical permutation/target windows."""
    starts = torch.randperm(math.ceil(len(train)/T), generator=
                            torch.Generator().manual_seed(seed)).numpy()*T
    pos = np.arange(T)
    for a in range(0, len(starts), bs):
        ix = starts[a:a+bs, None]+pos
        x = np.where(ix > 0, train[(ix-1).clip(0, len(train)-1)], V)
        y = np.where(ix < len(train), train[ix.clip(0, len(train)-1)], -1)
        yield torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))


def _encode(text, vocab):
    enc = {c: i for i, c in enumerate(vocab)}
    return np.array([[enc[c] for c in text]], dtype=np.int64)


def _signed64(value):
    return ((value+(1 << 63)) % (1 << 64))-(1 << 63)


def checks(vocab=None, rename=None, device="cpu"):
    """No training data needed; includes real I.rename when supplied by --eval."""
    vocab = list(vocab) if vocab is not None else ["\n", "\t"]+[chr(i) for i in range(32, 127)]
    text = 'alpha = 1\nbeta = alpha + alpha\nprint(beta)\n# alpha\ns = "alpha"\n'
    x = _encode(text, vocab)
    original = role_features(x, vocab, build_decode=True, frequency_threshold=0,
                             max_roles=8, device=device)
    enc = {c: i for i, c in enumerate(vocab)}
    # All cut points, including inside identifiers, strings and comments.
    for t in range(x.shape[1]):
        changed = x.copy()
        changed[:, t+1:] = enc["z"]
        other = role_features(changed, vocab, build_decode=True,
                              frequency_threshold=0, max_roles=8, device=device)
        for s in SCHEMES:
            for k in original.keys[s]:
                assert torch.equal(original.keys[s][k][:, :t+1], other.keys[s][k][:, :t+1]), ("future hash", s, t)
            assert torch.equal(original.role_next[s][:, :t+1], other.role_next[s][:, :t+1]), ("future decode", s, t)
    if rename is not None:
        renamed, _ = rename(x[0].astype(np.uint8))
    else:
        # The real rename excludes comments and strings.
        renamed = _encode(text.replace("alpha", "gamma").replace("beta", "zeta")
                          .replace("# gamma", "# alpha").replace('"gamma"', '"alpha"'), vocab)[0]
    other = role_features(renamed, vocab, frequency_threshold=0, device=device)
    outside = torch.tensor([not(c.isalnum() or c == "_") for c in text], device=device)
    for s in SCHEMES:
        for k in original.keys[s]:
            assert torch.equal(original.keys[s][k][0, outside], other.keys[s][k][0, outside]), ("rename", s, k)

    # Independent scalar recurrence over tracker events verifies every key.
    tracker = IdentifierTracker(frequency_threshold=0)
    history = {s: [] for s in SCHEMES}
    for t, (emitted, prefix, _) in enumerate(tracker.scan(text)):
        for s in SCHEMES:
            history[s].extend(emitted[s])
            for k in original.keys[s]:
                tokens = ([PAD_TOKEN]*k+history[s])[-k:]
                h = 17+k
                for token in tokens:
                    h = _signed64(h*BASE+token)
                h = _signed64(h*BASE+2*len(prefix)+bool(prefix))
                assert h == int(original.keys[s][k][0, t]), ("scalar hash", t, s, k)

    # Exact dict reference; contexts and targets exercise role reuse. Reject
    # collisions instead of allowing a vacuous replacement-policy comparison.
    hs = original.keys["first"][2].flatten().to(device)
    target = torch.as_tensor(np.roll(x, -1, axis=1).copy(), device=device).flatten()
    target = encode_targets(target, original.role_next["first"].reshape(-1, 8), len(vocab))
    symbols = len(vocab)+8+1
    table = BoundedCounts(symbols, slots=1 << 16, device=device)
    slots, exact = {}, defaultdict(Counter)
    for h, y in zip(hs, target):
        key, symbol, slot = int(h), int(y), int(table.slot(h))
        assert slot not in slots or slots[slot] == key, "tiny dict reference slot collision"
        slots[slot] = key
        before = table.read(h[None])[0].cpu().tolist()
        assert before == [exact[key][c] for c in range(symbols)]
        table.write(h[None], y[None])
        exact[key][symbol] += 1
        assert table.read(h[None])[0].cpu().tolist() == [exact[key][c] for c in range(symbols)]

    # EOF, dangling escapes and quote lookahead cannot change already emitted
    # features. These catch a lexer that requires a future closing quote.
    for fragment in ('name = "abc\\', "name = '''abc\\", 'name = """abc\\',
                     'name = ""+other', "# name\\", 'name = "abc\\\nxyz"'):
        fx = _encode(fragment, vocab)
        ff = role_features(fx, vocab, frequency_threshold=0, device=device)
        for t in range(len(fragment)):
            prefix = role_features(fx[:, :t+1], vocab, frequency_threshold=0, device=device)
            for s in SCHEMES:
                for k in ff.keys[s]:
                    assert torch.equal(ff.keys[s][k][:, :t+1], prefix.keys[s][k]), ("EOF causality", fragment, t)

    # Decoder prefix rejection, absent/new roles, duplicate proposals, and
    # causal target selection. Known prefix-end roles are invalid (-1).
    nxt = torch.tensor([[0, -1, -2], [1, 1, -1]], device=device)
    rm = torch.tensor([[2., 100., 3., 5.], [1., 2., 9., 0.]], device=device)
    lit = torch.tensor([[1., 3.], [0., 0.]], device=device)
    decoded = decode_roles(rm, lit, nxt)
    torch.testing.assert_close(decoded, torch.tensor([[5/14, 9/14], [0., 1.]], device=device))
    assert encode_targets(torch.tensor([0, 1], device=device), nxt, 2).tolist() == [2, 2]
    freq = Counter()
    trk = IdentifierTracker(freq, 2, True)
    assert trk.complete("alpha")["first"] == ROLE_BASE
    assert trk.complete("alpha")["first"] == ROLE_BASE
    assert trk.complete("alpha")["first"] == word_token("alpha")
    assert trk.complete("print")["first"] == word_token("print")

    # Production model prediction, not just features, must be causal and frozen.
    model = RoleMemoryLM(vocab, orders=(1, 2), slots=64, max_roles=8,
                         frequency_threshold=0, device=device)
    xt = torch.as_tensor(x, device=device)
    model.train_batch(xt, torch.roll(xt, -1, 1))
    before = model_digest(model)
    q = model.logprobs(xt)
    altered = xt.clone()
    altered[:, 21:] = enc["z"]
    torch.testing.assert_close(q[:, :21], model.logprobs(altered)[:, :21], rtol=0, atol=0)
    assert model_digest(model) == before
    # Concurrent colliding writes are deterministic and do not mix tags.
    tiny = BoundedCounts(3, 2, device)
    h0 = torch.arange(32, device=device, dtype=torch.int64)
    pairs = [(int(a), int(b)) for a in h0 for b in h0
             if int(a) != int(b) and int(tiny.slot(a)) == int(tiny.slot(b))]
    a, b = pairs[0]
    tiny.write(torch.tensor([a, b, a], device=device), torch.tensor([0, 1, 2], device=device))
    assert tiny.read(torch.tensor([a], device=device)).tolist() == [[1., 0., 1.]]
    assert not tiny.read(torch.tensor([b], device=device)).any()
    return dict(causality=True, rename_invariance=True, scalar_hash_reference=True,
                dict_counts=True, decoder=True, causal_frequency=True,
                model_causality=True, evaluation_frozen=True, deterministic_collisions=True)


def model_digest(model):
    digest = hashlib.sha256()
    tensors = [model.unigram, model.mixer.w]
    for table in [*model.literal.values(), *model.role.values()]:
        tensors.extend((table.tags, table.counts))
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    digest.update(repr(sorted(model.train_counts.items())).encode())
    return digest.hexdigest()


def _novelty(ce, x, y, original_y, vocab):
    """Exact iter15.Q.novel/iter30 metric: only renamed target characters."""
    sums, counts = [0., 0.], [0, 0]
    for b in range(len(x)):
        text = "".join(vocab[c] for c in [int(x[b, 0])]+y[b].tolist())
        seen = set()
        for match in re.finditer(r"[A-Za-z0-9_]+", text):
            k = int(match.group() in seen)
            for pos in range(max(1, match.start()), match.end()):
                if int(y[b, pos-1]) != int(original_y[b, pos-1]):
                    sums[k] += float(ce[b, pos-1])
                    counts[k] += 1
            seen.add(match.group())
    return sums, counts


def _process_peak_mib():
    """Process high-water RSS/working set, including CPU feature buffers."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class Info(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak_working", "working", "peak_paged", "paged",
                    "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
        info = Info()
        info.cb = ctypes.sizeof(info)
        current = ctypes.windll.kernel32.GetCurrentProcess
        current.restype = wintypes.HANDLE
        get = ctypes.windll.psapi.GetProcessMemoryInfo
        get.argtypes = [wintypes.HANDLE, ctypes.POINTER(Info), wintypes.DWORD]
        if not get(current(), ctypes.byref(info), info.cb):
            raise ctypes.WinError()
        return info.peak_working/2**20
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value/(2**20 if sys.platform == "darwin" else 1024)


@torch.no_grad()
def run_eval(args):
    # Reuse the actual benchmark and rename construction, never iter35.
    import iter30 as benchmark30
    B, I = benchmark30.B, benchmark30.I
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    path = ROOT/"results"/f'iter36_roles_{"literal" if args.literal_only else "role"}.json'
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    train, test, V = B.load()
    vocab = json.loads((ROOT/"data/meta.json").read_text(encoding="utf-8"))["vocab"]
    assert V == len(vocab)
    cut = int(len(train)*.95)
    train = train[:cut] if args.smoke is None else train[:cut][:args.smoke]
    if len(train) == 0:
        raise ValueError("empty training stream")
    renamed, rename_info = I.rename(test)
    verification = checks(vocab, I.rename, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    model = RoleMemoryLM(vocab, args.orders, args.slots, args.max_roles,
                         args.scheme, args.frequency_threshold, args.literal_only, device)
    start = time.perf_counter()
    chars, training_nll = 0, 0.
    for x, y in training_windows(train, V):
        training_nll += model.train_batch(x.to(device), y.to(device))
        chars += int((y >= 0).sum())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-start
    assert chars == len(train)
    train_peak = torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else 0.
    before = model_digest(model)
    old = B.N_EVAL_WINDOWS
    start = time.perf_counter()
    try:
        if args.smoke is not None:
            B.N_EVAL_WINDOWS = 16
        ox, oy = B.eval_windows(test)
        evaluated, novelty = {}, {}
        for name, data in (("test", test), ("renamed", renamed)):
            x, y = B.eval_windows(data)
            total, sums, counts = 0., [0., 0.], [0, 0]
            for a in range(0, len(x), 16):
                lp = model.logprobs(x[a:a+16].to(device))
                ce = -lp.gather(-1, y[a:a+16].to(device)[..., None]).squeeze(-1).cpu()
                total += float(ce.sum())
                if name == "renamed":
                    ss, nn = _novelty(ce, x[a:a+16], y[a:a+16], oy[a:a+16], vocab)
                    sums = [u+v for u, v in zip(sums, ss)]
                    counts = [u+v for u, v in zip(counts, nn)]
            evaluated[name] = total/y.numel()
            if name == "renamed":
                novelty = dict(novel_first_loss=sums[0]/counts[0] if counts[0] else None,
                               novel_repeat_loss=sums[1]/counts[1] if counts[1] else None,
                               novel_counts=dict(first=counts[0], repeat=counts[1]))
        assert model_digest(model) == before, "evaluation changed model"
        windows = B.N_EVAL_WINDOWS
    finally:
        B.N_EVAL_WINDOWS = old
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    eval_seconds = time.perf_counter()-start
    peak = torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else 0.
    result = dict(**evaluated, gap=evaluated["renamed"]-evaluated["test"],
                  test_loss=evaluated["test"], renamed_loss=evaluated["renamed"],
                  renamed_gap=evaluated["renamed"]-evaluated["test"], **novelty,
                  train_seconds=train_seconds, eval_seconds=eval_seconds,
                  training_loss=training_nll/chars, chars=chars,
                  peak_allocated_mib=peak, train_peak_allocated_mib=train_peak,
                  peak_process_mib=_process_peak_mib(), memory_bytes=model.memory_bytes(),
                  references=REFERENCES, checks=verification,
                  settings=dict(device=str(device), smoke=args.smoke, seed=0,
                                train_end=cut, trained_prefix_chars=len(train),
                                batch_size=16, window=128, eval_windows=windows,
                                sampling="iter30.R.epoch identical CPU permutation/partition",
                                evaluation_frozen=True, literal_only=args.literal_only,
                                scheme=args.scheme, orders=args.orders, slots=args.slots,
                                max_roles=args.max_roles, frequency_threshold=args.frequency_threshold,
                                frequency_policy="counts from scored earlier batches only",
                                partial_state="length; completed tokens exclude unfinished word",
                                recency_policy="pre-touch rank at emission, historical ranks frozen",
                                new_name_policy="literal write and literal decoding fallback",
                                renaming=rename_info,
                                invariance_scope="nonliteral identifiers, outside identifier characters; frequency promotion may break invariance"))
    path.parent.mkdir(exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--checks", action="store_true")
    parser.add_argument("--smoke", type=int)
    parser.add_argument("--literal-only", action="store_true")
    parser.add_argument("--scheme", choices=SCHEMES, default="first")
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--slots", type=int, default=4096)
    parser.add_argument("--max-roles", type=int, default=32)
    parser.add_argument("--frequency-threshold", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.smoke is not None and args.smoke < 1:
        parser.error("--smoke must be positive")
    if not args.eval and not args.checks:
        parser.error("choose --eval or --checks")
    if args.eval:
        result = run_eval(args)
    else:
        device = args.device if args.device != "auto" else "cpu"
        result = checks(device=device)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
