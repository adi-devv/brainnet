"""iter40 predictor with bounded fused-read scratch, without backprop.

Every chunk is scored with old parameters/counts, then learned/written once.
No autograd is used, including in the small keys module (its own exact CE
derivative is written explicitly). Evaluation uses iter30's actual functions.
Smoke results are retained, explicitly marked ineligible for the full-data bars.
Budgeted runs use a separately timed first-10% allocation probe and restart the
unchanged single training pass. See briefs/iter41_codex_out.md for verification.
"""
import os
import sys
sys.dont_write_bytecode = True
import argparse
import copy
import gc
import json
import math
import time
import weakref
from collections import Counter
from dataclasses import dataclass, asdict

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
CAP = int(os.environ.get('CAP', '0'))
TARGETS = dict(test=1.2562, renamed=1.5714, val=1.3869,
               train_seconds=79.4, peak_allocated_mib=151.)

# Execution workspaces are external to predictor state: frozen reads may reuse
# them without mutating the model, its diagnostics, or its persistent budget.
# Weak ownership releases each graph/private pool when its store is discarded
# (in particular between the allocation probe and the real training pass).
READ_GRAPHS = weakref.WeakKeyDictionary()
ROW_GRAPHS = weakref.WeakKeyDictionary()
READ_BLOCK = 512


class ReadGraph:
    """One fixed-shape, read-only graph; outputs are borrowed until next replay."""
    def __init__(self, store, code):
        self.code = torch.empty_like(code)
        self.code.copy_(code)
        stream = torch.cuda.Stream(device=code.device)
        stream.wait_stream(torch.cuda.current_stream(code.device))
        with torch.cuda.stream(stream):
            # Warm up CUDA operators off the capture stream. No table/stat writes.
            for _ in range(3):
                output = store.read_block(self.code)
            del output
        torch.cuda.current_stream(code.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = store.read_block(self.code)

    def replay(self, code):
        self.code.copy_(code)
        self.graph.replay()
        return self.output


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
        self.collect, self.diag_every = False, 16
        self.stats = torch.zeros(4, dtype=torch.int64, device=device)

    def read(self, code):
        if self.collect:
            sample = self.count[code[::self.diag_every]]
            self.stats[0] += sample.numel()
            self.stats[1] += (sample > 0).sum()
        return self.count[code].float()

    def write(self, code, y, scratch=None):
        self.stats[2] += len(y)
        self.count.view(-1).index_add_(0, code*self.V+y,
                                      torch.ones_like(y, dtype=torch.int32))

    def memory(self):
        return dict(counts=nbytes(self.count), diagnostics=nbytes(self.stats))

    def diagnostics(self):
        queries, hits, writes, _ = self.stats.cpu().tolist()
        occupied = int((self.count > 0).sum())
        return dict(hit_rate=hits/max(1, queries), sampled_read_pairs=queries,
                    sampled_hits=hits, eviction_count=0, write_events=writes,
                    admitted_events=writes, admitted_fraction=1. if writes else 0.,
                    final_occupancy=occupied, slots=self.count.numel(),
                    occupancy_fraction=occupied/self.count.numel(),
                    bytes=sum(self.memory().values()))


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


@dataclass(frozen=True)
class StorageOptions:
    storage: str = 'tagged'
    admit: str = 'none'
    budget_mib: float | None = None
    ways: int = 4
    age_chunks: int = 0
    door_reset_chunks: int = 256
    diag_every: int = 16
    stale_every: int = 64


def avalanche(h):
    """Full iter5 avalanche, without its 25-bit address mask."""
    h = (h ^ ((h >> 33) & 0x7fffffff)) * -49064778989728563
    h = (h ^ ((h >> 33) & 0x7fffffff)) * -4265267296055464877
    return h ^ ((h >> 33) & 0x7fffffff)


def uniform_hash(h, step=0, salt=0):
    # Stateless random rounding: reproducible, no consumption of the model RNG.
    nonce = (step*0x1e35a7bd + salt*0x519b3d27) & 0x7fffffffffffffff
    return (avalanche(h ^ nonce) & 0xffffff).float() / 16777216.


class LogCount:
    """One byte: exact 0..32, then base-1.06 levels, saturating at code 255.

    Round between adjacent decoded levels with the unbiased probability. Adding
    a chunk multiplicity in one operation is unbiased too (until saturation).
    A stochastic counter has no per-realization 10% error guarantee; the checks
    test both quantization error and the mean of independently rounded counters.
    """
    # The last level exceeds the entire 6.2M-target pass, so common pairs do
    # not clip at the 1e4-count range used by the accuracy check.
    threshold, base = 32, 1.06

    def __init__(self, device):
        c = torch.arange(256, dtype=torch.float32, device=device)
        self.levels = torch.where(c <= self.threshold, c,
                                 self.threshold+(self.base**(c-self.threshold)-1)/(self.base-1))

    def decode(self, c):
        return self.levels[c.long()]

    def encode(self, x, u):
        ceiling = self.threshold+(self.base**(255-self.threshold)-1)/(self.base-1)
        x = x.clamp(0, ceiling)
        logcode = self.threshold + torch.log1p((x-self.threshold).clamp_min(0)*
                                               (self.base-1))/math.log(self.base)
        lo = torch.where(x <= self.threshold, x, logcode).floor().long().clamp(0, 255)
        hi = (lo+1).clamp_max(255)
        p = ((x-self.levels[lo])/(self.levels[hi]-self.levels[lo]).clamp_min(1e-12)).clamp(0, 1)
        return (lo+((u < p) & (lo < 255)).long()).to(torch.uint8)


class Doorkeeper:
    """Two Bloom probes, two bits per storage slot, genuinely bit packed.

    Within a chunk, exact duplicates after the first occurrence are admitted.
    Other Bloom decisions use the pre-chunk bits. Bit updates elect each BIT
    once, then integer-add disjoint masks per byte: no lost concurrent bit sets.
    """
    def __init__(self, slots, device):
        self.nbits = max(64, slots*2)
        self.bits = torch.zeros(self.nbits//8, dtype=torch.uint8, device=device)

    def admit(self, h):
        a = h & (self.nbits-1)
        b = ((h >> 32) ^ h*P) & (self.nbits-1)
        present = lambda p: (self.bits[p >> 3].long() & (1 << (p & 7))) != 0
        allowed = present(a) & present(b)
        order = h.argsort(stable=True)
        duplicate = h[order[1:]] == h[order[:-1]]
        allowed[order[1:]] |= duplicate
        # unique_consecutive is required for the storage bit update, never a
        # diagnostic scan. Its input is at most twice the chunk's target count.
        positions = torch.cat((a, b)).sort().values.unique_consecutive()
        byte, masks = positions >> 3, (1 << (positions & 7)).long()
        cells, inverse = byte.unique_consecutive(return_inverse=True)
        update = torch.zeros_like(cells).index_add_(0, inverse, masks)
        self.bits[cells] = (self.bits[cells].long() | update).to(torch.uint8)
        return allowed


class StorageDiagnostics:
    """Sample reads, and sample 1/64 of physical slots for live usefulness.

    All counters remain on device during training. Evaluation never changes
    diagnostics. A slot's usefulness bit is cleared when its occupant changes.
    """
    def init_diagnostics(self, slots, V, device, options):
        self.options, self.collect, self.writes = options, False, 0
        self.stats = torch.zeros(7, dtype=torch.int64, device=device)
        self.useful = torch.zeros((slots+63)//64, dtype=torch.uint8, device=device)
        self.door = Doorkeeper(slots, device) if options.admit == 'doorkeeper' else None

    def record_hits(self, ix, hit, count_queries=True):
        # The caller passes only every diag_every-th context's read positions.
        if count_queries:
            self.stats[0] += hit.numel()
        self.stats[1] += hit.sum()
        sample = hit & ((ix & 63) == (((ix >> 6)*P) & 63))
        # Duplicate indices all store 1, so this assignment is deterministic.
        self.useful[ix[sample] >> 6] = 1

    def clear_useful(self, ix):
        selected = ix[(ix & 63) == (((ix >> 6)*P) & 63)] >> 6
        self.useful[selected] = 0

    def admission(self, code, y, resident=None):
        self.writes += 1
        if self.door and (self.writes-1) % self.options.door_reset_chunks == 0:
            self.door.bits.zero_()
        self.stats[3] += len(y)
        if self.door:
            h = avalanche(code*P+y+1)
            keep = self.door.admit(h)
            if resident is not None:
                # Admission gates insertion, never an existing entry's update,
                # including the first update after a periodic Bloom reset.
                keep |= resident
        else:
            keep = torch.ones_like(y, dtype=torch.bool)
        self.stats[4] += keep.sum()
        return keep

    def diagnostic_memory(self):
        out = dict(diagnostics=nbytes(self.stats), useful_sample=nbytes(self.useful))
        if self.door:
            out['doorkeeper'] = nbytes(self.door.bits)
        return out

    def diagnostics(self):
        queries, hits, evictions, writes, admits, dropped, aged = self.stats.cpu().tolist()
        occupied = int((self.count > 0).sum())
        # Only live sampled occupants count toward the allocation measurement.
        groups = torch.arange(len(self.useful), device=self.count.device)
        sample_ix = groups*64+((groups*P) & 63)
        live_useful = int(((self.count[sample_ix] > 0) & self.useful.bool()).sum())
        return dict(hit_rate=hits/max(1, queries), sampled_read_pairs=queries,
                    sampled_hits=hits, eviction_count=evictions,
                    write_events=writes, admitted_events=admits,
                    admitted_fraction=admits/max(1, writes),
                    chunk_collision_dropped_events=dropped, aged_out_entries=aged,
                    final_occupancy=occupied, slots=self.count.numel(),
                    occupancy_fraction=occupied/self.count.numel(),
                    sampled_useful_live_entries=live_useful,
                    useful_entries_estimate=live_useful*64,
                    useful_slot_sampling_stride=64,
                    read_context_sampling_stride=self.options.diag_every,
                    bytes=sum(self.memory().values()))


class MeasuredTagged(Tagged, StorageDiagnostics):
    """The original direct-map election, with optional Bloom admission."""
    def __init__(self, bits, V, device, options):
        super().__init__(bits, V, device)
        self.init_diagnostics(1 << bits, V, device, options)

    def read(self, code):
        ix, tag = self.address(code[:, None], self.chars)
        hit = (self.tag[ix] == tag) & (self.count[ix] > 0)
        if self.collect:
            s = slice(None, None, self.options.diag_every)
            self.record_hits(ix[s], hit[s])
        return (self.count[ix]*hit).float()

    def commit(self, ix, tag, scratch):
        # Coalesce writes as part of the storage algorithm. One winning record
        # per cell gives exact eviction counts without a diagnostic unique().
        key = (ix.long() << 31) | tag.long()
        keys, multiplicity = key.sort().values.unique_consecutive(return_counts=True)
        cells, tags = keys >> 31, (keys & 0x7fffffff).int()
        scratch.index_fill_(0, cells, 0)
        scratch.scatter_reduce_(0, cells, tags, reduce='amax', include_self=True)
        keep = tags == scratch[cells]
        self.stats[5] += multiplicity[~keep].sum()
        cells, tags, multiplicity = cells[keep], tags[keep], multiplicity[keep]
        changed = self.tag[cells] != tags
        self.stats[2] += (changed & (self.count[cells] > 0)).sum()
        self.clear_useful(cells[changed])
        self.count[cells] = torch.where(changed, 0, self.count[cells])+multiplicity.int()
        self.tag[cells] = tags

    def write(self, code, y, scratch):
        ix, tag = self.address(code, y)
        resident = (self.tag[ix] == tag) & (self.count[ix] > 0) if self.door else None
        keep = self.admission(code, y, resident)
        self.commit(ix[keep], tag[keep], scratch)

    def memory(self):
        return dict(**super().memory(), **self.diagnostic_memory())


@torch.jit.script
def sequential_victims(old: torch.Tensor, incoming: torch.Tensor) -> torch.Tensor:
    """Scripted local correction for buckets where a new entry can be re-evicted.

    Only local [bucket, way] scratch changes here; no global table writes,
    hashing, encoding, admission, or Python turn-by-turn dispatch.
    """
    victims = torch.empty_like(incoming, dtype=torch.long)
    if old.size(0) == 0:
        return victims
    current = old.clone()
    for turn in range(incoming.size(1)):
        way = current.to(torch.int32).argmin(-1)
        victims[:, turn] = way
        current.scatter_(1, way[:, None], incoming[:, turn:turn+1])
    return victims


class Buckets(StorageDiagnostics):
    """W-way storage, uint8/int16 compact or int32/int32 ablation.

    A chunk coalesces (bucket,fingerprint) duplicates, updates resident hits,
    then handles up to W distinct misses per bucket in descending fingerprint
    order. Each miss replaces the current lowest count (ties: lowest way),
    including entries inserted earlier in the chunk. Excess collisions are discarded.
    Encoding is batched; only the final writer to each victim scatters to storage.
    The usual distinct-victim path sorts old ways once; a scripted local recurrence
    preserves sequential semantics for buckets that might re-evict new entries.
    Counts are halved every age_chunks writes (0 disables aging).
    No full hashes survive a write: budget allocation is done by a replayed
    first-10% probe, rather than claiming fingerprints suffice for rehashing.
    """
    def __init__(self, bits, V, device, options):
        self.V, self.ways = V, options.ways
        slots = 1 << bits
        self.mask = slots//self.ways-1
        self.compact = options.storage == 'compact'
        self.tagbits = 16 if self.compact else 31
        self.count = torch.zeros(slots, dtype=torch.uint8 if self.compact else torch.int32,
                                 device=device)
        self.tag = torch.zeros(slots, dtype=torch.int16 if self.compact else torch.int32,
                               device=device)
        self.chars = torch.arange(V, device=device)
        self.codec = LogCount(device) if self.compact else None
        self.init_diagnostics(slots, V, device, options)

    def address(self, h, y):
        h = avalanche(h*P+y+1)
        bucket = h & self.mask
        tag = (h >> 32) & ((1 << self.tagbits)-1)
        if self.compact:
            tag = (tag+32768) % 65536-32768
        return bucket, tag.to(self.tag.dtype)

    def decode(self, c):
        return self.codec.decode(c) if self.codec else c.float()

    def read(self, code):
        bucket, tag = self.address(code[:, None], self.chars)
        result = torch.zeros(bucket.shape, dtype=torch.float32, device=code.device)
        if self.collect:
            self.stats[0] += bucket[::self.options.diag_every].numel()
        # Gather up to four ways together to reduce launch overhead. A 16-way
        # ablation still bounds address scratch at N*V*4, rather than N*V*16.
        for left in range(0, self.ways, 4):
            ways = torch.arange(left, min(left+4, self.ways), device=code.device)
            ix = bucket[..., None]*self.ways+ways
            c = self.count[ix]
            hit = (self.tag[ix] == tag[..., None]) & (c > 0)
            result.add_(self.decode(torch.where(hit, c, 0).amax(-1)))
            if self.collect:
                s = slice(None, None, self.options.diag_every)
                self.record_hits(ix[s], hit[s], count_queries=False)
        return result

    def put_count(self, ix, value, identity, salt=0):
        if self.codec:
            self.count[ix] = self.codec.encode(value, uniform_hash(identity, self.writes, salt))
        else:
            self.count[ix] = value.to(torch.int32)

    def age(self):
        # Eight large slabs for a 4M-slot table, instead of 64 small slabs.
        # Half-count rounding depends on just 256 codes. Compute its thresholds
        # once, then retain iter38's per-slot identity/nonce/random comparison.
        if self.codec:
            half = self.codec.levels*.5
            low = self.codec.encode(half, torch.ones_like(half)).long()
            high = (low+1).clamp_max(255)
            probability = ((half-self.codec.levels[low])/
                           (self.codec.levels[high]-self.codec.levels[low]).clamp_min(1e-12)).clamp(0, 1)
        slab = 1 << 19
        for left in range(0, self.count.numel(), slab):
            ix = torch.arange(left, min(left+slab, self.count.numel()), device=self.count.device)
            old = self.count[ix]
            if self.codec:
                u = uniform_hash(ix ^ self.tag[ix].long(), self.writes, salt=1)
                code = old.long()
                new = (low[code]+((u < probability[code]) & (low[code] < 255)).long()).to(torch.uint8)
            else:
                new = (old.float()*.5).to(torch.int32)
            self.count[ix] = new
            self.stats[6] += ((old > 0) & (new == 0)).sum()

    def write(self, code, y, scratch=None):
        bucket, tag = self.address(code, y)
        offsets = torch.arange(self.ways, device=code.device)
        if self.door:
            ix = bucket[:, None]*self.ways+offsets
            resident = ((self.tag[ix] == tag[:, None]) & (self.count[ix] > 0)).any(-1)
            allowed = self.admission(code, y, resident)
            bucket, tag = bucket[allowed], tag[allowed]
        else:
            # No resident admission scan, all-ones mask, filtering, or sum.
            self.writes += 1
            self.stats[3:5] += len(y)
        key = (bucket << self.tagbits) | (tag.long() & ((1 << self.tagbits)-1))
        keys, adds = key.sort(descending=True).values.unique_consecutive(return_counts=True)
        bucket = keys >> self.tagbits
        tag = (keys & ((1 << self.tagbits)-1)).to(self.tag.dtype)
        # Coalesced records are unique, so resident-hit assignments cannot race.
        ix = bucket[:, None]*self.ways+offsets
        counts = self.count[ix]
        matches = (self.tag[ix] == tag[:, None]) & (counts > 0)
        # amax retains iter38's highest-way choice even for duplicate resident tags.
        slot = torch.where(matches, ix, -1).amax(-1)
        hit = slot >= 0
        hit_slot, hit_keys = slot[hit], keys[hit]
        hit_value = self.decode(self.count[hit_slot])+adds[hit]
        bucket, tag, adds, keys = bucket[~hit], tag[~hit], adds[~hit], keys[~hit]
        pos = torch.arange(len(bucket), device=bucket.device)
        first = torch.ones_like(bucket, dtype=torch.bool)
        first[1:] = bucket[1:] != bucket[:-1]
        rank = pos-torch.where(first, pos, 0).cummax(0).values
        self.stats[5] += adds[rank >= self.ways].sum()
        keep = rank < self.ways
        bucket, tag, adds, keys, rank, first = (a[keep] for a in
                                               (bucket, tag, adds, keys, rank, first))
        # Stateless rounding depends on identity and write number, never victim
        # or turn. Encode hits and ALL accepted misses exactly once.
        value = torch.cat((hit_value, adds.float()))
        identity = torch.cat((hit_keys, keys))
        encoded = (self.codec.encode(value, uniform_hash(identity, self.writes))
                   if self.codec else value.to(torch.int32))
        self.count[hit_slot] = encoded[:len(hit_slot)]
        incoming = encoded[len(hit_slot):]

        group = first.long().cumsum(0)-1
        buckets = bucket[first]
        last = torch.ones_like(first)
        last[:-1] = bucket[:-1] != bucket[1:]
        nmiss = rank[last]+1
        ix = buckets[:, None]*self.ways+offsets
        old = self.count[ix]  # AFTER resident updates, as in iter38.
        ordered = old.int().argsort(dim=-1, stable=True)
        way = ordered[group, rank]
        # Sufficient exact fast-path condition: every non-final insertion sorts
        # after the last old victim. Otherwise use the local sequential oracle.
        tail_way = ordered.gather(1, (nmiss-1)[:, None]).squeeze(1)
        tail_count = old.gather(1, tail_way[:, None]).squeeze(1)
        risk = ((rank < nmiss[group]-1) &
                ((incoming < tail_count[group]) |
                 ((incoming == tail_count[group]) & (way < tail_way[group]))))
        unsafe = torch.zeros_like(buckets).scatter_reduce_(0, group, risk.long(),
                                                          reduce='amax').bool()
        values = torch.zeros_like(old)
        values[group, rank] = incoming
        chosen = ordered.clone()
        chosen[unsafe] = sequential_victims(old[unsafe], values[unsafe])
        way = chosen[group, rank]

        # New counts are >=1. Zeros are consumed first, then every replacement
        # is an eviction, even when it evicts an earlier miss from this chunk.
        self.stats[2] += (rank >= (old == 0).sum(-1)[group]).sum()
        # A repeated victim must have only its LAST writer scattered. Avoid
        # conflicting-value writes even when a low-count insertion is re-evicted.
        final_rank = torch.full_like(ix, -1)
        final_rank.view(-1).scatter_reduce_(0, group*self.ways+way, rank,
                                           reduce='amax')
        final = rank == final_rank[group, way]
        victim = bucket[final]*self.ways+way[final]
        self.clear_useful(victim)
        self.tag[victim] = tag[final]
        self.count[victim] = incoming[final]
        if self.options.age_chunks and self.writes % self.options.age_chunks == 0:
            self.age()

    def memory(self):
        out = dict(counts=nbytes(self.count), tags=nbytes(self.tag),
                   character_indices=nbytes(self.chars), **self.diagnostic_memory())
        if self.codec:
            out['count_decode_lut'] = nbytes(self.codec.levels)
        return out


def avalanche_inplace(h):
    """Same integer hash on PRIVATE scratch, with one full-size temporary."""
    h.bitwise_xor_((h >> 33).bitwise_and_(0x7fffffff)).mul_(-49064778989728563)
    h.bitwise_xor_((h >> 33).bitwise_and_(0x7fffffff)).mul_(-4265267296055464877)
    return h.bitwise_xor_((h >> 33).bitwise_and_(0x7fffffff))


class BucketView(Buckets):
    """Per-table reporting view; Model reads/writes through FusedBuckets."""
    @property
    def writes(self):
        return self.owner.writes


class FusedBuckets:
    """Disjoint bucket ranges, one read and write dispatch for every table.

    Only addresses use global buckets. Rounding, Bloom probes, usefulness and
    aging retain table-local identities. Views own no separate count/tag store.
    """
    def __init__(self, sizes, V, device, options):
        self.sizes, self.V, self.options = tuple(sizes.items()), V, options
        self.ways, self.writes, self.collect = options.ways, 0, False
        self.compact = options.storage == 'compact'
        self.tagbits = 16 if self.compact else 31
        rows, slot, useful, bloom = [], 0, 0, 0
        for _, slots in self.sizes:
            nbits = max(64, slots*2)
            rows.append((slots//self.ways-1, slot//self.ways, slot, slots,
                         useful, bloom, nbits))
            slot += slots
            useful += (slots+63)//64
            bloom += nbits
        if slot//self.ways >= min(2**31, 2**(63-self.tagbits)):
            raise ValueError('fused bucket addresses exceed the signed index/key range')
        # Transpose on the host: each metadata row/bucket boundary is contiguous.
        self.layout = torch.tensor(list(zip(*rows)), dtype=torch.long, device=device)
        self.count = torch.zeros(slot, dtype=torch.uint8 if self.compact else torch.int32,
                                 device=device)
        self.tag = torch.zeros(slot, dtype=torch.int16 if self.compact else torch.int32,
                               device=device)
        self.stats = torch.zeros((len(rows), 7), dtype=torch.long, device=device)
        self.useful = torch.zeros(useful, dtype=torch.uint8, device=device)
        self.bloom = (torch.zeros(bloom//8, dtype=torch.uint8, device=device)
                      if options.admit == 'doorkeeper' else None)
        self.codec, self.chars = None, None

    def views(self):
        tables, slot, useful, bloom = {}, 0, 0, 0
        for i, (name, slots) in enumerate(self.sizes):
            t = BucketView.__new__(BucketView)
            t.owner, t.V, t.ways = self, self.V, self.ways
            t.mask, t.compact, t.tagbits = slots//self.ways-1, self.compact, self.tagbits
            t.count, t.tag = self.count[slot:slot+slots], self.tag[slot:slot+slots]
            t.chars = torch.arange(self.V, device=self.count.device)
            t.codec = LogCount(self.count.device) if self.compact else None
            t.options, t.collect, t.stats = self.options, False, self.stats[i]
            t.useful = self.useful[useful:useful+(slots+63)//64]
            t.door = None
            if self.bloom is not None:
                t.door = Doorkeeper.__new__(Doorkeeper)
                t.door.nbits = max(64, slots*2)
                t.door.bits = self.bloom[bloom//8:(bloom+t.door.nbits)//8]
            tables[name] = t
            slot += slots
            useful += (slots+63)//64
            bloom += max(64, slots*2)
        self.codec, self.chars = t.codec, t.chars
        return tables

    def owners(self, bucket):
        return torch.bucketize(bucket, self.layout[1, 1:], right=True)

    def address(self, code, y):
        h = avalanche_inplace(code*P+y+1)
        shape = (-1,)+((1,)*(h.ndim-1))
        bucket = (h & self.layout[0].reshape(shape)).int()
        bucket.add_(self.layout[1].reshape(shape).int())
        # h is private scratch. Reuse it rather than holding two more full
        # int64 read-address temporaries at the 4096-target memory peak.
        h.bitwise_right_shift_(32).bitwise_and_((1 << self.tagbits)-1)
        # Narrowing reproduces the original signed int16 representation.
        return bucket, h.to(self.tag.dtype)

    def decode(self, c):
        return self.codec.decode(c) if self.codec else c.float()

    def read_block(self, code):
        bucket, tag = self.address(code[..., None], self.chars)
        ways = torch.arange(self.ways, dtype=torch.int32, device=code.device)
        # Broadcasting avoids materializing T*N*V*W physical int64 indices.
        hit = self.tag.view(-1, self.ways)[bucket[..., None], ways] == tag[..., None]
        del tag
        c = self.count.view(-1, self.ways)[bucket[..., None], ways]
        hit &= c > 0
        c.masked_fill_(~hit, 0)
        # Retain iter39's max within groups of four and ordered group additions,
        # even if duplicate resident tags are present in several groups.
        width = min(4, self.ways)
        encoded = c.reshape(*c.shape[:-1], self.ways//width, width).amax(-1)
        del c
        decoded = self.decode(encoded)
        result = decoded[..., 0].clone()
        for group in range(1, self.ways//width):
            result.add_(decoded[..., group])
        return result, hit, bucket

    def read(self, code, out=None):
        """Bound hashing/gather scratch while retaining global sample positions.

        Reads are independent across targets; only the diagnostic integer sum
        spans blocks. Writes still execute once for the entire original chunk.
        """
        tables, n = code.shape
        if out is None:
            out = torch.empty((tables, n, self.V), device=code.device)
        stride = self.options.diag_every
        if self.collect:
            samples = (n+stride-1)//stride
            sampled = torch.empty((tables, samples, self.V, self.ways),
                                  dtype=torch.bool, device=code.device)
            buckets = torch.empty((tables, samples, self.V),
                                  dtype=torch.int32, device=code.device)
        for left in range(0, n, READ_BLOCK):
            right = min(left+READ_BLOCK, n)
            block = code[:, left:right]
            if (code.is_cuda and right-left == READ_BLOCK and
                    os.environ.get('ITER41_GRAPHS', '1') != '0'):
                graph = READ_GRAPHS.get(self)
                if graph is None:
                    graph = ReadGraph(self, block)
                    READ_GRAPHS[self] = graph
                values, hit, bucket = graph.replay(block)
            else:
                values, hit, bucket = self.read_block(block)
            out[:, left:right].copy_(values)
            if self.collect:
                # The first sample in a block need not be its first target.
                first = (-left) % stride
                start, stop = (left+first)//stride, (right+stride-1)//stride
                sampled[:, start:stop].copy_(hit[:, first::stride])
                buckets[:, start:stop].copy_(bucket[:, first::stride])
            del values, hit, bucket
        if self.collect:
            self.stats[:, 0] += samples*self.V
            self.stats[:, 1] += sampled.sum((1, 2, 3))
            ways = torch.arange(self.ways, dtype=torch.int32, device=code.device)
            local = ((buckets.long()-self.layout[1, :, None, None])
                     [..., None]*self.ways+ways)
            selected = sampled & ((local & 63) == (((local >> 6)*P) & 63))
            ui = (local >> 6)+self.layout[4, :, None, None, None]
            self.useful[ui[selected]] = 1
        return out

    def admit(self, h, bucket, tag):
        """The same independent Bloom filters, with global packed bit positions."""
        if (self.writes-1) % self.options.door_reset_chunks == 0:
            self.bloom.zero_()
        a = h & (self.layout[6, :, None]-1)
        b = ((h >> 32) ^ h*P) & (self.layout[6, :, None]-1)
        a += self.layout[5, :, None]
        b += self.layout[5, :, None]
        present_a = (self.bloom[a >> 3].long() & (1 << (a & 7))) != 0
        present_b = (self.bloom[b >> 3].long() & (1 << (b & 7))) != 0
        allowed = present_a & present_b
        order = h.argsort(dim=1, stable=True)
        sorted_h = h.gather(1, order)
        duplicate = torch.zeros_like(allowed)
        duplicate[:, 1:] = sorted_h[:, 1:] == sorted_h[:, :-1]
        allowed |= torch.zeros_like(allowed).scatter_(1, order, duplicate)
        positions = torch.cat((a.flatten(), b.flatten())).sort().values.unique_consecutive()
        byte, masks = positions >> 3, (1 << (positions & 7)).long()
        cells, inverse = byte.unique_consecutive(return_inverse=True)
        update = torch.zeros_like(cells).index_add_(0, inverse, masks)
        self.bloom[cells] = (self.bloom[cells].long() | update).to(torch.uint8)
        ways = torch.arange(self.ways, device=h.device)
        ix = bucket[..., None].long()*self.ways+ways
        allowed |= ((self.tag[ix] == tag[..., None]) & (self.count[ix] > 0)).any(-1)
        self.stats[:, 4] += allowed.sum(-1)
        return allowed.flatten()

    def clear_useful(self, ix):
        owner = self.owners(ix//self.ways)
        local = ix-self.layout[2, owner]
        keep = (local & 63) == (((local >> 6)*P) & 63)
        self.useful[(local[keep] >> 6)+self.layout[4, owner[keep]]] = 0

    def age(self):
        if self.codec:
            half = self.codec.levels*.5
            low = self.codec.encode(half, torch.ones_like(half)).long()
            high = (low+1).clamp_max(255)
            probability = ((half-self.codec.levels[low])/
                           (self.codec.levels[high]-self.codec.levels[low]).clamp_min(1e-12)).clamp(0, 1)
        for left in range(0, len(self.count), 1 << 19):
            ix = torch.arange(left, min(left+(1 << 19), len(self.count)), device=self.count.device)
            owner = self.owners(ix//self.ways)
            old = self.count[ix]
            if self.codec:
                local = ix-self.layout[2, owner]
                u = uniform_hash(local ^ self.tag[ix].long(), self.writes, salt=1)
                code = old.long()
                new = (low[code]+((u < probability[code]) & (low[code] < 255)).long()).to(torch.uint8)
            else:
                new = (old.float()*.5).to(torch.int32)
            self.count[ix] = new
            self.stats[:, 6].index_add_(0, owner, ((old > 0) & (new == 0)).long())

    def write(self, code, y):
        # Reuse this avalanche for optional Bloom probes as well.
        h = avalanche_inplace(code*P+y+1)
        bucket = ((h & self.layout[0, :, None])+self.layout[1, :, None]).int()
        tag = ((h >> 32) & ((1 << self.tagbits)-1)).to(self.tag.dtype)
        self.writes += 1
        self.stats[:, 3] += code.shape[1]
        if self.bloom is not None:
            allowed = self.admit(h, bucket, tag)
            bucket, tag = bucket.flatten()[allowed], tag.flatten()[allowed]
        else:
            self.stats[:, 4] += code.shape[1]
            bucket, tag = bucket.flatten(), tag.flatten()
        del h
        offsets = torch.arange(self.ways, device=code.device)
        key = (bucket.long() << self.tagbits) | (tag.long() & ((1 << self.tagbits)-1))
        keys, adds = key.sort(descending=True).values.unique_consecutive(return_counts=True)
        bucket = keys >> self.tagbits
        owner = self.owners(bucket)
        # Offset is excluded from stochastic rounding, exactly as in iter39.
        identity = keys-(self.layout[1, owner] << self.tagbits)
        tag = (keys & ((1 << self.tagbits)-1)).to(self.tag.dtype)
        ix = bucket[:, None]*self.ways+offsets
        counts = self.count[ix]
        matches = (self.tag[ix] == tag[:, None]) & (counts > 0)
        slot = torch.where(matches, ix, -1).amax(-1)
        hit = slot >= 0
        hit_slot, hit_keys = slot[hit], identity[hit]
        hit_value = self.decode(self.count[hit_slot])+adds[hit]
        bucket, tag, adds, identity, owner = (a[~hit] for a in
                                             (bucket, tag, adds, identity, owner))
        pos = torch.arange(len(bucket), device=bucket.device)
        first = torch.ones_like(bucket, dtype=torch.bool)
        first[1:] = bucket[1:] != bucket[:-1]
        rank = pos-torch.where(first, pos, 0).cummax(0).values
        self.stats[:, 5].index_add_(0, owner, torch.where(rank >= self.ways, adds, 0))
        keep = rank < self.ways
        bucket, tag, adds, identity, rank, first, owner = (a[keep] for a in
                                      (bucket, tag, adds, identity, rank, first, owner))
        value = torch.cat((hit_value, adds.float()))
        identity = torch.cat((hit_keys, identity))
        encoded = (self.codec.encode(value, uniform_hash(identity, self.writes))
                   if self.codec else value.to(torch.int32))
        self.count[hit_slot] = encoded[:len(hit_slot)]
        incoming = encoded[len(hit_slot):]
        group = first.long().cumsum(0)-1
        buckets = bucket[first]
        last = torch.ones_like(first)
        last[:-1] = bucket[:-1] != bucket[1:]
        nmiss = rank[last]+1
        ix = buckets[:, None]*self.ways+offsets
        old = self.count[ix]
        ordered = old.int().argsort(dim=-1, stable=True)
        way = ordered[group, rank]
        tail_way = ordered.gather(1, (nmiss-1)[:, None]).squeeze(1)
        tail_count = old.gather(1, tail_way[:, None]).squeeze(1)
        risk = ((rank < nmiss[group]-1) &
                ((incoming < tail_count[group]) |
                 ((incoming == tail_count[group]) & (way < tail_way[group]))))
        unsafe = torch.zeros_like(buckets).scatter_reduce_(0, group, risk.long(),
                                                          reduce='amax').bool()
        values = torch.zeros_like(old)
        values[group, rank] = incoming
        chosen = ordered.clone()
        chosen[unsafe] = sequential_victims(old[unsafe], values[unsafe])
        way = chosen[group, rank]
        self.stats[:, 2].index_add_(0, owner, (rank >= (old == 0).sum(-1)[group]).long())
        final_rank = torch.full_like(ix, -1)
        final_rank.view(-1).scatter_reduce_(0, group*self.ways+way, rank, reduce='amax')
        final = rank == final_rank[group, way]
        victim = bucket[final]*self.ways+way[final]
        self.clear_useful(victim)
        self.tag[victim] = tag[final]
        self.count[victim] = incoming[final]
        if self.options.age_chunks and self.writes % self.options.age_chunks == 0:
            self.age()


@torch.jit.script
def table_proposals(counts: torch.Tensor, p: torch.Tensor, raw_count: int):
    """Original arithmetic; overwrite disposable counts after each row sum."""
    totals = torch.empty(counts.shape[:2], dtype=counts.dtype, device=counts.device)
    high = p
    for j in range(counts.size(0)):
        c = counts[j]
        total = c.sum(-1)
        if j < raw_count:
            p = (c+.5*p)/(total[:, None]+.5)
            high = p
            candidate = p
        else:
            candidate = (c+.5*high)/(total[:, None]+.5)
        counts[j] = candidate
        totals[j] = total
    return counts, totals, high


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


def sum_rows(template, index, values):
    """Sum learner contributions in a fixed order, with one writer per row.

    CUDA float index_add_ uses unordered atomics for repeated indices. Stable
    sorting retains input order within each row; segmented reduction owns each
    output element, and the final assignment has unique row indices.
    """
    if index.numel() == 0:
        return torch.zeros_like(template)
    order = index.argsort(stable=True)
    # Integer row lengths are exact and have a fixed output shape. Empty rows
    # reduce to zero; nonempty segments retain iter39's exact sorted sequence.
    lengths = torch.zeros(len(template), dtype=torch.long, device=index.device)
    lengths.index_add_(0, index, torch.ones_like(index))
    return torch.segment_reduce(values[order], 'sum', lengths=lengths, unsafe=True)


class RowGraph:
    """Replay both unchanged deterministic row reducers with fixed shapes."""
    def __init__(self, mixer, ch, pair, values):
        self.ch, self.pair, self.values = (t.clone() for t in (ch, pair, values))
        # Only shapes/dtypes are used by sum_rows. Do not retain a reference to
        # the mixer (the weak-key cache must release this graph with its owner).
        char = mixer.opt.p['char']
        pairs = mixer.opt.p['pair']
        stream = torch.cuda.Stream(device=values.device)
        stream.wait_stream(torch.cuda.current_stream(values.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                output = (sum_rows(char, self.ch, self.values),
                          sum_rows(pairs, self.pair, self.values))
            del output
        torch.cuda.current_stream(values.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = (sum_rows(char, self.ch, self.values),
                           sum_rows(pairs, self.pair, self.values))

    def replay(self, ch, pair, values):
        self.ch.copy_(ch)
        self.pair.copy_(pair)
        self.values.copy_(values)
        self.graph.replay()
        # Outputs are consumed before the next gradient call; Adam never
        # changes them in place (its first operation is out-of-place g/norm).
        return self.output


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
        graph = ROW_GRAPHS.get(self)
        if (d.is_cuda and len(d) == 4096 and
                os.environ.get('ITER41_GRAPHS', '1') != '0'):
            if graph is None:
                graph = RowGraph(self, ch, pair, d)
                ROW_GRAPHS[self] = graph
            char_sum, pair_sum = graph.replay(ch, pair, d)
        else:
            char_sum = sum_rows(self.opt.p['char'], ch, d)
            pair_sum = sum_rows(self.opt.p['pair'], pair, d)
        return dict(char=char_sum,
                    pair=pair_sum,
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
        grads.update({f'A{j}': sum_rows(self.opt.p[f'A{j}'], idx.flatten(), df)
                      for j, idx in enumerate(ids)})
        grads.update({f'P{k}': em[:, k].flatten(0, 1).T @ df for k in range(3)})
        return grads

    def memory(self):
        return dict(**self.opt.memory(), embedding=nbytes(self.embedding),
                    projection=nbytes(self.projection), ema_matrix=nbytes(self.M),
                    bitweights=nbytes(self.bitweights))


class Model:
    def __init__(self, vocab, mode, device='cuda', tiny=False, options=None, table_slots=None):
        self.V, self.mode, self.device = len(vocab), mode, device
        self.storage_options = options or StorageOptions()
        opts = self.storage_options
        V = self.V
        sparse_sizes = {}
        def sparse(name, bits):
            slots = (table_slots or {}).get(name, 1 << bits)
            bits = slots.bit_length()-1
            assert slots == 1 << bits and slots >= opts.ways
            if opts.storage == 'tagged':
                return MeasuredTagged(bits, V, device, opts)
            sparse_sizes[name] = slots
            return None  # Allocate the flat store once, without temporary tables.
        self.word = torch.tensor([c.isascii() and (c.isalnum() or c == '_')
                                  for c in vocab]+[False], device=device)
        self.tables = {'raw1': Dense(V+1, V, device),
                       'raw2': Dense((V+1)**2, V, device)}
        for n in RAW_ORDERS[2:]:
            self.tables[f'raw{n}'] = sparse(f'raw{n}', 16 if tiny else (19 if n == 3 else 20)+CAP)
        for name in ('wordshape', 'clock1', 'clock3'):
            self.tables[name] = sparse(name, 16 if tiny else 18+CAP)
        self.net = KeysNet(V, device) if mode == 'keys' else None
        if self.net:
            for n in KEY_BITS:
                self.tables[f'key{n}'] = (Dense(1 << n, V, device) if n <= 12
                                          else sparse(f'key{n}', 16 if tiny else 18))
        self.fused = FusedBuckets(sparse_sizes, V, device, opts) if sparse_sizes else None
        if self.fused:
            self.tables.update(self.fused.views())
        self.sparse_names = tuple(sparse_sizes)
        self.dense_tables = {name: t for name, t in self.tables.items() if isinstance(t, Dense)}
        for table in self.tables.values():
            if isinstance(table, Dense):
                table.diag_every = opts.diag_every
        max_slots = max((t.count.numel() for t in self.tables.values() if isinstance(t, Tagged)), default=0)
        self.scratch = torch.zeros(max_slots, dtype=torch.int32, device=device)
        self.uni = torch.zeros(V, dtype=torch.int32, device=device)
        self.expert_names = ['unigram', *self.tables, *(['local_net'] if self.net else []),
                             *[f'copy{n}' for n in C.LENS]]
        self.mixer = LocalMixer(V, len(self.expert_names), device)

    def set_collect(self, enabled):
        if self.fused:
            self.fused.collect = enabled
        for table in self.tables.values():
            table.collect = enabled

    def diagnostics(self):
        return {name: table.diagnostics() for name, table in self.tables.items()}

    def read_tables(self, codes):
        if not self.fused:
            return {name: t.read(codes[name]) for name, t in self.tables.items()}
        packed = torch.stack(tuple(codes[name] for name in self.sparse_names))
        sparse = dict(zip(self.sparse_names, self.fused.read(packed).unbind(0)))
        dense = {name: t.read(codes[name]) for name, t in self.dense_tables.items()}
        # Metadata iteration only; sparse hashing/gathers already ran together.
        return {name: dense[name] if name in dense else sparse[name] for name in self.tables}

    def read_counts(self, codes):
        # The tables-only layout has two dense tables followed by all sparse
        # tables. Fill the final packed buffer directly rather than retaining
        # the individual reads alongside a second full torch.stack copy.
        if self.fused and self.mode == 'tables':
            n = codes['raw1'].numel()
            counts = torch.empty((len(self.tables), n, self.V), device=self.uni.device)
            packed = torch.stack(tuple(codes[name] for name in self.sparse_names))
            self.fused.read(packed, out=counts[2:])
            for row, (name, table) in enumerate(self.dense_tables.items()):
                counts[row].copy_(table.read(codes[name]))
            return counts
        return torch.stack(tuple(self.read_tables(codes).values()))

    def write_tables(self, codes, keep, target):
        if not self.fused:
            for name, table in self.tables.items():
                code = codes[name] if keep is None else codes[name][keep]
                table.write(code, target, self.scratch)
            return
        packed = torch.stack(tuple(codes[name] for name in self.sparse_names))
        if keep is not None:
            packed = packed[:, keep]
        self.fused.write(packed, target)
        for name, table in self.dense_tables.items():
            code = codes[name] if keep is None else codes[name][keep]
            table.write(code, target)

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
        packed_counts = self.read_counts(codes)
        # Keep eager's separate float operations (including multiply then add)
        # even on installations with a TorchScript pointwise fuser enabled.
        with torch.jit.optimized_execution(False):
            proposals, table_totals, high = table_proposals(packed_counts, p, len(RAW_ORDERS))
        del packed_counts
        ps.extend(proposals.unbind(0))
        totals.extend(table_totals.unbind(0))
        del proposals, table_totals
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
        del ps, heads, hp, proposal, p
        lp.clamp_min_(FLOOR).log_()
        counts = torch.stack(totals, 1)
        q, phi = self.mixer.forward(lp, counts, selectors)
        return q.reshape(*x.shape, self.V), (codes, cache, lp, phi, selectors)

    def logprobs(self, x, copy=True):
        return self.score(x, copy)[0]

    def learn_write(self, x, y, q, state, lr, valid_count=None):
        codes, cache, lp, phi, selectors = state
        # All scores, clock decisions, h and codes were materialized before ANY
        # learner/count write. The mixer cannot send a derivative into the net.
        self.mixer.opt.update(self.mixer.gradients(lp, q.flatten(0, 1), phi,
                                                  selectors, y.flatten()), .005)
        if cache:
            self.net.opt.update(self.net.gradients(cache, y), lr)
        keep = None if valid_count == y.numel() else y.flatten() >= 0
        target = y.flatten() if keep is None else y.flatten()[keep]
        self.write_tables(codes, keep, target)
        self.uni.index_add_(0, target, torch.ones_like(target, dtype=torch.int32))

    def memory(self):
        out = {name: table.memory() for name, table in self.tables.items()}
        out.update(unigram=dict(counts=nbytes(self.uni)),
                   shared_write_scratch=dict(indices=nbytes(self.scratch)),
                   character_classes=dict(word=nbytes(self.word)),
                   mixer=self.mixer.opt.memory())
        if self.net:
            out['keys_net'] = self.net.memory()
        if self.fused:
            out['shared_sparse_layout'] = dict(offsets_and_masks=nbytes(self.fused.layout))
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
        x_cuda, y_cuda = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
        # Count on the already-materialized CPU batch, including a partial tail
        # window at ANY shuffled position. No device scalar read per chunk.
        y_cuda._iter41_valid_counts = tuple(
            int((y.reshape(-1)[left:left+chunk_size] >= 0).sum())
            for left in range(0, y.size, chunk_size))
        yield x_cuda, y_cuda


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
    with torch.random.fork_rng(devices=([torch.cuda.current_device()]
                                       if torch.device(device).type == 'cuda' else [])):
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


def sparse_names(mode):
    return [*[f'raw{n}' for n in RAW_ORDERS[2:]], 'wordshape', 'clock1', 'clock3',
            *(['key16'] if mode == 'keys' else [])]


def sparse_bytes(slots, V, options):
    # Must match memory(), including diagnostics/filter/LUT, not just entries.
    size = slots*(3 if options.storage == 'compact' else 8)+V*8+7*8+(slots+63)//64
    if options.storage == 'compact':
        size += 256*4
    if options.admit == 'doorkeeper':
        size += max(64, slots*2)//8
    return size


def plan_slots(names, weights, V, options, fixed_bytes, budget_bytes):
    """Greedy power-of-two allocation proportional to observed useful entries.

    Double the most under-provisioned affordable table. Recompute the shared
    direct-map scratch cost at every step. Rounding leaves explicit budget slack.
    """
    sizes = dict.fromkeys(names, max(64, options.ways))
    def cost(candidate):
        scratch = 4*max(candidate.values()) if options.storage == 'tagged' else 0
        return fixed_bytes+sum(sparse_bytes(n, V, options) for n in candidate.values())+scratch
    if cost(sizes) > budget_bytes:
        raise ValueError(f'budget needs at least {cost(sizes)/2**20:.3f} MiB for dense/fixed/minimum storage')
    while True:
        choices = []
        for name in names:
            # Original Tagged uses iter5's 25-bit low address. Do not allocate
            # unreachable cells; associative storage uses the full avalanche.
            if options.storage == 'tagged' and sizes[name] >= 1 << 25:
                continue
            candidate = dict(sizes)
            candidate[name] *= 2
            if cost(candidate) <= budget_bytes:
                priority = max(1., weights.get(name, 0.))/sizes[name]
                choices.append((priority, name, candidate))
        if not choices:
            break
        sizes = max(choices, key=lambda c: (c[0], c[1]))[2]
    return sizes, cost(sizes)


def prepare_model(vocab, mode, train, seed, chunk_size, options):
    if options.budget_mib is None:
        return Model(vocab, mode, options=options), dict(strategy='fixed CAP sizes',
                                                       allocation_seconds=0., warmup_seconds=0.)
    # Deliberate one-shot alternative to retaining 64-bit hashes per entry.
    # Probe, discard, and restart from the SAME seed/empty state. Thus each real
    # training target still has exactly one score-before-write and one update.
    torch.cuda.synchronize()
    started = time.perf_counter()
    names = sparse_names(mode)
    budget = int(options.budget_mib*2**20)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        minimal = {name: max(64, options.ways) for name in names}
        bootstrap = Model(vocab, mode, options=options, table_slots=minimal)
        memory = bootstrap.memory()
        fixed = memory['total_persistent']-sum(sum(memory[n].values()) for n in names)
        fixed -= sum(memory['shared_write_scratch'].values())
        minimum = fixed+sum(sparse_bytes(n, len(vocab), options) for n in minimal.values())
        if options.storage == 'tagged':
            minimum += 4*max(minimal.values())
        if minimum > budget:
            raise ValueError(f'--budget-mib too small; minimum {minimum/2**20:.3f} MiB')
        # A common initial size makes usefulness measurements comparable.
        probe_budget = max(minimum, budget//8)
        probe_sizes, _ = plan_slots(names, dict.fromkeys(names, 1), len(vocab), options,
                                    fixed, probe_budget)
        del bootstrap
        gc.collect()
        torch.cuda.empty_cache()
        torch.manual_seed(seed)
        probe = Model(vocab, mode, options=options, table_slots=probe_sizes)
        probe.set_collect(True)
        warm_chars, warm_steps = 0, 0
        target = math.ceil(len(train)*.1)
        total_steps = math.ceil(len(train)/chunk_size)
        torch.cuda.synchronize()
        warm_start = time.perf_counter()
        for x, y in epoch_cpu(train, seed, chunk_size, len(vocab)):
            for left in range(0, x.numel(), chunk_size):
                masked = y.clone()
                pos = torch.arange(x.numel(), device=x.device).reshape_as(x)
                masked[(pos < left) | (pos >= left+chunk_size)] = -1
                valid = masked.flatten() >= 0
                masked.flatten()[valid.long().cumsum(0) > target-warm_chars] = -1
                valid_chars = min(y._iter41_valid_counts[left//chunk_size], target-warm_chars)
                warm_steps += 1
                if mode == 'tables':
                    # Mixer values cannot affect table contexts. Probe only
                    # their reads/writes, avoiding disposable mixer training.
                    codes = probe.contexts(x)[0]
                    codes = {name: c.flatten() for name, c in codes.items()}
                    probe.read_tables(codes)
                    keep = None if valid_chars == masked.numel() else masked.flatten() >= 0
                    target_y = masked.flatten() if keep is None else masked.flatten()[keep]
                    probe.write_tables(codes, keep, target_y)
                    probe.uni.index_add_(0, target_y, torch.ones_like(target_y, dtype=torch.int32))
                    del codes, keep, target_y
                else:
                    q, state = probe.score(x)
                    lr = .001 if warm_steps <= .75*total_steps else .0001+.0009*.5*(
                        1+math.cos(math.pi*min(1., (warm_steps-.75*total_steps)/(.25*total_steps))))
                    probe.learn_write(x, masked, q, state, lr, valid_count=valid_chars)
                    del q, state
                warm_chars += valid_chars
                del masked
                if warm_chars >= target:
                    break
            if warm_chars >= target:
                break
        torch.cuda.synchronize()
        warm_seconds = time.perf_counter()-warm_start
        measured = probe.diagnostics()
        weights = {name: measured[name]['useful_entries_estimate'] for name in names}
        sizes, planned_bytes = plan_slots(names, weights, len(vocab), options, fixed, budget)
        del probe, x, y
        gc.collect()
        torch.cuda.empty_cache()
    # fork_rng restores the state before the probe, including the model's RNG.
    model = Model(vocab, mode, options=options, table_slots=sizes)
    actual_bytes = model.memory()['total_persistent']
    assert actual_bytes == planned_bytes and actual_bytes <= budget
    torch.cuda.synchronize()
    allocation = dict(strategy='one-shot allocation after replayed first-10% storage probe',
                      warmup_chars=warm_chars, warmup_steps=warm_steps,
                      warmup_seconds=warm_seconds, allocation_seconds=time.perf_counter()-started,
                      warmup_diagnostics=measured, allocation_weights=weights,
                      probe_slots=probe_sizes, final_slots=sizes,
                      budget_bytes=budget, planned_persistent_bytes=planned_bytes,
                      unused_budget_bytes=budget-actual_bytes,
                      note='Main pass restarts empty with original RNG; probe is extra work. '
                           'No full-key hashes stored and no online rehash. Zero-use tables get weight 1.')
    return model, allocation


def associative_exactness_checks(device):
    """Use the unchanged iter38 implementation as the independent write oracle."""
    import iter38 as original

    def same(a, b):
        for name in ('count', 'tag', 'stats', 'useful'):
            assert torch.equal(getattr(a, name), getattr(b, name)), (device, name)
        assert a.writes == b.writes
        if a.door:
            assert torch.equal(a.door.bits, b.door.bits)

    generator = torch.Generator(device='cpu').manual_seed(390038)
    variants = 0
    for storage in ('compact', 'int32'):
        for ways in (1, 2, 4, 8, 16):
            for age_chunks in (0, 7):
                settings = dict(storage=storage, admit='none', ways=ways,
                                age_chunks=age_chunks, diag_every=1)
                bits = ways.bit_length()+2  # Only eight buckets: heavy evictions.
                a = original.Buckets(bits, 7, device, original.StorageOptions(**settings))
                b = Buckets(bits, 7, device, StorageOptions(**settings))
                a.collect = b.collect = True
                for step in range(32):
                    # Hits, new identities, duplicates, >W misses per bucket,
                    # stochastic log rounding and periodic age all coexist.
                    n = (0, 1, 17, 257, 1024)[step % 5]
                    code = torch.randint(0, 400, (n,), generator=generator)
                    y = torch.randint(0, 7, (n,), generator=generator)
                    if step % 4 == 0:
                        code = torch.cat((code, torch.full((100, ), step % 13)))
                        y = torch.cat((y, torch.full((100, ), step % 7)))
                    code, y = code.to(device), y.to(device)
                    torch.testing.assert_close(a.read(code), b.read(code), rtol=0, atol=0)
                    a.write(code, y)
                    b.write(code, y)
                    same(a, b)
                assert int(a.stats[2]) > 0, 'fixture must evict'
                variants += 1

    # Counterexample to distinct old victims: four one-count misses into a full
    # bucket [20,21,22,23] must all replace way zero, not four different ways.
    for storage in ('compact', 'int32'):
        settings = dict(storage=storage, admit='none', ways=4, age_chunks=0)
        a = original.Buckets(2, 4, device, original.StorageOptions(**settings))
        b = Buckets(2, 4, device, StorageOptions(**settings))
        code = torch.arange(100, device=device)
        target = code % 4
        _, tags = a.address(code, target)
        _, unique_indices = np.unique(tags.cpu().numpy(), return_index=True)
        code = code[torch.tensor(unique_indices[:4], device=device)]
        target = code % 4
        assert len(code) == 4
        missing = set(a.address(code, target)[1].cpu().tolist())
        resident_tags = [t for t in range(1000, 1100) if t not in missing][:4]
        for table in (a, b):
            table.count[:] = torch.tensor([20, 21, 22, 23], dtype=table.count.dtype, device=device)
            table.tag[:] = torch.tensor(resident_tags, dtype=table.tag.dtype, device=device)
            table.write(code, target)
        same(a, b)
        assert b.count.cpu().tolist() == [1, 21, 22, 23]
        assert int(b.stats[2]) == 4

    # All count codes, signed tags, and multiple large aging slabs. Compare the
    # actual encode/hash results, not just decoded estimates.
    settings = dict(storage='compact', admit='none', age_chunks=0)
    a = original.Buckets(20, 4, device, original.StorageOptions(**settings))
    b = Buckets(20, 4, device, StorageOptions(**settings))
    ix = torch.arange(a.count.numel(), device=device)
    for table in (a, b):
        table.count[:] = (ix % 256).to(torch.uint8)
        table.tag[:] = ((ix*P) % 65536-32768).to(torch.int16)
        table.writes = 23
        table.useful.fill_(1)
        table.age()
    same(a, b)
    return dict(device=str(device), random_stream_variants=variants,
                repeated_victim_fixture=True, large_slab_aging=True,
                oracle='iter38.Buckets.write', bit_exact=True)


def row_sum_checks(device):
    """Exercise repeated learner rows, cancellation, and empty updates."""
    import iter39 as original
    index = torch.tensor([2, 0, 2, 0, 2, 0, 5], device=device)
    for dtype, large in ((torch.float32, 2**24), (torch.float64, 2**53)):
        template = torch.zeros(6, 2, dtype=dtype, device=device)
        values = torch.tensor([[large, 1], [4, 2], [-large, 3], [-4, 4],
                               [1, 5], [.5, 6], [7, 8]], dtype=dtype, device=device)
        want = torch.tensor([[.5, 12], [0, 0], [1, 9], [0, 0], [0, 0], [7, 8]],
                            dtype=dtype, device=device)
        assert torch.equal(sum_rows(template, index, values), want)
        assert torch.equal(sum_rows(template, index, values),
                           original.sum_rows(template, index, values))
        assert torch.equal(sum_rows(template, index[:0], values[:0]), template)
        assert torch.equal(sum_rows(template, index[-1:], values[-1:]),
                           torch.cat((template[:5], values[-1:])))

    # Match the learner's large repeated-index updates, with unequal signed
    # floats and missing rows. Require exact repeats, including every feature.
    index = torch.arange(4096, device=device) % 7
    values = torch.arange(4096*96, dtype=torch.float32, device=device).reshape(4096, 96).sin()
    template = torch.zeros(9, 96, device=device)
    want = sum_rows(template, index, values)
    assert torch.equal(want, original.sum_rows(template, index, values))
    for repeat in range(8):
        assert torch.equal(sum_rows(template, index, values), want), (device, repeat)
    reference = torch.stack([values[index == row].double().sum(0) for row in range(9)])
    torch.testing.assert_close(want.double(), reference, rtol=0, atol=1e-4)
    return dict(stable_row_order=True, empty_and_single_updates=True,
                repeated_float_rows_bit_exact=True, iter39_bit_exact=True)


def fused_exactness_checks(device):
    """Compare flat stores with independent iter39 tables after EVERY chunk."""
    import iter39 as original
    generator = torch.Generator(device='cpu').manual_seed(400039)
    extremes = torch.tensor([-2**63, -1, 0, 1, 2**63-1], device=device)
    hash_input = torch.cat((extremes, torch.arange(4096, device=device)*-4265267296055464877))
    assert torch.equal(avalanche(hash_input), avalanche_inplace(hash_input.clone()))

    def same(flat, views, reference):
        for name, table in views.items():
            oracle = reference[name]
            for attr in ('count', 'tag', 'stats', 'useful'):
                assert torch.equal(getattr(table, attr), getattr(oracle, attr)), (device, name, attr)
            assert table.writes == oracle.writes == flat.writes
            if table.door:
                assert torch.equal(table.door.bits, oracle.door.bits), (device, name, 'bloom')

    variants = 0
    for storage in ('compact', 'int32'):
        for ways in (1, 2, 4, 8, 16):
            for admit in ('none', 'doorkeeper'):
                for age in (0, 7):
                    options = StorageOptions(storage=storage, ways=ways, admit=admit,
                                             age_chunks=age, door_reset_chunks=5, diag_every=3)
                    sizes = dict(a=ways*2, b=ways*8, c=ways*4)
                    flat = FusedBuckets(sizes, 7, device, options)
                    views = flat.views()
                    reference = {name: original.Buckets(slots.bit_length()-1, 7, device,
                                   original.StorageOptions(**asdict(options)))
                                 for name, slots in sizes.items()}
                    for step in range(24):
                        n = (0, 1, 17, 257, 1024, 73)[step % 6]
                        code = torch.randint(0, 400, (3, n), generator=generator)
                        target = torch.randint(0, 7, (n,), generator=generator)
                        if step % 3 == 0:
                            # Identical cross-table identities MUST NOT coalesce;
                            # repeated local writes also exercise log rounding.
                            code = torch.cat((code, torch.full((3, 100), step % 13)), 1)
                            target = torch.cat((target, torch.full((100,), step % 7)))
                        code, target = code.to(device), target.to(device)
                        flat.collect = step % 4 != 1
                        for table in reference.values():
                            table.collect = flat.collect
                        before = I.stamp(flat)
                        actual = flat.read(code)
                        if not flat.collect:
                            assert I.stamp(flat) == before, 'fused frozen read wrote state'
                        for row, table in enumerate(reference.values()):
                            assert torch.equal(actual[row], table.read(code[row]))
                            table.write(code[row], target)
                        flat.write(code, target)
                        same(flat, views, reference)
                    assert int(flat.stats[:, 2].sum()) > 0, 'fixture must evict'
                    variants += 1

    # Eight tables, explicit repeated victims in every table.
    options = StorageOptions(storage='compact', ways=4, diag_every=1)
    sizes = dict.fromkeys([f't{i}' for i in range(8)], 4)
    flat = FusedBuckets(sizes, 4, device, options)
    views = flat.views()
    reference = {name: original.Buckets(2, 4, device, original.StorageOptions(**asdict(options)))
                 for name in sizes}
    code = torch.arange(100, device=device)
    _, tags = next(iter(reference.values())).address(code, code % 4)
    _, positions = np.unique(tags.cpu().numpy(), return_index=True)
    code = code[torch.tensor(positions[:4], device=device)]
    target = code % 4
    missing = set(next(iter(reference.values())).address(code, target)[1].cpu().tolist())
    resident = [tag for tag in range(1000, 1100) if tag not in missing][:4]
    for table in (*views.values(), *reference.values()):
        table.count[:] = torch.tensor([20, 21, 22, 23], dtype=table.count.dtype, device=device)
        table.tag[:] = torch.tensor(resident, dtype=table.tag.dtype, device=device)
    flat.write(code.expand(8, -1), target)
    for table in reference.values():
        table.write(code, target)
    same(flat, views, reference)
    assert all(t.count.cpu().tolist() == [1, 21, 22, 23] for t in views.values())

    # Preserve max-per-four-way decoding and ordered group sums, even with
    # unusual preexisting duplicate resident tags across different way groups.
    duplicate_options = StorageOptions(storage='compact', ways=16, diag_every=1)
    duplicate = FusedBuckets(dict(a=16, b=16), 4, device, duplicate_options)
    duplicate_views = duplicate.views()
    duplicate_reference = {name: original.Buckets(4, 4, device,
                            original.StorageOptions(**asdict(duplicate_options)))
                           for name in duplicate_views}
    code = torch.tensor([2], device=device)
    target = torch.tensor([3], device=device)
    for name, view in duplicate_views.items():
        oracle = duplicate_reference[name]
        _, tag = oracle.address(code, target)
        for table in (view, oracle):
            table.tag.fill_(int(tag[0]))
            table.count[:] = (torch.arange(16, device=device)*13).to(torch.uint8)
            table.collect = True
    duplicate.collect = True
    actual = duplicate.read(code.expand(2, -1))
    for row, table in enumerate(duplicate_reference.values()):
        assert torch.equal(actual[row], table.read(code))
    same(duplicate, duplicate_views, duplicate_reference)

    # All 256 codes, signed fingerprints and a slab crossing a table boundary.
    sizes = dict(small=1024, large=1 << 20)
    flat = FusedBuckets(sizes, 4, device, options)
    views = flat.views()
    reference = {name: original.Buckets(slots.bit_length()-1, 4, device,
                                      original.StorageOptions(**asdict(options)))
                 for name, slots in sizes.items()}
    flat.writes = 23
    for name, view in views.items():
        oracle = reference[name]
        oracle.writes = 23
        ix = torch.arange(len(view.count), device=device)
        for table in (view, oracle):
            table.count[:] = (ix % 256).to(torch.uint8)
            table.tag[:] = ((ix*P) % 65536-32768).to(torch.int16)
            table.useful.fill_(1)
        oracle.age()
    flat.age()
    same(flat, views, reference)
    return dict(device=str(device), random_stream_variants=variants,
                oracle='iter39.Buckets', bit_exact=True, eight_tables_repeated_victims=True,
                table_local_aging_all_codes=True, duplicate_resident_tags=True, frozen_reads=True)


def fused_model_checks(vocab, device):
    """Exact scores, storage and nested optimizers against the original Model."""
    import iter39 as original

    def allocated_bytes(model):
        seen, storages = set(), {}
        def visit(obj):
            if id(obj) in seen:
                return
            seen.add(id(obj))
            if isinstance(obj, torch.Tensor):
                storage = obj.untyped_storage()
                storages[storage.data_ptr()] = storage.nbytes()
            elif isinstance(obj, dict):
                for value in obj.values():
                    visit(value)
            elif isinstance(obj, (tuple, list)):
                for value in obj:
                    visit(value)
            elif hasattr(obj, '__dict__'):
                visit(vars(obj))
        visit(model)
        return sum(storages.values())

    variants = 0
    devices = [torch.cuda.current_device()] if torch.device(device).type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        for mode in ('tables', 'keys'):
            for storage in ('compact', 'int32'):
                for admit in ('none', 'doorkeeper'):
                    options = StorageOptions(storage=storage, admit=admit, age_chunks=2,
                                             door_reset_chunks=3, diag_every=1)
                    torch.manual_seed(40)
                    a = original.Model(vocab, mode, device, tiny=True,
                                       options=original.StorageOptions(**asdict(options)))
                    torch.manual_seed(40)
                    b = Model(vocab, mode, device, tiny=True, options=options)
                    assert b.memory()['total_persistent'] == allocated_bytes(b)
                    a.set_collect(True)
                    b.set_collect(True)
                    x = (torch.arange(24, device=device) % 4).reshape(2, 12)
                    y = (x+1) % len(vocab)
                    y[:, -2:] = -1
                    for step in range(4):
                        qa, sa = a.score(x)
                        qb, sb = b.score(x)
                        assert torch.equal(qa, qb), (mode, storage, admit, step, 'score')
                        assert torch.equal(sa[2], sb[2]), (mode, step, 'experts')
                        a.learn_write(x, y, qa, sa, .001)
                        b.learn_write(x, y, qb, sb, .001)
                        for name, table in a.tables.items():
                            other = b.tables[name]
                            for attr in ('count', 'stats'):
                                assert torch.equal(getattr(table, attr), getattr(other, attr)), (name, attr)
                            if isinstance(table, original.Buckets):
                                for attr in ('tag', 'useful'):
                                    assert torch.equal(getattr(table, attr), getattr(other, attr)), (name, attr)
                                if table.door:
                                    assert torch.equal(table.door.bits, other.door.bits)
                        for module in ('mixer', 'net') if a.net else ('mixer',):
                            for kind in ('p', 'm', 'v'):
                                aa = getattr(getattr(a, module).opt, kind)
                                bb = getattr(getattr(b, module).opt, kind)
                                assert all(torch.equal(value, bb[name]) for name, value in aa.items()), (module, kind)
                        assert torch.equal(a.uni, b.uni)
                    variants += 1
    return dict(iter39_scores_and_nested_optimizers_bit_exact=True, variants=variants)


def storage_checks(device='cpu'):
    """Small storage checks, runnable without loading data or requiring CUDA."""
    codec = LogCount(device)
    levels = codec.levels.cpu().numpy()
    assert np.all(np.diff(levels) > 0)
    truth = torch.arange(1, 10001, dtype=torch.float32, device=device)
    encoded = codec.encode(truth, torch.full_like(truth, .5))
    estimates = codec.decode(encoded)
    quantization_error = float(((estimates-truth).abs()/truth).max())
    assert quantization_error <= .10
    assert bool((estimates[1:] >= estimates[:-1]).all())
    assert int(codec.encode(torch.tensor([1e9], device=device),
                            torch.tensor([0.], device=device))[0]) == 255
    # Exercise the actual tensor update for 1e4 sequential increments. Independent
    # virtual key identities give a tight ensemble-mean check at EVERY prefix.
    identities = torch.arange(512, dtype=torch.long, device=device)*P
    counts = torch.zeros(512, dtype=torch.uint8, device=device)
    mean_error = torch.zeros((), device=device)
    for n in range(1, 10001):
        counts = codec.encode(codec.decode(counts)+1, uniform_hash(identities, n))
        mean_error = torch.maximum(mean_error, (codec.decode(counts).mean()/n-1).abs())
    assert float(mean_error) <= .10, float(mean_error)

    exercised = 0
    for storage in ('tagged', 'int32', 'compact'):
        for admit in ('none', 'doorkeeper'):
            options = StorageOptions(storage=storage, admit=admit, ways=4,
                                     age_chunks=0, door_reset_chunks=10000, diag_every=1)
            cls = MeasuredTagged if storage == 'tagged' else Buckets
            table = cls(12, 4, device, options)
            assert sum(table.memory().values()) == sparse_bytes(4096, 4, options)
            scratch = torch.zeros(4096, dtype=torch.int32, device=device)
            # Pick distinct buckets/cells; Bloom fixtures also avoid false positives.
            candidates = torch.arange(1, 200, device=device)
            buckets, _ = table.address(candidates, candidates % 4)
            selected, used, used_bits = [], set(), set()
            for code, bucket in zip(candidates.cpu().tolist(), buckets.cpu().tolist()):
                h = int(avalanche(torch.tensor([code*P+code % 4+1], device=device))[0])
                mask = (table.door.nbits-1) if table.door else 0
                bloom_bits = {h & mask, ((h >> 32) ^ h*P) & mask}
                if bucket not in used and (not table.door or not (bloom_bits & used_bits)):
                    selected.append(code)
                    used.add(bucket)
                    used_bits |= bloom_bits
                if len(selected) == 6:
                    break
            assert len(selected) == 6
            stream = selected+selected[::-1]+selected*2
            reference, seen = Counter(), set()
            for code in stream:
                c = torch.tensor([code], device=device)
                target = torch.tensor([code % 4], device=device)
                want = torch.tensor([[reference[code, y] for y in range(4)]],
                                    dtype=torch.float32, device=device)
                torch.testing.assert_close(table.read(c), want, rtol=0, atol=0)
                if admit == 'none' or code in seen:
                    reference[code, code % 4] += 1
                seen.add(code)
                table.write(c, target, scratch)
            assert table.diagnostics()['eviction_count'] == 0
            exercised += 1
            # Batch duplicates must admit the second occurrence within a chunk.
            duplicate = cls(12, 4, device, options)
            c = torch.tensor([selected[0]]*3, device=device)
            duplicate.write(c, c % 4, scratch)
            expected = 2 if admit == 'doorkeeper' else 3
            assert float(duplicate.read(c[:1])[0, selected[0] % 4]) == expected

    # Distinct bit updates in the same byte must all survive a batch write.
    bloom = Doorkeeper(32, device)
    h = avalanche(torch.arange(32, dtype=torch.long, device=device))
    bloom.admit(h)
    packed = [0]*len(bloom.bits)
    for value in h.cpu().tolist():
        for bit in (value & 63, ((value >> 32) ^ value*P) & 63):
            packed[bit >> 3] |= 1 << (bit & 7)
    assert bloom.bits.cpu().tolist() == packed
    assert bool(bloom.admit(h).all())
    # Resetting admission must not drop updates to an already resident pair.
    options = StorageOptions(storage='compact', admit='doorkeeper', door_reset_chunks=2,
                             age_chunks=0)
    table = Buckets(6, 4, device, options)
    c = torch.tensor([1], device=device)
    for _ in range(3):
        table.write(c, c)
    assert float(table.read(c)[0, 1]) == 2

    # Force bucket collisions, frequency eviction, aging, and nontrivial log
    # counts. Repeat actual writes with no RNG reset and require byte equality.
    deterministic = []
    for repeat in range(2):
        options = StorageOptions(storage='compact', admit='doorkeeper', ways=4,
                                 age_chunks=7, door_reset_chunks=11, diag_every=1)
        table = Buckets(5, 4, device, options)
        table.collect = True
        c = torch.zeros(1000, dtype=torch.long, device=device)
        table.write(c, c)
        for step in range(20):
            c = (torch.arange(80, device=device)*17+step % 3) % 43
            table.read(c)
            table.write(c, c % 4)
        deterministic.append((table.count.clone(), table.tag.clone(), table.door.bits.clone(),
                              table.stats.clone(), table.useful.clone(), table.read(c)))
    assert all(torch.equal(a, b) for a, b in zip(*deterministic))
    # A single miss must replace the unique least-frequency way.
    options = StorageOptions(storage='compact', ways=4, age_chunks=0)
    table = Buckets(2, 4, device, options)
    codes, fingerprints = [], set()
    for code in range(100):
        _, fingerprint = table.address(torch.tensor([code], device=device),
                                       torch.tensor([code % 4], device=device))
        fingerprint = int(fingerprint[0])
        if fingerprint not in fingerprints:
            codes.append(code)
            fingerprints.add(fingerprint)
        if len(codes) == 5:
            break
    assert len(codes) == 5
    for amount, code in enumerate(codes[:4], 1):
        c = torch.tensor([code]*amount, device=device)
        table.write(c, c % 4)
    before = table.count.clone()
    before_tag = table.tag.clone()
    victim = int(before.int().argmin())
    c = torch.tensor([codes[-1]], device=device)
    table.write(c, c % 4)
    assert int(table.count[victim]) == 1
    assert int(table.tag[victim]) != int(before_tag[victim])
    assert table.diagnostics()['eviction_count'] == 1
    assert torch.equal(table.count[torch.arange(4, device=device) != victim],
                       before[torch.arange(4, device=device) != victim])
    table.count[:] = torch.tensor([2, 4, 8, 16], dtype=torch.uint8, device=device)
    table.age()
    assert table.count.cpu().tolist() == [1, 2, 4, 8]
    # Reuse the baseline's collision election to verify bit-for-bit equivalence.
    old = Tagged(4, 4, device)
    new = MeasuredTagged(4, 4, device, StorageOptions())
    scratch = torch.zeros(16, dtype=torch.int32, device=device)
    for step in range(8):
        c = torch.arange(300, device=device)+step
        old.write(c, c % 4, scratch)
        new.write(c, c % 4, scratch)
        assert torch.equal(old.count, new.count) and torch.equal(old.tag, new.tag)
    # Budget plan must include scratch/filter/diagnostic costs and favor useful tables.
    options = StorageOptions(storage='compact', admit='doorkeeper')
    sizes, cost = plan_slots(['a', 'b'], dict(a=64, b=1), 4, options, 100, 65536)
    assert cost <= 65536 and sizes['a'] > sizes['b']
    return dict(compact_decode_monotonic=True, compact_quantization_max_relative_error=quantization_error,
                compact_increment_ensemble_max_relative_error=float(mean_error),
                compact_max_decoded_count=float(levels[-1]),
                compact_error_note='10% checks cover deterministic quantization and ensemble mean; '
                                   'individual Morris paths have stochastic error',
                dictionary_stream_variants=exercised, same_chunk_doorkeeper=True,
                deterministic_storage_twice=True, frequency_eviction=True, count_aging=True,
                packed_bloom_concurrent_bits=True, resident_updates_survive_bloom_reset=True,
                tagged_matches_iter37=True, budget_includes_all_persistent_storage=True,
                associative_matches_iter38=associative_exactness_checks(device),
                learner_row_sums=row_sum_checks(device),
                fused_matches_iter39=fused_exactness_checks(device))


def result_tag(mode, options, chunk_size, seed):
    budget = 'off' if options.budget_mib is None else f'{options.budget_mib:g}'.replace('.', 'p')
    ways = 1 if options.storage == 'tagged' else options.ways
    return (f'{mode}_{options.storage}_{options.admit}_budget{budget}_ways{ways}'
            f'_age{options.age_chunks}_door{options.door_reset_chunks}'
            f'_diag{options.diag_every}_stale{options.stale_every}_cap{CAP}'
            f'_chunk{chunk_size}_seed{seed}')


def validate_options(options):
    if options.storage not in ('tagged', 'int32', 'compact'):
        raise ValueError('unknown --storage')
    if options.admit not in ('none', 'doorkeeper'):
        raise ValueError('unknown --admit')
    if options.ways not in (1, 2, 4, 8, 16):
        raise ValueError('--ways must be 1, 2, 4, 8, or 16')
    if options.age_chunks < 0:
        raise ValueError('--age-chunks must be nonnegative (0 disables aging)')
    if min(options.door_reset_chunks, options.diag_every, options.stale_every) < 1:
        raise ValueError('reset and sampling intervals must be positive')
    if options.budget_mib is not None and (not math.isfinite(options.budget_mib) or options.budget_mib <= 0):
        raise ValueError('--budget-mib must be finite and positive')


def configured_checks(vocab, mode, device, options):
    outputs = []
    learned_states = []
    with torch.random.fork_rng(devices=([torch.cuda.current_device()]
                                       if torch.device(device).type == 'cuda' else [])):
        for repeat in range(2):
            torch.manual_seed(38)
            model = Model(vocab, mode, device, tiny=True, options=options)
            x = (torch.arange(24, device=device) % 4).reshape(2, 12)
            y = (x+1) % model.V
            for step in range(4):
                q, state = model.score(x)
                model.learn_write(x, y, q, state, .001)
                # Optimizer tensors are nested, not direct mixer attributes.
                optimizers = dict(mixer=model.mixer.opt)
                if model.net:
                    optimizers['keys'] = model.net.opt
                snapshot = {f'{module}.{kind}.{name}': tensor.clone()
                            for module, opt in optimizers.items()
                            for kind in ('p', 'm', 'v')
                            for name, tensor in getattr(opt, kind).items()}
                if repeat == 0:
                    learned_states.append(snapshot)
                else:
                    for name, tensor in snapshot.items():
                        assert torch.equal(tensor, learned_states[step][name]), (step, name)
            model.set_collect(False)
            before = I.stamp(model)
            a = model.score(x)
            other = x.clone()
            other[:, 7:] = (other[:, 7:]+3) % model.V
            b = model.score(other)
            torch.testing.assert_close(a[0][:, :7], b[0][:, :7], rtol=0, atol=1e-6)
            # Inspect every expert as well as the mixer output.
            torch.testing.assert_close(a[1][2].reshape(2, 12, -1, model.V)[:, :7],
                                       b[1][2].reshape(2, 12, -1, model.V)[:, :7],
                                       rtol=0, atol=1e-6)
            assert I.stamp(model) == before, 'configured evaluation wrote state'
            Q.plain(model)
            outputs.append(a[0].clone())
            del model, q, state, a, b
    assert torch.equal(*outputs), 'configured outputs differ between identical runs'
    return dict(configured_storage_causal=True, configured_eval_frozen=True,
                configured_outputs_deterministic_twice=True,
                configured_optimizer_deterministic_twice=True)


def bounded_execution_checks(vocab, device):
    """Exercise block edges, graph replay, and full-shape learner equivalence."""
    import iter40 as original
    devices = [torch.cuda.current_device()] if torch.device(device).type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(41)
        for stride in (1, 3, 16, 1000):
            opts = StorageOptions(storage='compact', diag_every=stride)
            sizes = {f't{i}': 64 << (i % 3) for i in range(8)}
            a = original.FusedBuckets(sizes, 7, device, opts)
            b = FusedBuckets(sizes, 7, device, opts)
            a.views(), b.views()
            code = (torch.arange(8*4096, device=device).reshape(8, 4096)*P) % 317
            y = torch.arange(4096, device=device) % 7
            a.write(code, y)
            b.write(code, y)
            for n in (0, 1, 511, 512, 513, 1023, 1024, 4096):
                a.collect = b.collect = True
                want, actual = a.read(code[:, :n]), b.read(code[:, :n])
                assert torch.equal(want, actual), (stride, n, 'read')
                assert torch.equal(a.stats, b.stats), (stride, n, 'stats')
                assert torch.equal(a.useful, b.useful), (stride, n, 'useful')
                b.collect = False
                stamp = I.stamp(b)
                saved = actual.clone()
                b.read(code[:, :n]+1)
                assert torch.equal(actual, saved), 'read result aliases replay workspace'
                assert stamp == I.stamp(b), 'frozen block read mutated predictor'
            del a, b
        for mode in ('tables', 'keys'):
            opts = StorageOptions(storage='compact', diag_every=3)
            torch.manual_seed(41)
            a = original.Model(vocab, mode, device, tiny=True, options=opts)
            torch.manual_seed(41)
            b = Model(vocab, mode, device, tiny=True, options=opts)
            a.set_collect(True)
            b.set_collect(True)
            x = (torch.arange(4096, device=device).reshape(32, 128)*7) % len(vocab)
            for step in range(3):
                y = (x+step+1) % len(vocab)
                if step == 1:
                    y[5, -11:] = -1  # Tail window in the middle of a shuffled batch.
                qa, sa = a.score(x)
                qb, sb = b.score(x)
                assert torch.equal(qa, qb), (mode, step, 'scores')
                assert torch.equal(sa[2], sb[2]), (mode, step, 'experts')
                a.learn_write(x, y, qa, sa, .001)
                b.learn_write(x, y, qb, sb, .001, valid_count=4096-(11 if step == 1 else 0))
                for name in a.tables:
                    for attr in ('count', 'tag', 'stats', 'useful'):
                        if hasattr(a.tables[name], attr):
                            assert torch.equal(getattr(a.tables[name], attr),
                                               getattr(b.tables[name], attr)), (mode, name, attr)
                for module in ('mixer', 'net') if a.net else ('mixer',):
                    for kind in ('p', 'm', 'v'):
                        aa = getattr(getattr(a, module).opt, kind)
                        bb = getattr(getattr(b, module).opt, kind)
                        assert all(torch.equal(v, bb[k]) for k, v in aa.items()), (mode, step, module, kind)
                assert torch.equal(a.uni, b.uni)
                del qa, qb, sa, sb
            del a, b
        if torch.device(device).type == 'cuda':
            # All chunk sizes, including sub-window chunks, count the partial
            # shuffled tail on CPU exactly as the actual device mask does.
            train = np.arange(128*35+37) % len(vocab)
            for chunk in (1, 37, 128, 512, 4096):
                total = 0
                for x, y in epoch_cpu(train, 41, chunk, len(vocab)):
                    for left in range(0, y.numel(), chunk):
                        actual = y._iter41_valid_counts[left//chunk]
                        assert actual == int((y.flatten()[left:left+chunk] >= 0).sum())
                        total += actual
                assert total == len(train), (chunk, total)
    gc.collect()
    assert not READ_GRAPHS and not ROW_GRAPHS, 'execution graph cache retained discarded models'
    return dict(iter40_full_shape_bit_exact=True, block_edges=True,
                global_diagnostic_sampling=True, nonaliased_read_outputs=True,
                full_and_masked_writes=True, frozen_reads=True,
                host_target_counts=(True if torch.device(device).type == 'cuda' else 'cuda_only'),
                cuda_graph_replay=(torch.device(device).type == 'cuda' and
                                   os.environ.get('ITER41_GRAPHS', '1') != '0'),
                weak_graph_lifetime=True)


def self_tests(device):
    result = storage_checks(device)
    vocab = json.loads((I.ROOT/'data/meta.json').read_text())['vocab']
    result['inherited_model_checks'] = {mode: checks(vocab, mode, device)
                                        for mode in ('tables', 'keys')}
    result['fused_model_equivalence'] = fused_model_checks(vocab, device)
    result['bounded_execution'] = bounded_execution_checks(vocab, device)
    result['configured_checks'] = {
        f'{mode}_{storage}': configured_checks(vocab, mode, device,
                                              StorageOptions(storage=storage))
        for mode in ('tables', 'keys') for storage in ('tagged', 'compact', 'int32')}
    return result


def run(mode, smoke=None, chunk_size=4096, seed=0, options=None):
    options = options or StorageOptions()
    validate_options(options)
    if smoke is not None and smoke < 1:
        raise ValueError('--smoke must be positive')
    if chunk_size < 1 or (chunk_size >= 128 and chunk_size % 128):
        raise ValueError('--chunk-size must be 1..127 or a positive multiple of 128')
    if chunk_size > 4096:
        raise ValueError('chunk sizes above 4096 are outside the 151 MiB working-set design')
    if seed < 0:
        raise ValueError('--seed must be nonnegative')
    tag = result_tag(mode, options, chunk_size, seed)
    path = I.ROOT/f'results/iter41_{tag}{"_smoke" if smoke is not None else ""}.json'
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
    verified['storage'] = storage_checks('cpu')
    verified.update(configured_checks(vocab, mode, 'cuda', options))
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, allocation = prepare_model(vocab, mode, train, seed, chunk_size, options)
    allocation_peak = torch.cuda.max_memory_allocated()/2**20
    torch.cuda.reset_peak_memory_stats()
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
        train_peak, seconds, chars, steps, curve = allocation_peak, 0., 0, 0, []
        stale_seconds, stale_chars, stale_chunks = 0., 0, 0
        stale = {k: torch.zeros((), dtype=torch.long, device='cuda')
                 for k in [*model.tables, 'any_context', 'including_unigram']}
        milestones = [math.ceil(len(train)*k/10) for k in range(1, 11)]
        next_milestone = 0
        total_steps = math.ceil(len(train)/chunk_size)
        model.set_collect(True)
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
                if (steps-1) % options.stale_every == 0:
                    # Semantic unique() is sampled and explicitly OUTSIDE the
                    # baseline training timer. Synchronize both boundaries so
                    # asynchronous CUDA diagnostic work cannot leak into it.
                    torch.cuda.synchronize()
                    seconds += time.perf_counter()-start
                    diagnostic_start = time.perf_counter()
                    stale_contexts(state[0], masked_y, stale)
                    stale_chars += int((masked_y >= 0).sum())
                    stale_chunks += 1
                    torch.cuda.synchronize()
                    stale_seconds += time.perf_counter()-diagnostic_start
                    start = time.perf_counter()
                lr = .001 if steps <= .75*total_steps else .0001+.0009*.5*(
                    1+math.cos(math.pi*min(1., (steps-.75*total_steps)/(.25*total_steps))))
                valid_chars = y._iter41_valid_counts[left//chunk_size]
                model.learn_write(x, masked_y, q, state, lr, valid_count=valid_chars)
                chars += valid_chars
                del q, state, masked_y
                if next_milestone < 10 and chars >= milestones[next_milestone]:
                    torch.cuda.synchronize()
                    seconds += time.perf_counter()-start
                    train_peak = max(train_peak, torch.cuda.max_memory_allocated()/2**20)
                    torch.cuda.reset_peak_memory_stats()
                    eval_start = time.perf_counter()
                    model.set_collect(False)
                    before = I.stamp(model)
                    value = CONTROL.evaluate_subset(model.logprobs, subset)
                    assert I.stamp(model) == before, 'curve evaluation wrote state'
                    model.set_collect(True)
                    torch.cuda.synchronize()
                    eval_seconds += time.perf_counter()-eval_start
                    eval_peak = max(eval_peak, torch.cuda.max_memory_allocated()/2**20)
                    while next_milestone < 10 and chars >= milestones[next_milestone]:
                        curve.append(dict(fraction=(next_milestone+1)/10, chars=chars,
                                          train_seconds=seconds, val_subset=value,
                                          train_seconds_with_stale_diagnostics=seconds+stale_seconds,
                                          train_seconds_with_allocation=seconds+allocation['allocation_seconds']))
                        next_milestone += 1
                    print(json.dumps(dict(progress=chars/len(train), val_subset=value,
                                          train_seconds=seconds)), flush=True)
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
        torch.cuda.synchronize()
        seconds += time.perf_counter()-start
        train_peak = max(train_peak, torch.cuda.max_memory_allocated()/2**20)
        model.set_collect(False)
        assert chars == len(train) and len(curve) == 10
        assert memory == model.memory(), 'persistent memory grew'
        Q.plain(model)
        assert not torch.is_grad_enabled()
        diagnostic_start = time.perf_counter()
        diagnostics = model.diagnostics()
        torch.cuda.synchronize()
        final_diagnostics_seconds = time.perf_counter()-diagnostic_start
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
        observed = dict(test=test, renamed=renamed_loss, val=val,
                        train_seconds=seconds+allocation['allocation_seconds'],
                        peak_allocated_mib=train_peak,
                        flops_per_char_estimate=estimate['iter35'])
        limits = dict(TARGETS, test=1.185, renamed=1.550, train_seconds=67.,
                      flops_per_char_estimate=estimate['iter30_backprop'])
        # Validation is diagnostic: requested all-target conjunction covers five
        # resource/quality bars plus compute, not an additional val threshold.
        required = [k for k in limits if k != 'val']
        bars = {k: dict(value=observed[k], threshold=v, comparison='<=',
                        pass_observed=observed[k] <= v,
                        pass_full_run=(observed[k] <= v) if smoke is None else None)
                for k, v in limits.items()}
        reference = dict(val=1.2769815154075623, test=1.1712950946092606,
                         renamed=1.5450188283920288)
        reference_applies = (smoke is None and mode == 'tables' and seed == 0 and
                             chunk_size == 4096 and options == StorageOptions(
                                 storage='compact', budget_mib=64, ways=4, admit='none', age_chunks=0))
        if reference_applies:
            exact = all(observed[k] == v for k, v in reference.items())
            bars['iter40_bit_exact'] = dict(value={k: observed[k] for k in reference},
                                           threshold=reference, comparison='==',
                                           pass_observed=exact, pass_full_run=exact)
            required.append('iter40_bit_exact')
        result = dict(mode=mode, tag=tag, smoke=smoke, comparable_full_run=smoke is None,
                      val=val, test=test, renamed=renamed_loss, test_nocopy=nocopy,
                      val_loss=val, test_loss=test, renamed_loss=renamed_loss,
                      test_loss_nocopy=nocopy, renamed_gap=renamed_loss-test,
                      train_seconds=seconds, eval_seconds=eval_seconds,
                      train_seconds_with_stale_diagnostics=seconds+stale_seconds,
                      train_seconds_with_allocation=seconds+allocation['allocation_seconds'],
                      train_seconds_with_all_diagnostics_and_allocation=(seconds+stale_seconds+
                          final_diagnostics_seconds+allocation['allocation_seconds']),
                      stale_diagnostics_seconds=stale_seconds,
                      final_diagnostics_seconds=final_diagnostics_seconds,
                      allocation=allocation, storage_options=asdict(options),
                      iter40_bit_exact_reference=dict(applicable=reference_applies,
                                                      expected=reference),
                      diagnostics=diagnostics,
                      persistent_bytes_per_table={name: sum(table.memory().values())
                                                  for name, table in model.tables.items()},
                      references=dict(bp_test=1.2562, bp_renamed=1.5714, bp_peak_mib=151,
                                      tables16x_test=1.1768, tables16x_renamed=1.5441),
                      peak_allocated_mib=dict(train=train_peak, eval=eval_peak),
                      peak_allocated_mib_train=train_peak, peak_allocated_mib_eval=eval_peak,
                      memory_bytes=memory, flops_per_char_estimate=estimate,
                      stale_sampled_chars=stale_chars, stale_sampled_chunks=stale_chunks,
                      stale_fraction=float(stale['including_unigram'])/max(1, stale_chars),
                      stale_fraction_context=float(stale['any_context'])/max(1, stale_chars),
                      stale_fraction_by_structure={k: float(v)/max(1, stale_chars) for k, v in stale.items()},
                      stale_definition='Estimated from every stale_every-th chunk, with sampled-char denominator. '
                                       'Fraction of scored positions with an earlier valid write in '
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
                                    read_subchunk_targets=READ_BLOCK,
                                    cuda_graphs=os.environ.get('ITER41_GRAPHS', '1') != '0',
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
                                    collision=('31-bit tag; max tag wins a chunk collision; replace then add'
                                               if options.storage == 'tagged' else
                                               'set associative; coalesce duplicates; deterministic W miss rounds; '
                                               'minimum count victim, ties lowest way; excess misses dropped'),
                                    count_dtype=('uint8 exact to 32, then base-1.06 stochastic log; int16 fingerprint'
                                                 if options.storage == 'compact' else 'int32 count and tag'),
                                    storage=asdict(options),
                                    copy_lengths=C.LENS, probability_floor=FLOOR,
                                    nocopy='replace copy experts with highest lexical expert; frozen mixer',
                                    surprise='old exact bigram probability of observed input; thresholds 1/3; '
                                             'two iter10 three-event hashes',
                                    word_context='normalized 8-char shape + current run length (cap 15) + delimiter',
                                    metric_note='novel: iter15/iter30 aligned changed targets, maximal ASCII '
                                                'runs in a window; repeat requires an earlier complete run; '
                                                'README attributes earlier first/repeat observations to iter12',
                                    train_seconds_scope='CPU batch assembly/transfers, scoring, local updates, '
                                                        'memory writes and sampled storage diagnostics; excludes '
                                                        'eval/setup/checks, separately timed stale unique diagnostics '
                                                        'and separately reported allocation probe. Time target includes allocation.',
                                    memory_scope='CUDA max allocated, separately sampled for train and eval; '
                                                 'persistent bytes separate from transient working set'))
        path.parent.mkdir(exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, allow_nan=False)
        return result
    finally:
        B.N_EVAL_WINDOWS = old_windows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('tables', 'keys'), default='tables')
    parser.add_argument('--smoke', type=int, nargs='?', const=16384, metavar='N',
                        help='prefix N targets (default 16384); 16 eval windows; labeled JSON')
    parser.add_argument('--chunk-size', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--storage', choices=('tagged', 'int32', 'compact'), default='tagged',
                        help='original direct map; int32 associative; uint8/int16 associative')
    parser.add_argument('--admit', choices=('none', 'doorkeeper'), default='none',
                        help='admission for sparse tables only; raw1/raw2 remain dense')
    parser.add_argument('--budget-mib', type=float,
                        help='total persistent MiB, including dense/mixer/scratch/filter/diagnostics')
    parser.add_argument('--ways', type=int, default=4, help='associativity; tagged always uses 1 way')
    parser.add_argument('--age-chunks', type=int, default=0,
                        help='halve associative counts every N writes; default 0 disables; tagged ignores')
    parser.add_argument('--door-reset-chunks', type=int, default=256)
    parser.add_argument('--diag-every', type=int, default=16, help='sample every Nth context read')
    parser.add_argument('--stale-every', type=int, default=64, help='sample semantic staleness outside train timer')
    parser.add_argument('--self-test', action='store_true', help='storage/model/configured checks; no training')
    parser.add_argument('--check-device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    options = StorageOptions(**{name: getattr(args, name) for name in StorageOptions.__dataclass_fields__})
    validate_options(options)
    result = (self_tests(args.check_device) if args.self_test else
              run(args.mode, args.smoke, args.chunk_size, args.seed, options))
    print(json.dumps(result, indent=2), flush=True)
