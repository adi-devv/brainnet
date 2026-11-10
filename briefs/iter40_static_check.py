"""Dependency-free source and fused-order checks; does not import PyTorch."""
import ast
import bisect
import collections
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
old_tree = ast.parse((ROOT/'iter39.py').read_text())
candidate = ROOT/(sys.argv[1] if len(sys.argv) > 1 else 'iter40.py')
new_tree = ast.parse(candidate.read_text())
compile(new_tree, str(candidate), 'exec')
old = {n.name: n for n in old_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
new = {n.name: n for n in new_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
counts = collections.Counter(n.name for n in new_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)))
assert max(counts.values()) == 1
same = lambda a, b: ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
changed = [name for name in old if not same(old[name], new[name])]
assert set(changed) <= {'sum_rows', 'Model', 'checks', 'prepare_model', 'row_sum_checks',
                        'storage_checks', 'configured_checks', 'run'}
for name in ('Dense', 'Tagged', 'Buckets', 'LogCount', 'avalanche', 'uniform_hash',
             'StorageOptions', 'Adam', 'LocalMixer', 'KeysNet', 'epoch_cpu',
             'stale_contexts', 'flops', 'plan_slots', 'finite_difference_checks'):
    assert same(old[name], new[name]), name
print('AST/compile passed; inherited definitions changed:', ', '.join(changed))

# Execute only the dependency-free budget functions from the actual sources.
budget_funcs = ast.Module(body=[new[name] for name in ('sparse_bytes', 'plan_slots')], type_ignores=[])
ns = {}
exec(compile(budget_funcs, '<budget functions>', 'exec'), ns)
result = json.loads((ROOT/'results/iter39_tables_compact_none_budget64_ways4_age0_door256_diag16_stale64_cap0_chunk4096_seed0.json').read_text())
alloc = result['allocation']
names = list(alloc['final_slots'])
opts = SimpleNamespace(**result['storage_options'])
fixed = result['memory_bytes']['total_persistent']-sum(sum(result['memory_bytes'][name].values()) for name in names)
extra = 7*len(names)*8
for weights, budget, wanted in ((dict.fromkeys(names, 1), alloc['budget_bytes']//8, alloc['probe_slots']),
                                (alloc['allocation_weights'], alloc['budget_bytes'], alloc['final_slots'])):
    sizes, cost = ns['plan_slots'](names, weights, 96, opts, fixed+extra, budget)
    assert sizes == wanted, (sizes, wanted)
print('Supplied full-run probe/final allocation unchanged; layout bytes:', extra)

# Model coalesced accepted records as (table, local bucket, unsigned tag, adds).
# Reference is sequential victim election per table. The transformed path sorts
# globally, restores local identity, applies the exact safe/correction test and
# elects the last rank for every victim. Integer codes suffice for that logic.
rng = random.Random(400042)
cases = records = repeats = 0
for ways in (1, 2, 4, 8, 16):
    for tagbits in (16, 31):
        for trial in range(400):
            nbuckets = [rng.choice((1, 2, 4, 8)) for _ in range(8)]
            offsets = [0]
            for size in nbuckets:
                offsets.append(offsets[-1]+size)
            count = [[rng.randrange(0, 256) for _ in range(ways)] for _ in range(offsets[-1])]
            tags = [[rng.randrange(1, 23) for _ in range(ways)] for _ in count]
            stream = [(rng.randrange(8), rng.randrange(100), rng.randrange(1, 32)) for _ in range(257)]
            stream += stream[:51]
            coalesced = collections.Counter(((offsets[t]+b % nbuckets[t]) << tagbits) | tag
                                             for t, b, tag in stream)
            keys = sorted(coalesced, reverse=True)
            hit_count = [row[:] for row in count]
            ref_tags = [row[:] for row in tags]
            misses = collections.defaultdict(list)
            for key in keys:
                bucket, tag = key >> tagbits, key & ((1 << tagbits)-1)
                owner = bisect.bisect_right(offsets[1:-1], bucket)
                local = key-(offsets[owner] << tagbits)
                assert local == ((bucket-offsets[owner]) << tagbits) | tag
                hit = [w for w in range(ways) if tags[bucket][w] == tag and count[bucket][w] > 0]
                if hit:
                    w = max(hit)
                    hit_count[bucket][w] = min(255, count[bucket][w]+coalesced[key])
                else:
                    misses[bucket].append((tag, min(255, coalesced[key])))
            reference = [row[:] for row in hit_count]
            actual = [row[:] for row in hit_count]
            actual_tags = [row[:] for row in tags]
            for bucket, incoming in misses.items():
                incoming = incoming[:ways]
                victims = []
                for tag, value in incoming:
                    w = min(range(ways), key=lambda w: (reference[bucket][w], w))
                    victims.append(w)
                    reference[bucket][w], ref_tags[bucket][w] = value, tag
                old_counts = hit_count[bucket]
                ordered = sorted(range(ways), key=lambda w: (old_counts[w], w))
                tail = ordered[len(incoming)-1]
                unsafe = any((value, ordered[rank]) < (old_counts[tail], tail)
                             for rank, (_, value) in enumerate(incoming[:-1]))
                if unsafe:
                    scratch = old_counts[:]
                    chosen = []
                    for _, value in incoming:
                        w = min(range(ways), key=lambda w: (scratch[w], w))
                        chosen.append(w)
                        scratch[w] = value
                else:
                    chosen = ordered[:len(incoming)]
                assert chosen == victims
                last = {w: rank for rank, w in enumerate(chosen)}
                for w, rank in last.items():
                    tag, value = incoming[rank]
                    actual[bucket][w], actual_tags[bucket][w] = value, tag
                repeats += len(chosen) != len(set(chosen))
                records += len(incoming)
            assert actual == reference and actual_tags == ref_tags
            cases += 1
print('Fused scalar cases:', cases, 'accepted misses:', records, 'repeated-victim buckets:', repeats)
