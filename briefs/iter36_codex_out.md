# Iter36 role contexts

Added `roles.py`. No existing file was edited. The implementation does not
import, inspect, or depend on `iter35.py`. No training ran.

## Implementation

- Per-window identifier bookkeeping exposes zero-based first-occurrence ids
  and move-to-front recency ranks. Identifiers are committed only when their
  delimiter arrives; the unfinished word is never flushed at the window end.
  Hard Python keywords and builtins use stable literal-word tokens. Other
  words become literal when their prior training count reaches the threshold.
  `frequency_threshold <= 0` disables frequency promotion.
- Completed identifiers each occupy one context token. Punctuation,
  whitespace, numbers and quoted/comment text emit literal characters. The
  partial-word state is its length, which avoids retaining its spelling.
  Recency tokens record the rank immediately before moving that identifier
  to the front. Historical tokens keep the rank originally emitted.
- CPU regex/span bookkeeping constructs token events and per-position event
  indices. Torch gathers and int64 rolling hashes operate on batches/chunks,
  on the requested CPU or CUDA device. GPU arithmetic intentionally wraps
  int64. Full-stream timing has not been measured.
- A bounded role decoder maps each available role to the next character of
  its known identifier consistent with the consumed prefix. Inconsistent and
  exhausted identifiers are excluded and the remaining mass renormalizes.
  Unbound roles and an optional NEW symbol use literal probabilities.
  Ambiguous prefix matches choose the lowest matching role for the observed
  next character when writing. This uses no future complete target word.
- The evaluator combines tagged, direct-mapped literal and role count rows
  with a local logistic mixer using the observed character and count
  confidence. It computes the score before updating the mixer and tables.
  Different windows within a timestep are parallel examples. Colliding writes
  deterministically select the last input tag; matching writers add counts.
  No backpropagation or autograd is used.

## API for iter35 integration

```python
from roles import role_features, encode_targets, decode_roles

# x: [B,T] character ids, with optional pad == len(vocab).
# Keep train_counts frozen while scoring a parallel batch.
features = role_features(
    x, vocab, orders=(1, 2, 4, 8),
    schemes=("first", "recency"),
    train_counts=train_counts, frequency_threshold=100,
    update_counts=False, max_roles=32, build_decode=True,
    device=x.device,
)
first_keys = features.keys["first"]       # order -> int64 [B,T]
recency_keys = features.keys["recency"]   # order -> int64 [B,T]
next_chars = features.role_next["first"]  # int16 [B,T,32]

# At t, read memory and obtain literal_counts [...,V] and role_mass
# [...,32] (or [...,33], where the last column is NEW).
probabilities = decode_roles(role_mass, literal_counts, next_chars[:, t], fallback)
# Score y[:,t] NOW, before writing.
symbols = encode_targets(y[:, t], next_chars[:, t], len(vocab))
# Symbol ids: 0..V-1 literal chars, V..V+31 roles, V+32 optional NEW.
# Ignore -1 padded targets. Write symbols to the selected role-key table.

# After the entire parallel batch has been scored and written:
train_counts.update(features.observed_counts)
```

`role_next` values are vocabulary indices for valid proposals, -1 for a
known inconsistent role, and -2 for an unbound role. Set `build_decode=False`
for hash extraction alone; this avoids the [B,T,R] proposal allocation.
`partial_length` is [B,T]. `trackers` contains final window states, not
per-position features. `IdentifierTracker.scan(text)` also exposes causal
per-character emitted tokens, partial prefixes and current role bindings.
Use a fresh tracker for every benchmark window. `complete(word)` is the
delimiter-triggered commit operation for callers with an existing lexer.

Other public building blocks are `literal_keys`, `BoundedCounts` (read/write),
and `RoleMemoryLM` (`train_batch`, frozen `logprobs`, `memory_bytes`).
The module has no import-time data loading, CUDA setup or grad-mode changes.

## Evaluator and output

The evaluator imports `iter30` only inside `run_eval`, reusing its actual
`B.load`, `B.eval_windows` and `I.rename`. Its CPU training-window generator
duplicates `iter30.R.epoch`: seed-zero permutation of non-overlapping
128-target windows, with the preceding input character or initial pad,
batch size 16, and masked final padding. The training partition is the first
95% of train; smoke restricts that partition to the requested prefix.
Evaluation is frozen and uses the same deterministic fixed windows, reduced
to 16 for smoke as in iter30. First/repeat losses use iter30's inherited
`Q.novel` definition: changed renamed target characters only, with word
occurrence determined within each window. There is no validation tuning.

Successful runs create, without overwriting:

- `results/iter36_roles_role.json`
- `results/iter36_roles_literal.json`

Each records test, renamed, gap, novelty losses/counts, train/eval seconds,
training chars/loss, CUDA peak allocation, process peak working set/RSS,
bytes per persistent structure, check outcomes, settings and the requested
reference numbers. `--scheme recency` selects the other role variant;
`--literal-only` uses the same evaluator with the role experts disabled.
Existing result files cause an error to respect the no-existing-file-edits
constraint. Smoke results are explicitly marked with their training budget.

For V=96, four orders, 4096 slots and 32 decoded roles, fixed literal tables
use 6,422,528 bytes and role tables 8,585,216 bytes. The mixer adds 13,968
bytes and the unigram 384 bytes: about 14.33 MiB total before CPU metadata,
temporary features and Torch allocator overhead. Literal-only fixed arrays
use about 6.13 MiB. Counts halve at 2**23 to retain unit-update precision.
The exact causal training-word Counter is separate CPU metadata and grows
with distinct training words; its measured size is reported separately.
Window identity state resets and is bounded by the window length. Optional
decoder proposals need B*T*R*2 bytes per scheme. Hash extraction should be
called in window batches rather than retaining every stream-wide tensor.

## Verification / smoke output

Both requested commands were attempted with the Windows equivalent of the
specified executable path:

```text
C:/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B roles.py --eval --smoke 204800
Program 'python.exe' failed to run: Access is denied
CategoryInfo: ResourceUnavailable
FullyQualifiedErrorId: NativeCommandFailed
exit code: 1

C:/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B roles.py --eval --smoke 204800 --literal-only
Program 'python.exe' failed to run: Access is denied
CategoryInfo: ResourceUnavailable
FullyQualifiedErrorId: NativeCommandFailed
exit code: 1
```

Python did not start. Per the brief's sandbox exception, execution was
skipped. No runtime checks, syntax compilation, losses, timings or memory
measurements are claimed, and no fabricated result JSON files were created.

`--eval` runs built-in tiny checks before training: future-suffix causality
for hashes and decoder proposals, invariance against actual `I.rename`, an
independent scalar hash recurrence, chunk-one counts versus a Python dict
with role-symbol targets, decoder rejection/new-role behavior, causal
frequency promotion, production prediction causality, frozen evaluation,
deterministic colliding writes, and prefix/EOF cases with dangling escapes
and triple quotes. `--checks` runs these without loading experiment data;
it uses a fixed manual rename when `I.rename` is not supplied.

## Limits / deviations

- The approximately 10-second feature target and all evaluation results
  remain unverified because Python execution is denied.
- Frequency-promoted words intentionally remain literal. Since `I.rename`
  also changes such names, unconditional invariance and frequency promotion
  cannot both hold. The strict invariance check disables promotion, checks
  both role schemes, and compares positions outside identifier characters.
- The lexer treats complete quoted regions as opaque, including f-strings;
  it does not parse embedded f-string expressions. Python 3.12 `I.rename`
  can rename identifiers in those expressions, so invariance is not
  guaranteed for those cases. Independently reset fragments and collisions
  in `I.rename`'s random name mapping can also limit whole-corpus invariance.
- Partial-word hashing keeps length rather than literal prefix or a large
  candidate set. The decoder still checks the actual causal prefix.
- The context count arrays are fixed size; exact global word-frequency
  metadata is not. This is reported explicitly rather than treating the
  CPU Counter as part of the fixed table budget.
