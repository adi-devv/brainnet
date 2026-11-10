# Build iter35.py: fast bounded one-shot memory LM that must beat backprop on loss, time, memory and compute

Repo: this folder. Read `README.md` (Findings), `iter10.py`, `iter12.py`, `hybrid_mixer.py`, `iter8.py` (and the files they import: the count tables), `iter27.py`, `iter30.py` (data, splits, evals, BP control). Do NOT edit existing files. Do NOT run full training (GPU shared); only the smoke command below.

## Why
Gradient-descent-style local learning can at best tie backprop. The only thing in this repo that has beaten backprop is the backprop-free one-shot count tables (iter12: test 1.165, against iter30 BP 1.2562). They lost on cost (244 s, about 1.9 GB) and on renamed text (1.606, against BP 1.5714). iter35 keeps one-shot memory as the core and fixes both weaknesses.

## Targets (all must hold vs iter30 `backprop` seed 0: test 1.2562, renamed 1.5714, val 1.3869, train 79.4 s, peak 151 MiB)
test < 1.2562, renamed < 1.5714, train_seconds < 79.4, peak allocated < 151 MiB, and estimated training FLOPs per char < BP's. Same data, splits, one pass, and the same eval functions as iter30, so numbers are comparable.

## Spec
No backprop anywhere. Grad mode stays off, except that a layer-local exact gradient inside one small module is allowed (as in iter27 local). Every prediction is scored before its memory write (causal).

`--mode tables`
- Bounded hashed count memory on the GPU, for several context orders (char n-grams of a few lengths, plus the word/identifier-aware and surprise-clocked contexts from iter10/iter12 if they're cheap). It is fixed-size: pick the slot layout (dense per-slot vocab counts, or count-min over (context,next) pairs) and the dtype to fit the memory budget, and explain the choice.
- Process the stream in parallel chunks: score a chunk against the memory state from before it, then scatter-add its writes. Choose the chunk size to trade speed against missed within-chunk repeats. Report what fraction of positions see a stale memory.
- Keep cheap copy/pointer heads (iter27 suffix copy or iter12 pointer heads) so recent repeats are caught inside a chunk.
- Mixer: local logistic mixing / gating over the order predictions plus confidence features (counts, order), learned online with a local rule (hybrid_mixer style).

`--mode keys` (tables plus learned keys, aimed at renamed generalization)
- A small local net (iter27 layer-0 sized or smaller, trained on its own local CE as in iter27) produces a hidden state h. Its quantized code (sign bits of a projection of h, a few bit-widths) keys extra memory tables. Similar contexts, e.g. the same code with a different identifier name, then share slots, so a write to one generalizes to the other.
- These tables feed the same mixer. The net learns only from its own local loss.

## Checks
- Causality: prediction at position t is unchanged if chars after t change (tiny case).
- With chunk size 1, memory counts equal an exact dict-based reference on a tiny stream.
- Memory stays bounded: report bytes per structure.

## Output
`results/iter35_{mode}.json` with: val, test, renamed, test_nocopy, train_seconds (excluding evals), eval_seconds, peak_allocated_mib (train vs eval), memory_bytes per structure, flops_per_char_estimate for iter35 and for the iter30 BP model (count matmul FLOPs, 3x forward for BP), stale_fraction, the val curve every 10%, and the targets with pass/fail. Also report losses on first-occurrence vs repeat positions, matching iter12's metric.

## Verify
Smoke both modes: `/c/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B iter35.py --mode MODE --smoke 204800`. If your sandbox can't run Python, write the file completely and skip. Report in `briefs/iter35_codex_out.md`: summary, design choices with expected memory and time, and any deviation.
