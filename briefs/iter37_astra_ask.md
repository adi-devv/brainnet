# Astra: review the iter35 one-shot memory results and say what's next (read-only advice)

Repo: this folder. Read `README.md`, `iter35.py`, `briefs/iter35_memory_brief.md`, `briefs/iter35_codex_out.md` (if present), `results/iter35_tables.json`, `results/iter35_keys.json`, `results/iter30_backprop.json`, `briefs/iter35_astra_out.md` (your last advice), `briefs/iter36_roles_brief.md`.

## Goal (unchanged)
A system that learns WITHOUT backprop and is better than backprop on loss, training time and compute at LARGE scale. Do not reframe to "our arch trained with BP".

## Results (seed 0, same data, splits and evals as iter30, one pass)
| | val | test | renamed | train s | peak MiB | FLOPs/char |
|---|---|---|---|---|---|---|
| BP (iter30) | 1.3869 | 1.2562 | 1.5714 | 79.4 | 151 | 7,188,480 |
| iter35 tables | 1.3405 | 1.2225 | 1.5878 | 66.9 | 103 | ~29k (check json) |
| iter35 keys (+96-unit local KeysNet) | 1.3422 | 1.2279 | 1.5939 | 85.8 | 132 | 137,316 |

Tables: novel_first_loss 6.01, novel_repeat_loss 2.61. Keys: val 1.337 already at 10% of data (8.6 s). Stale fraction 0.9998 (chunked score-then-write).
Tables beats BP on everything except renamed code (+0.016). The learned keys did not help. iter36 (role-keyed contexts: identifiers replaced by causal role ids) is being built for the renamed gap.

## Questions (concise, <500 words; label subjective numbers)
1. Is this a real win, or an artifact of small data (6.2M chars) where count/n-gram models are known to be competitive? Historically, transformers overtake n-gram/count models as data grows. What does the literature (infini-gram, kNN-LM, Kneser-Ney vs neural scaling, memory layers / product keys, Engram-style hashed n-gram memory) say about how this gap moves with data and model size?
2. Wall time: FLOPs are ~250× below BP, yet train time is only 1.2× faster. Where does the time likely go (hash, gather/scatter, Python overhead, evals), and what would make wall-clock track compute? What is the realistic speedup ceiling?
3. What is the cheapest decisive experiment to test whether this beats BP at scale? E.g. a data-scaling curve (10/30/100% and a larger dataset) vs matched BP at each point, plus a bigger BP. What size and dataset, and what kill criterion?
4. How should memory size scale with data (bounded tables saturate; collisions)? Is there a design where generalization comes from something other than literal context matching, without backprop (e.g. role keys, learned-but-local similarity keys, clustering), that could keep up with BP at scale?
5. Top 3 next experiments, ranked, with predicted outcome and kill criterion each.

Do not change files or run training.
