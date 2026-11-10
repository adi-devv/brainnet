# Build roles.py: role-keyed contexts (structure separate from names) for the one-shot memory LM

Repo: this folder. Read `README.md`, `iter30.py` (data, splits, the `I.rename` renamed-test construction, evals), `iter12.py`/`iter10.py` (identifier-aware contexts, pointer heads), and `briefs/iter35_memory_brief.md`. ANOTHER AGENT IS WRITING `iter35.py` RIGHT NOW: do not create, read-depend on, or edit it. Do NOT edit existing files. Do NOT run full training; only smoke/tests below.

## Why
The brain's hippocampal system stores relational structure ("a name defined here and used later") separately from content (the literal name), so it generalizes to new names instantly. Our one-shot count tables beat backprop on test, but they lose on renamed code (1.606 vs BP 1.5714) because their keys contain literal names. Role-keyed contexts fix this: the same code with different names should hit the same memory slots.

## Deliverable: `roles.py` (a library plugged into iter35 later, plus a standalone evaluator)
All of it is causal: the role features at position t use chars <= t only. No backprop.

1. **Causal identifier tracker over the char stream, per document/window as iter30 defines them.** It tracks the current partial word and the completed identifiers. Keywords, builtins and frequent global words stay literal. "Frequent" is decided causally from train counts so far (threshold as an arg). All other identifiers get a role id. Implement both of these and expose both:
   - (a) first-occurrence index in the document;
   - (b) recency rank (move-to-front).
2. **Role-abstracted context keys.** For orders k in a configurable set, a hash of the last k tokens (literal chars/keywords/punct, with each non-literal identifier replaced by its role id) plus the current partial-word state. Return int64 hashes per position per order. They should be vectorized or chunked GPU/torch so the full 6.2M stream takes about 10 s or less.
3. **Decoding roles back to chars.** Memory keyed by role contexts predicts the next *symbol*, which may be a role id. Provide a function that turns a distribution over roles (plus literal next-char counts) into a next-char distribution at position t. It does this by looking up the current document's identifier for that role and the char that follows the current partial prefix. If the role's identifier is inconsistent with the prefix, renormalize; if the role is a new name, fall back to the literal char distribution.
4. **Standalone evaluator `--eval`.** A minimal bounded count-table LM using only role keys plus literal keys, with a simple online local mixer (hybrid_mixer style). It is scored before each write, on the iter30 test and renamed splits. This measures the generalization effect on its own. Report test, renamed, renamed gap, time and memory, alongside a `--literal-only` run of the same evaluator as the control.

## Checks
- Causality on a tiny case: changing chars after t leaves the features at t unchanged.
- Rename invariance: on a test doc and its `I.rename` version, the role-key hashes are identical at every position outside identifier characters.
- An exact dict-based reference matches the counts on a tiny stream.

## Output
`results/iter36_roles_{role|literal}.json` with test, renamed, gap, the first-occurrence vs repeat loss split, train_seconds, peak memory, and `references=dict(bp_test=1.2562, bp_renamed=1.5714, tables_test=1.165, tables_renamed=1.606)`.

## Verify
`/c/Users/aadit/AppData/Local/Programs/Python/Python312/python.exe -X utf8 -B roles.py --eval --smoke 204800`, then the same with `--literal-only`. If your sandbox can't run Python, write it completely and skip. Report in `briefs/iter36_codex_out.md`: summary, API for iter35 integration, smoke output, and any deviation.
