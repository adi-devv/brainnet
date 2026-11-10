Created [iter39.py](iter39.py) and [briefs/iter39_codex_out.md](briefs/iter39_codex_out.md). Existing files were untouched.

Writes are batched, admission and aging default off, and a scripted correction preserves sequential re-eviction behavior.

Syntax/source checks and 83,420 scalar ordering cases passed. Torch self-tests and smoke were blocked by Python access restrictions, so speed and peak memory remain unmeasured. No training ran.