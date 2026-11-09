Follow-up on your design A. Consult only: read anything, but do not edit, create or run anything. Answer in under 600 words.

We built A as you specified and ran it. It failed your loss bars but is very fast. Please check the work, then tell us what to do.

Files to read (all in D:/Aadit/claudez/brainnet):
- `iter20.py`: design A, all three modes. `briefs/iter20_assoc.md` is the build spec.
- `iter18.py`: `local` net plus suffix-match copy heads. `briefs/iter18_copy.md` is its spec.
- `iter14.py`, `iter15.py`: the base networks.
- `results/iter20_assoc.json`, `iter20_noretrieve.json`, `iter20_learned_input.json`, `iter18_copy.json`, `iter18_backprop_same.json`.

All runs: 6.2M chars, one pass, seed 0, GTX 1650. Test and renamed loss in nats/char.

| Run | Time | Test | Renamed | Novel first / repeat | Peak MB |
|---|---|---|---|---|---|
| Backprop GPT | 180 s | 1.329 | 1.667 | 5.64 / 5.21 | 321 |
| Backprop, same one-pass sampler as local | 181 s | 1.341 | 1.686 | 5.71 / 5.38 | 319 |
| 1. `local` + copy heads (iter 18) | 74.5 s | 1.411 | 1.697 | 5.69 / 1.66 | 103 |
| `local` alone (for reference) | ~70 s | 1.492 | 1.819 | 5.77 / 5.45 | 105 |
| 2. iter 20 `assoc` | 39 s | 1.645 | 1.920 | 5.56 / 3.34 | 74 |
| 3. iter 20 `noretrieve` | 32 s | 1.573 | 1.921 | 5.96 / 5.58 | 72 |
| 4. iter 20 `learned_input` | 53 s | 1.654 | 1.935 | 5.70 / 3.83 | 77 |

Per-block test loss: `assoc` 1.727, 1.669, 1.645. `noretrieve` 1.657, 1.594, 1.573. `learned_input` 1.706, 1.662, 1.654.
Retrieval match rate on test: L2 48%, L4 33%, L8 20%.
Iter 18 mean gate weights: net .858, L1 .019, L2 .023, L4 .027, L8 .073.

Choices we made that your design did not specify: `s[0]` keeps iter 14's six lag lookups plus three EMAs; fixed inputs are multiplied by `INPUT_SCALE` 16; the block input is the unweighted average of the current state and the matched states; blocks have biases. Treat these as suspects too.

Questions:
1. Check `iter20.py` for bugs or a mismatch with your design. Say which, with line numbers, or say you found none.
2. Why is `noretrieve` worse than `local` (1.573 vs 1.492)? Why does retrieval make test loss worse (1.645 vs 1.573) while repeats improve? Rank the causes, each with the cheapest experiment that would confirm or reject it.
3. Iter 20 has room: 39 s against a 90 s budget. What specific changes spend that budget best? Give exact widths, block counts, learning rates, and how retrieved states should be combined if averaging is wrong. Or say plainly that this design cannot reach the bars and why.
4. Iter 18: what would close its remaining 0.07 test loss (1.411 vs same-sampler backprop 1.341) inside 90 s? It has about 15 s spare. Is it better to add copy heads to a repaired iter 20, or to keep improving iter 18?
5. Pick the one next experiment. Give exact configuration, pass and kill numbers on the same 6.2M benchmark (at most 90 s, test at most 1.329, renamed at most 1.667), and your odds with one reason.

Be direct. If something we did was a mistake, say so.
