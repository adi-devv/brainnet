**A is defensible as an experiment; neither target is something I can promise.** I would first test whether content retrieval closes your local model’s quality gap.

1. **A: residual prediction with fixed associative attention.** Keep three width-256 blocks. Each retrieves previous-layer states following the most recent matching character suffixes of lengths 2, 4, and 8; retrieve only positions already observed. Average retrieved states into its input \(x\). Compute:
   \[
   h=\tanh(Wx),\quad z=z_{\rm previous}+Rh,\quad x_{\rm next}=x+Uh
   \]
   Here \(h\) has width 128; \(U\) is a fixed signed scatter; initial embeddings are fixed.

   Each block computes \(e=\mathrm{softmax}(z)-y\), then updates:
   \[
   R\leftarrow R-\eta eh^\top,\qquad
   W\leftarrow W-\eta[(R^\top e)\odot(1-h^2)]x^\top.
   \]
   Use old weights throughout each minibatch; update after emitting its activations. Stop derivatives into previous logits, input states, and retrieval. **This crosses two learned maps within each block: it is local backprop, not gradient-free learning.**

   Retrieval restores content-dependent access missing from EMAs/lags. Accumulated logits make later blocks correct earlier predictions instead of solving independent prediction tasks. Neither guarantees recovering the 0.163 loss gap.

   For width \(d\), bottleneck \(r\), alphabet \(C\), dense work is approximately \(2dr+3Cr\) multiply-accumulates/block/character, plus retrieval. A conventional expansion-4 GPT block needs approximately \(36d^2\), plus attention, during training. At \(r=d/2,C\le d\), this is \(2.5d^2\). Your GPT’s actual dimensions and lookup overhead determine the real ratio; reduced capacity contributes substantially.

2. **B: causal lexical binding and copying.** Initially use file-local spelling identities, **not semantic bindings**. A streaming lexer creates an ID when an identifier finishes. Store spelling, occurrence count, and last position; expire at file end, with bounded-memory eviction.

   At each possible identifier start, mix a character generator with a distribution over stored spellings. Update mixture weights and copy-rank counts using posterior responsibilities after observing characters. Filter candidates as the prefix grows; include identifier termination. First spellings receive generator probability. Score the resulting mixture on original characters.

   Key structural counts by token classes and locally renumbered identifier IDs. Thus alpha-renamed contexts share counts. Cap packed structural tables at 128 MB and strings/indexes at 32 MB: combined with your 105 MB local model, approximately 265 MB before additional buffers; enforce the remaining budget.

   My speculative expectation: **5–10% loss reduction**, assuming reusable identifiers contribute 20% of loss and copying removes 25–50% of that contribution. Measure both assumptions. This does not establish 3–5× sample efficiency.

3. **Combination:** A supplies structural predictions. B supplies spelling probabilities. Train A against mixture responsibilities.

4. **Build A first.** At 6.2M characters, pass requires ≤90 seconds, test ≤1.329, renamed ≤1.667. Kill this configuration if any threshold fails; replicate passes across three seeds. B passes only at those losses using ≤2.07M characters; ≤1.24M demonstrates 5×.

5. **Subjective odds:** A **15%**—fixed retrieval may underfit; B **10%**—naming explains limited loss; combined superiority **10%**—lookup overhead threatens speed.