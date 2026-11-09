My bet: **a hierarchical predictive-memory transformer**, learning discrete representations by merging memories that predict the same futures. The full goal seems possible, but the brain does not establish that removing backprop causes its advantage.

1. **Brain property:** two complementary memory systems. Immediate storage separates acquiring an example from discovering reusable structure, reducing interference. That is the computational argument behind [complementary learning systems](https://web.stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf). I regard locality primarily as a biological constraint; sparsity helps efficiency but does not explain abstraction.

2. **Concrete architecture:** stack blocks using hard associative attention. Each block queries a bounded memory using suffixes of incoming symbols; retrieved entries emit discrete state IDs for the next block. Entries store context keys, successor-state counts, and character distributions at horizons 1, 8, and 64. Predict before inserting each observation.

   Learning increments observed counts. Periodic replay proposes merging states with similar future distributions or splitting states by an additional context symbol. Accept changes only when they improve local prediction loss plus a storage penalty, evaluated on separate replay examples. Rebuild affected counts after restructuring. The output uses smoothed next-character counts from the deepest matched state. These tables and routing assignments are all the learned parameters.

3. **Speed and quality:** eliminate backward activation storage, reverse computation, and repeated distributed weight changes to remember individual examples. Shared predictive states pool evidence across superficially different contexts. [Predictive-state theory](https://arxiv.org/abs/1408.2025) provides the mechanism: genuinely equivalent histories can merge without losing predictive information. **Finite-horizon estimated equivalence offers no such guarantee**—that is the central gamble.

4. **Scale:** bound retrieval and restructuring candidates per token; shard memories; compose state IDs across layers. Capacity can grow without activating everything. However, useful abstractions must remain discoverable within those budgets. State explosion could defeat scalability.

5. **No disguised backprop:** no derivatives, Jacobians, backward errors, or gradients through any chain. Counts, discrete proposals, and loss comparisons perform every update.

6. **Smallest kill test:** two layers, 6M characters, three seeds, ten-minute caps on your hardware. Compare against GPT and the same memory architecture without merging. Match context and total model-plus-memory bytes; include replay/indexing time. Measure held-out bits/character, data needed for a fixed loss, and transfer to unseen code-template combinations. Stop this implementation if compression fails to improve transfer/sample efficiency, or if it cannot beat GPT’s loss within equal time. This rejects the implementation, not every possible scaling variant.

7. **Closest alternatives:** pure episodic retrieval lacks learned abstraction. Predictive coding retains expensive iterative inference and can [approximate backprop itself](https://arxiv.org/abs/2206.02629).

8. **Odds:** subjectively, 5% for all requested advantages at scale. Likeliest failure: merging erases rare, long-range distinctions. Closest reachable outcome: faster, sample-efficient learning in structured domains, without broad superiority.