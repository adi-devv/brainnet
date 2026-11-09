**Pick: add window-local suffix-match attention to `local`, with a learned copy gate.** Keep the existing network; add four heads matching suffixes of lengths 1, 2, 4, and 8 against earlier positions. Each head predicts the character following matching occurrences, using only characters already observed. Mix these distributions with the network prediction through a linear softmax gate trained analytically from mixture cross-entropy on detached features.

This provides content-addressed copying without persistent count tables. Its plausible advantage is combining `local`’s cheap training with immediate reuse of unseen identifiers—the strongest missing capability in your results. Whether copying removes enough *overall* loss is the hypothesis.

**Smallest test:** one seed, one 6.2M-character pass, unchanged network settings, 128-character windows, memory reset each window. Unmatched heads are masked. No gradient crosses into the network from the gate.

- **Pass:** test ≤ **1.30**, renamed ≤ **1.64**, training ≤ **180 seconds**, peak memory ≤ **321 MB**. This beats the supplied backprop reference under both data and time budgets.
- Mechanism check: repeat-name loss ≤ **4.0**; disabling copy at evaluation should erase much of the improvement.
- Missing the overall-loss/time bars means failure against the goal, even if copying improves.

**Why not the closest alternatives?** More passes or depth retain fixed context mixing; your declining returns and worsening name losses discourage that. Fully learned local attention adds score-learning and derivative costs before establishing whether retrieval itself closes the gap.

**What I distrust:** baseline samples overlapping random windows from all training data; local uses partitioned windows from 95%. That confounds learning-rule comparisons. First/repeat losses compare different token populations, and smaller renamed gaps need not mean better robustness. Also, `iter15` uses exact within-layer chain-rule gradients: it avoids *cross-layer* backprop.

I don’t expect unchanged fixed-EMA `local` scaling to reach your goal.