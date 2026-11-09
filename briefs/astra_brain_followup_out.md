**I’d change the pick to predictive memory with causal identifier binding and copying.** Your results favor improving what gets shared across examples; they do not establish that removing backward computation is the main opportunity.

1. **A: roughly 2–3× less arithmetic for otherwise identical dense computation**, assuming backward costs approximately one to two forward passes. That is an accounting estimate, not a wall-clock guarantee: forward-time updates still cost work and can disrupt batching. Your 70-second model is 2.57× faster, but removing attention confounds attribution. Passing this range requires reducing forward computation too—conditional execution, fewer active parameters, or cached computation—not merely relocating updates.

2. **B: no defensible numerical ceiling follows from these measurements.** Two GPT checkpoints cannot establish the learning curve’s shape. A reasonable *experimental target*, not forecast, is 3–5× fewer characters. The mechanism I would test is **sharing structural evidence across identifier spellings**: represent repeated identifiers using binding IDs, and retrieve their original spellings through a copy distribution. Renamed programs then reinforce the same structural statistics. First occurrences still require spelling predictions; this cannot remove arbitrary naming entropy.

3. **My original memory proposal targets B, not A.** Your tables already demonstrate the distinction: better predictions per character, worse time per character. Generic predictive-state merging is too indirect. Explicit binding supplies the invariance that merging would otherwise need to discover.

4. **Build binding-aware copying first**, inside the existing table model. Derive bindings exclusively from the observed prefix. Score probabilities of the original characters, including first spellings and copy-versus-generation uncertainty; canonicalized-text loss would be misleading.

   A convincing **5× speedup** is:
   - Test ≤ **1.329**, renamed ≤ **1.667**, within **36 seconds**; or
   - Test ≤ **1.124**, renamed ≤ **1.530**, within **93 seconds**.

   Include preprocessing, indexing, and updates. At current table throughput, those budgets permit roughly **0.91M** or **2.36M** characters before added overhead. Use three seeds, fixed evaluation splits, and a binding-disabled control. Report characters needed alongside time.

5. **Updated subjective odds:** **2%** for broad superiority at scale; **20%** for a 5× equal-loss win here. The likely failure is that naming invariance explains too little total loss.