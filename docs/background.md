# Background

Source: ChatGPT conversation "Decompile Small LLMs Feasibility" (shared 2026-11-14).

## Core idea
Incrementally move computation in a small transformer from weights into explicit code.
Two framings were discussed:

1. **Decompile / replace** (post-hoc): frozen reference model, replaceable computation graph,
   agent proposes code replacements for heads / MLPs / multi-layer regions. Objective:
   minimize `L(program) + L(constants/tables) + L(remaining weights)` (bits, fixed encoding)
   subject to a fidelity budget vs. the original (e.g. KL on next-token dist), per-behavior
   limits, runtime/memory limits. Allow `f(x) ≈ P(x) + r_φ(x)` (program + shrinking neural
   correction). Counterexample-guided loop; always evaluate the whole accumulated hybrid
   against the original; private held-out eval.

2. **Inject during training** (doubleAI, "Can LLMs redesign themselves as they train?", Nov 6 2026):
   teacher agent inserts code modules into the forward pass:
   `h̃_j = h_j + g · W_out · P(W_in · h_i)` with scalar gate `g` init 0, learned `W_in/W_out`,
   read layer i, write layer j. Teacher used "phantom gradients" (grads wrt hidden activations)
   + failure analysis to pick modules and insertion points. Training then continues normally.
   Reported: CLRS-style algorithmic acc 55%→84%; lookup recall 72% (inserted mid-training) vs 55%
   (present from start).

## User's hypothesis (capacity reuse)
Code subsumes circuits → neural capacity is freed → fixed-size model learns *new* things better.
Three escalating claims to test:
1. Code adds a capability.
2. Code takes over existing work (old neural path becomes dispensable).
3. Network reuses freed capacity (learns new skill set B better while retaining A).

## Proposed experiment
- Train identical small models on skill set A (copy, count, lookup, sort, brackets...).
- Variants: (a) continued training; (b) code insertion + adaptation; (c) insertion + adaptation +
  substitution pressure (prune/suppress redundant neural path, replay A); (d) matched extra neural
  params (control).
- Then train all on skill set B that the code can't directly solve; require retention of A.
- Bottleneck diagnostics per primitive: oracle args/timing vs learned routing vs substitution vs
  downstream learning.
- Baselines: pruning, quantization, distillation, I/O-only program synthesis, external tool use.

## Relevant recent work
- Decompiling Transformers to RASP (2026); Learning Transformer Programs (2023);
  Neural decompiling of Tracr (2024); GPT-2 IOI circuit (2022).
- "Explaining Attention with Program Synthesis" (rev. Oct 2026): Python reproducing attention
  patterns in GPT-2/TinyLlama/Llama-3B; 25% of heads replaced, ~16% ppl increase.
- Cammarata "Neural Networks Are Already Clean" (Oct 15 2026); Leo Gao response (replacement
  test as the bar); "Many Transcoder Features Are Neurons in Disguise"; Goodfire Nov 10 report
  (Qwen3.6-35B-A3B, 92% routed activation removal → +2.9% loss).
- Open questions: differentiability of arbitrary code modules; resource accounting
  (fixed neural budget incl. projections/gates; track code size, runtime, teacher cost).
