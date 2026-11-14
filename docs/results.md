# Results log

## M1 — gated code module on running sum mod 5 (2026-11-14)
Task: `BOS x1..xn`, predict `(x1+..+xi) mod 5` at each position. Tiny NoPE GPT (2 layers, d=64, ~100k params).
Train lengths 2–16; eval id 2–16, ood 17–32, ood2 33–64. 3 seeds each. Seq accuracy at end.

### M1a: random interfaces, gate init 0 (`scripts/milestone1.sh`, 4000 steps)
| variant | id | ood | ood2 | gate |
|---|---|---|---|---|
| baseline | 0.991 | 0.529 | 0.003 | – |
| soft module injected @1000 | 0.991 | 0.545 | 0.004 | 0.00 |
| soft module from step 0 | 1.000 | 0.882 | 0.669 | 0.43 |
| hard (STE) @1000 | 0.995 | 0.680 | 0.336 | 0.26 |
| python (STE) @1000 | 0.991 | 0.535 | 0.005 | 0.00 |
| control (same params, GELU) @1000 | 0.991 | 0.536 | 0.004 | 0.04 |

Bimodal per seed: the module is either adopted (→ 1.0 everywhere) or ignored.
**Zero-gate saddle**: with g=0 and random W_in/W_out, dL/dg = <dL/dh, W_out P(W_in h)> is noise and
W_in/W_out receive no gradient. Early in training loss noise kicks g off zero; at step 1000 it doesn't.

![m1](img/m1_curves.png)

### M1b: teacher-designed interfaces, injected @1000 (`scripts/milestone1b.sh`, 2500 steps)
Teacher init = W_in from a ridge probe of the program's input (x at items, 0 at BOS) at site 0;
W_out columns = unembedding directions of the answer tokens (×20). Gate still starts at 0.

| variant | id | ood | ood2 | steps after inject to ood2>0.95 | ablated ood2 |
|---|---|---|---|---|---|
| baseline | 0.984 | 0.471 | 0.003 | never | – |
| control + teacher interface | 0.985 | 0.471 | 0.004 | never | 0.004 |
| soft, random | 0.984 | 0.472 | 0.003 | never | 0.003 |
| hard, random | 0.987 | 0.600 | 0.203 | never | 0.002 |
| python, random | 0.984 | 0.470 | 0.003 | never | 0.003 |
| **soft, teacher** | 1.000 | 1.000 | 1.000 | 100 / 200 / 100 | 0.000 |
| **hard, teacher** | 1.000 | 1.000 | 1.000 | 100 / 200 / 100 | 0.000 |
| **python (arbitrary numpy), teacher** | 1.000 | 1.000 | 1.000 | 100 / 200 / 100 | 0.000 |

Takeaways:
- Late insertion works reliably **iff the teacher designs the interface**; then even non-differentiable
  Python code is adopted within ~100–200 steps, generalizes to 4× training length, and the model fully
  depends on it (ablation → 0).
- The control (same interface init, no algorithm) does nothing: gains come from the code, not the params.
- Differentiability of the program doesn't matter once the interface is set; the interface is the bottleneck.

![m1b](img/m1b_curves.png)
