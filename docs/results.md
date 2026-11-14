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

## M2 — does moving skill A into code free capacity for skill B? (disjoint B)
Multitask NoPE GPT, d=32, 2 layers (26.5k neural params). Task token prefixes each sequence.
A = {summod, max} for 2000 steps; module (routed gate, teacher interface, truth contract) injected at 800.
B = {induct, count_eq, lag2} for 3000 steps with 30% A replay. Pressure arms from step 800:
`window` = group lasso (l1 3e-3) on MLP neurons + heads until end of A; `recycle` = reinit all units with
group norm < 0.05 at start of B; `sparsity` = group lasso for the whole run. 3 seeds (`scripts/milestone2.sh`).

| arm | A id | A ood | B id (per seed) | B auc | core params end |
|---|---|---|---|---|---|
| no module | 0.917 | 0.536 | 0.526 (0.54/0.51/0.53) | 0.394 | 26.5k |
| no module + window+recycle | 0.916 | 0.540 | 0.539 | 0.413 | 26.5k |
| control module | 0.921 | 0.541 | 0.536 | 0.401 | 26.5k |
| **py module** | 0.999 | 0.994 | **0.711** (0.67/0.74/0.73) | **0.514** | 26.5k |
| soft module | 0.999 | 0.992 | 0.711 | 0.514 | 26.5k |
| py + window (prune, no recycle) | 1.000 | 0.984 | 0.521 | 0.413 | 17.6k |
| py + window+recycle | 0.999 | 0.954 | 0.721 | 0.513 | 26.5k |
| py + continuous sparsity | 0.999 | 0.928 | 0.459 | 0.358 | 9.9k |

(Calibration, 1 seed: no module at d=48 / 58k params got B id 0.62.)

Takeaways:
- Externalizing sum-mod raises B learning by ~18 points with non-overlapping seeds; the control rules
  out "just more params". It beat a 2.2× wider network (1 seed).
- With the module, A needs almost no MLP: during the prune window 118–128/128 neurons died in both layers
  while A stayed ~perfect (attention + code carry A).
- Explicit prune→recycle gave no gain over plain continued training — gradient descent already reallocates.
  Pruning without recycling hurts (dead units stay dead). Continuous sparsity gives a 63% smaller core
  that holds A but can't learn B: there's a real core-size vs. new-learning trade-off.

## M2 — compositional B: reuse the sum-mod code on a feature the network must build
B = {summod_even: Σ x_j·[x_j even] mod 5, summod_pair: Σ x_j·x_{j-1} mod 5}. At the start of B the teacher
inserts a second instance of the *same* sum-mod code reading site 1 (after block 0), with a new contract
whose labels are the per-item summand ("x if even else 0", "x·x_{i-1} mod 5"). 3 seeds.

| arm | even id/ood | pair id/ood | even ood w/ modules ablated |
|---|---|---|---|
| no module | 0.90 / 0.16 | 0.18 / 0.00 | – |
| py A-module only | 0.96 / 0.28 | 0.21 / 0.00 | 0.28 |
| + py2 @site1, frozen probe interface (no contract) | 0.97 / 0.30 | 0.20 / 0.00 | 0.30 |
| + control2 @site1 + contract (supervision, no algorithm) | 0.94 / 0.21 | 0.20 / 0.00 | 0.11 |
| **+ py2 @site1 + teacher contract** | **1.00 / 0.99** | 0.21 / 0.00 | 0.00 |
| **+ soft2 @site1, no contract (gradient through code)** | **1.00 / 0.99** | 0.22 / 0.00 | 0.00 |

Takeaways:
- A new contract teaches the network to *build* the input the existing code needs ("x if even else 0")
  and route it into non-differentiable code: 0.28 → 0.99 ood, full dependence. The contract-only control
  shows this is not just auxiliary supervision.
- For this simple feature, gradient through soft code discovers the same thing without a contract.
- summod_pair fails in every arm. Contract agreement plateaus at ~0.88 (the even part only): the
  d=32 NoPE block 0 can't produce x_i·x_{i-1} at site 1. The bottleneck is building the input
  representation, not the computation — the case where the teacher would need another code primitive
  (e.g. a `prev_token` module feeding a `mul_mod` module), i.e. module composition.

## M3 — register file + curriculum ladder, code-first continual learning
Hybrid model (`hybrid.py`): growable core + register file of per-position int slots shared by code
(`oplib.py` generic ops: xval, task, shift, add/mul/cumsum mod, cummax, eq, mask, map-table,
count_prefix_eq, prev_occ_next, isitem), readers (zero-init lookup table + router; no zero-gate saddle)
and contract-trained decoders. Ladder (`scripts/ladder_m3.sh`): 9 tasks added one per stage, 50% replay of
all earlier tasks; a stage ends when the new task is ≥0.99 id / ≥0.95 ood **and** no earlier task dropped
>1 pt (id or ood).

Core: 1 block with **0 heads, 0 MLP**, d=16 → 752 params (embeddings + unembedding).
Without code this core gets summod to 14% and, given max next, forgets summod (−43 pts) while learning nothing.

| stage | task | plan (ops → reader) | steps | core | interface params | ops | table entries | worst regression |
|---|---|---|---|---|---|---|---|---|
| 0 | summod | xval, cumsum_mod | 200 | 752 | 129 | 2 | 0 | – |
| 1 | max | task(+reader@0 for routing), cummax | 200 | 752 | 627 | 4 | 0 | 0 |
| 2 | lag1 | shift | 100 | 752 | 756 | 5 | 0 | 0 |
| 3 | summod_pair | mul_mod(x, shift x), cumsum_mod | 200 | 752 | 885 | 7 | 0 | 0 |
| 4 | summod_even | map[0,0,2,0,4], cumsum_mod | 200 | 752 | 1014 | 9 | 5 | 0 |
| 5 | depth | isitem, map[1,4,0,0,0], mask, cumsum_mod | 200 | 752 | 1143 | 13 | 10 | 0 |
| 6 | count_eq | count_prefix_eq(tok) | 200 | 752 | 1272 | 14 | 10 | 0 |
| 7 | lag2 | shift k=2 | 200 | 752 | 1401 | 15 | 10 | 0 |
| 8 | induct | prev_occ_next(tok, x) | 200 | 752 | 1530 | 16 | 10 | 0 |

All 9 tasks end at 1.00 id and 1.00 ood (2× train length), zero regression at every stage, core never grew.
summod_pair (failed in M2) is solved by code→code composition through slots.

Caveat: every ladder task is exactly expressible by composing generic ops, so the network only routes and
the teacher does the algorithmic work; growth was never needed. Testing grow-vs-code needs tasks that
are *not* cheaply codeable (statistical / soft structure), with an MDL-style cost for code + tables vs weights.

## M4 — grow on demand where code can't help (hidden grammar), then partly can (planted copies)
`lm.py`: next-token LM from an empty core (1 block, 0 heads, 0 MLP, d=16). Trial-based growth: on a
plateau, try +1 head / +8 MLP per layer / +1 layer (1 head, 8 MLP) for 1000 steps each against a
no-growth control from identical state and data; grow only if a candidate beats the control by >0.005
nats, choosing best gain per added param; stop when nothing beats the control.

Data: random HMM (16 states, 2 successors each, 16 tokens, emission Dirichlet α=0.1) with secret
parameters → not codeable by the teacher. Exact Bayes-optimal loss via the forward algorithm: 1.695
(unigram 2.42).

### Pure HMM: grown vs fixed
| model | core params | loss | gap to Bayes |
|---|---|---|---|
| fixed 1 layer, 1 head | 1,136 | 2.055 | 0.360 |
| fixed 1 layer, 2 heads, 32 MLP | 2,704 | 2.047 | 0.352 |
| fixed 2 layers, 2 heads, 32 MLP | 4,864 | 1.890 | 0.195 |
| **grown** → 6 layers ≈1 head each, MLP 0–16 | 6,192 | **1.778** | **0.083** |

Growth path: +head → +layer → +layer → +head → +8mlp → +layer ×3 → stop. The policy discovers that this
task wants **depth, not width** (iterated belief updates); fixed references confirm width barely helps.

### HMM + planted copy spans (50% of sequences repeat an earlier 4–8 token span after a COPY marker)
Code = generic `suffix_next` / `suffix_len` ops (continuation + length of the longest earlier suffix
match), read at site 0. Both grown from empty.

| run | final core | loss all | copy-span tokens | grammar tokens |
|---|---|---|---|---|
| grown, no code | 4,776 | 1.871 | 2.094 | 1.858 |
| grown + suffix-match code | **4,200** | **1.821** | **1.672** | 1.829 |

Code improves copy-span loss and ends with a 12% smaller core. But copied tokens are deterministic
(ideal ≈ 0): the network doesn't fully trust the op because short spurious suffix matches are common in
HMM text and it can't tell it's inside a copy span — again an interface/routing bottleneck, not compute.
