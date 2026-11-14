# llm-compiler

Experiments on moving computation in small transformers from weights into explicit code.

See [docs/background.md](docs/background.md) for the motivating discussion and related work.

## Decisions (2026-11-14)
- **Direction:** inject gated code modules during training (doubleAI-style), then test whether the
  model reuses freed capacity (learn skills A → insert code → substitution pressure → learn B, retain A).
- **Models:** tiny toy transformers (~1–10M params) on algorithmic tasks first; scale to nanoGPT/nanochat later.
- **Teacher:** Claude Code acts as the teacher interactively — reads failures / phantom gradients,
  writes modules, chooses insertion points. No separate agent infra.
- **Compute:** local only, Apple M5 Pro (64 GB), MPS. Optimize for many fast, small runs.
- **Stack:** PyTorch + MPS, managed with uv.
- **Modules:** tiered — differentiable torch tensor programs first; add a discrete Python path
  (straight-through / supervised interfaces) once the basics work.
- **Tracking:** local files — per-run JSONL metrics + config snapshot under `runs/`, plus a plotting script.

## Milestone 1: reproduce the mechanism
Tiny transformer on a task like running-max or key→value lookup. Partway through training,
hand-inject one module `h_j += g · W_out · P(W_in · h_i)` with `g` initialized to 0. Success =
the gate opens, held-out / length-generalization accuracy jumps vs. a no-injection baseline, and
ablating the module confirms the model depends on it.

## Layout
- `src/llmc/tasks.py` — scan tasks (`running_sum_mod`, `running_max`), format `BOS x1 … xn`, predict `y_i` at `x_i`.
- `src/llmc/model.py` — tiny NoPE GPT; residual *sites* `0..n_layers` where modules read/write.
- `src/llmc/programs.py` — the code library. Tier 1: exact distributional scans (`soft_*`, `hard_*` STE).
  Tier 2: `PyProgram` (arbitrary numpy on argmax ints, straight-through grad). `control`: same params, no algorithm.
- `src/llmc/modules.py` — `CodeModule`: `h_write += gate · W_out(P(W_in(LN(h_read))))`, gate init 0.
- `src/llmc/train.py` — training loop, mid-run injection, JSONL metrics, end-of-run ablation.
- `src/llmc/teacher.py` — diagnostics for choosing modules/sites: accuracy by position, phantom
  gradients per site, linear probes per site, module ablation.
- `scripts/milestone1.sh`, `scripts/plot.py` — experiment sweep and comparison plot/table.

## Usage
```bash
uv run python -m llmc.train --name baseline
uv run python -m llmc.train --name inject --inject_step 1000 --program soft_summod
uv run python -m llmc.teacher runs/baseline/final.pt
PAR=9 scripts/milestone1.sh 0 1 2
```
Tiny models run faster on CPU than MPS here; each process uses `--threads 2`, so run several in parallel.
