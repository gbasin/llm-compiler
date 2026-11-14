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
