#!/usr/bin/env bash
# Milestone 1: does a gated code module, injected mid-training, get used and help?
# Usage: scripts/milestone1.sh [seeds...]   (default: 0 1 2)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=runs/m1
SEEDS=${*:-0 1 2}
COMMON="--task running_sum_mod --V 5 --steps 4000 --eval_every 200 --out_dir $OUT"
mkdir -p $OUT
jobs_=()
for s in $SEEDS; do
  jobs_+=("--name baseline_s$s --seed $s")
  jobs_+=("--name soft_inject1000_s$s --seed $s --inject_step 1000 --program soft_summod")
  jobs_+=("--name soft_inject0_s$s --seed $s --inject_step 0 --program soft_summod")
  jobs_+=("--name hard_inject1000_s$s --seed $s --inject_step 1000 --program hard_summod")
  jobs_+=("--name py_inject1000_s$s --seed $s --inject_step 1000 --program py_summod")
  jobs_+=("--name control_inject1000_s$s --seed $s --inject_step 1000 --program control")
done
# -L1: one job per line, words appended (avoids macOS xargs -I 255-byte limit)
printf '%s\n' "${jobs_[@]}" | xargs -P "${PAR:-6}" -L1 uv run python -m llmc.train $COMMON | grep -E "done in|injected"
uv run python scripts/plot.py $OUT
