#!/usr/bin/env bash
# Milestone 1b: late injection (step 1000) with random vs teacher-designed interfaces.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=runs/m1b
SEEDS=${*:-0 1 2}
COMMON="--task running_sum_mod --V 5 --steps 2500 --eval_every 100 --inject_step 1000 --out_dir $OUT"
T="--init_in probe --init_out unembed --out_scale 20"
mkdir -p $OUT
jobs_=()
for s in $SEEDS; do
  jobs_+=("--name baseline_s$s --seed $s --inject_step -1")
  for p in soft_summod hard_summod py_summod; do
    jobs_+=("--name ${p}_rand_s$s --seed $s --program $p")
    jobs_+=("--name ${p}_teacher_s$s --seed $s --program $p $T")
  done
  jobs_+=("--name control_teacher_s$s --seed $s --program control $T")
done
printf '%s\n' "${jobs_[@]}" | xargs -P "${PAR:-9}" -L1 uv run python -m llmc.train $COMMON | grep -E "done in"
uv run python scripts/plot.py $OUT
