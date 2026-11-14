#!/usr/bin/env bash
# Milestone 2: does moving skill A into code free capacity to learn skill B?
# Usage: B=disjoint|comp scripts/milestone2.sh [seeds...]
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${*:-0 1 2}
case "${B:-disjoint}" in
  disjoint) TB="induct,count_eq,lag2" ;;
  comp)     TB="summod_even,summod_pair" ;;
esac
OUT=runs/m2_${B:-disjoint}
COMMON="--d_model 32 --steps_A 2000 --steps_B 3000 --inject_step 800 --l1 3e-3 --tasks_B $TB --out_dir $OUT"
jobs_=()
if [ "${B:-disjoint}" = comp ]; then
  for s in $SEEDS; do
    jobs_+=("--name base_none_s$s --seed $s")
    jobs_+=("--name py_none_s$s --seed $s --program py_summod")
    # teacher proposes a new contract at site 1 for the existing sum-mod code
    jobs_+=("--name py+py2contract_s$s --seed $s --program py_summod --program2 py_summod")
    # same new module, no contract: frozen probe-init interface only
    jobs_+=("--name py+py2nocontract_s$s --seed $s --program py_summod --program2 py_summod --contract2 none")
    # soft code at site 1, no contract: can gradient through code discover the feature?
    jobs_+=("--name py+soft2nocontract_s$s --seed $s --program py_summod --program2 soft_summod --contract2 none")
    # control: contract supervision with no algorithm behind it
    jobs_+=("--name py+control2contract_s$s --seed $s --program py_summod --program2 control")
  done
fi
for s in $( [ "${B:-disjoint}" = comp ] || echo $SEEDS ); do
  jobs_+=("--name base_none_s$s --seed $s")
  jobs_+=("--name base_window+recycle_s$s --seed $s --pressure window+recycle")
  jobs_+=("--name py_none_s$s --seed $s --program py_summod")
  jobs_+=("--name py_window_s$s --seed $s --program py_summod --pressure window")
  jobs_+=("--name py_window+recycle_s$s --seed $s --program py_summod --pressure window+recycle")
  jobs_+=("--name py_sparsity_s$s --seed $s --program py_summod --pressure sparsity")
  jobs_+=("--name soft_none_s$s --seed $s --program soft_summod")
  jobs_+=("--name control_none_s$s --seed $s --program control")
done
printf '%s\n' "${jobs_[@]}" | xargs -P "${PAR:-9}" -L1 uv run python -m llmc.train2 $COMMON | grep -E "done in" || true
uv run python scripts/m2_table.py $OUT
