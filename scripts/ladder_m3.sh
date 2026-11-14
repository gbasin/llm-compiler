#!/usr/bin/env bash
# M3 ladder: code-first continual learning from a 752-param core (1 empty block, d=16).
# Each stage: teacher plan (generic ops composed in the register file + one reader), then train
# on the new task with 50% replay of all earlier tasks.
set -euo pipefail
cd "$(dirname "$0")/.."
R=runs/m3/ladder
run() { # name task plan
  uv run python -m llmc.ladder stage --ckpt $R/$PREV/model.pt --out $R/$1 --task $2 --steps 1500 --plan "$3" | tail -1
  PREV=$1
}
mkdir -p $R
uv run python -m llmc.ladder stage --out $R/s0 --task summod --arch 1x0x0 --d 16 --steps 1500 \
  --plan '[{"op":"xval","inputs":["tok"],"out":"x"},{"op":"cumsum_mod","inputs":["x"],"out":"s"},{"reader":"s","site":1}]' | tail -1
PREV=s0
run s1 max         '[{"op":"task","inputs":["tok"],"out":"t"},{"reader":"t","site":0},{"op":"cummax","inputs":["x"],"out":"mx"},{"reader":"mx","site":1}]'
run s2 lag1        '[{"op":"shift","inputs":["x"],"out":"px"},{"reader":"px","site":1}]'
run s3 summod_pair '[{"op":"mul_mod","inputs":["x","px"],"out":"xp"},{"op":"cumsum_mod","inputs":["xp"],"out":"sp"},{"reader":"sp","site":1}]'
run s4 summod_even '[{"op":"map","inputs":["x"],"out":"xe","kwargs":{"table":[0,0,2,0,4]}},{"op":"cumsum_mod","inputs":["xe"],"out":"se"},{"reader":"se","site":1}]'
run s5 depth       '[{"op":"isitem","inputs":["tok"],"out":"isitem"},{"op":"map","inputs":["x"],"out":"dx","kwargs":{"table":[1,4,0,0,0]}},{"op":"mask","inputs":["dx","isitem"],"out":"dxi"},{"op":"cumsum_mod","inputs":["dxi"],"out":"dp"},{"reader":"dp","site":1}]'
run s6 count_eq    '[{"op":"count_prefix_eq","inputs":["tok"],"out":"ce"},{"reader":"ce","site":1}]'
run s7 lag2        '[{"op":"shift","inputs":["x"],"out":"ppx","kwargs":{"k":2}},{"reader":"ppx","site":1}]'
run s8 induct      '[{"op":"prev_occ_next","inputs":["tok","x"],"out":"ind"},{"reader":"ind","site":1}]'
