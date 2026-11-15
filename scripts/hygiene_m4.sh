#!/usr/bin/env bash
# M4 hygiene: Tiny Shakespeare, fixed sizes x {no code, n-gram table, suffix, both}, 3 seeds, 30k steps.
set -uo pipefail
cd "$(dirname "$0")/.."
S=char:data/tinyshakespeare.txt
SUF='[{"op":"suffix_next","inputs":["tok"],"out":"sn"},{"op":"suffix_len","inputs":["tok"],"out":"sl"},{"reader":"sn","site":0},{"reader":"sl","site":0}]'
jobs=()
for seed in 0 1 2; do
  for A in 1x0x0 2x2x64 4x4x128; do
    [ $A != 1x0x0 ] && jobs+=("--name h_${A}_none_s$seed --arch $A --seed $seed")
    jobs+=("--name h_${A}_ngram_s$seed --arch $A --seed $seed --ngram 4")
  done
  jobs+=("--name h_2x2x64_suffix_s$seed --arch 2x2x64 --seed $seed --plan '$SUF'")
  jobs+=("--name h_2x2x64_both_s$seed --arch 2x2x64 --seed $seed --ngram 4 --plan '$SUF'")
done
printf '%s\n' "${jobs[@]}" | xargs -P ${PAR:-6} -L1 sh -c 'uv run python -m llmc.lm --out_dir runs/m4h --source '"$S"' --d 32 --rounds 60 --no_grow --threads 2 "$@" | tail -1' _
