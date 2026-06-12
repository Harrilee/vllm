#!/usr/bin/env bash
# Harvest one finished FP8 DeepEP-LL job's per-bs traces. Usage: ./harvest_deepep_fp8.sh <jid> <ep>
set -uo pipefail; cd "$(dirname "$0")"
JID="$1"; EP="$2"
DIR="vllm_traces/vllm_ep${EP}_${JID}/torchprof"
[ -d "$DIR" ] || { echo "no trace dir $DIR"; exit 1; }
for bsdir in "$DIR"/bs*; do
  [ -d "$bsdir" ] || continue
  bs=$(basename "$bsdir" | sed 's/^bs//')
  n=$(find "$bsdir" -name 'dp*rank0*.pt.trace.json*' 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] && { echo "bs$bs: no rank traces, skip"; continue; }
  python3 scripts/analyze_moe_perf_vllm.py --job "$JID" \
    --torch "$bsdir/dp*rank0*.pt.trace.json*" --phase decode \
    --meta "ep${EP},fp8,bs${bs},deepep_ll_fabric" --csv sweep_data/vllm_results.csv 2>&1 \
    | grep -E "dispatch|combine|moe_compute|appended" | sed "s/^/  [ep$EP bs$bs] /"
done
echo "harvested $JID (ep$EP fp8 deepep_ll)"
