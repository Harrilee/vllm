#!/usr/bin/env bash
# Harvest one finished vLLM sweep job's per-bs traces into vllm_results.csv.
# Usage: ./harvest_vllm.sh <jobid> <ep> <backend_tag>
#   e.g. ./harvest_vllm.sh 3263865 4 nvlink1s
set -uo pipefail
cd "$(dirname "$0")"
JID="$1"; EP="$2"; TAG="$3"
DIR="vllm_traces/vllm_ep${EP}_${JID}/torchprof"
[ -d "$DIR" ] || { echo "no trace dir $DIR"; exit 1; }
for bsdir in "$DIR"/bs*; do
  [ -d "$bsdir" ] || continue
  bs=$(basename "$bsdir" | sed 's/^bs//')
  n=$(find "$bsdir" -name 'dp*rank0*.pt.trace.json*' 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] && { echo "bs$bs: no rank traces, skip"; continue; }
  python3 scripts/analyze_moe_perf_vllm.py --job "$JID" \
    --torch "$bsdir/dp*rank0*.pt.trace.json*" \
    --phase decode --meta "ep${EP},nvfp4,bs${bs},${TAG}" \
    --csv sweep_data/vllm_results.csv 2>&1 | grep -E "per-op|dispatch|combine|moe_compute|appended" | sed "s/^/  [bs$bs] /"
done
echo "harvested $JID (ep$EP $TAG)"