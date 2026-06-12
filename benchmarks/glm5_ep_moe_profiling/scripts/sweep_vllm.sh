#!/usr/bin/env bash
# vLLM GLM-5 EP MoE full sweep: 3 a2a backends × EP{4,8,16,32,64} × bs{1,2,8,32,64,128}.
# Mirrors sweep_queue_fp16.sh. Each job = one (backend, EP) and runs the FULL bs
# sweep in a single server launch (vllm_moe_trace.sh loops VLLM_PROFILE_BS).
#
# a2a backends (all NVFP4, validated on this image):
#   flashinfer_nvlink_one_sided  — TRT-LLM NVLink one-sided LL a2a (closest to sglang DeepEP-LL)
#   allgather_reducescatter      — NCCL allgather/reducescatter (the "NCCL" path)
#   deepep_high_throughput       — DeepEP HT (Standard fmt; LL blocked by CUTEDSL bug)
# All use --moe-backend flashinfer_cutlass (dodges broken FLASHINFER_CUTEDSL_BATCHED).
#
# EP4=1 tray, EP8=2, EP16=4, EP32=8, EP64=16 — multi-node pinned to ONE NVL72 rack.
# Usage:  ./sweep_vllm.sh                 # all backends, all EP
#         BACKENDS="flashinfer_nvlink_one_sided" EP_LIST="4 8" ./sweep_vllm.sh
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p .sweep_done_vllm logs vllm_traces

IMG=/home/harrli/sglang-moe-images/vllm-glm5-aarch64.sqsh
[ -f "$IMG" ] || { echo "ERROR: image missing $IMG"; exit 1; }

BACKENDS="${BACKENDS:-flashinfer_nvlink_one_sided allgather_reducescatter deepep_high_throughput}"
EP_LIST="${EP_LIST:-4 8 16 32 64}"
BS_SWEEP="${BS_SWEEP:-1-2-8-32-64-128}"   # dash list (sbatch --export comma-safe)
MOE_BACKEND="${MOE_BACKEND:-flashinfer_cutlass}"

UNS="-u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_NODELIST -u SLURM_NNODES -u SLURM_NTASKS -u SLURM_PROCID -u SLURM_TASKS_PER_NODE -u SLURM_GPUS_PER_NODE -u SLURM_NODEID -u SLURM_SUBMIT_DIR -u SLURM_CPUS_PER_TASK -u SLURM_MEM_PER_NODE -u SLURM_JOB_NODELIST"

ep_to_nodes() { echo $(( $1 / 4 )); }   # 4 GPU/node

# Pick a rack with >=need bare State=IDLE trays (reject DRAIN/RESERVED -> pin-death).
# $1=need  $2=used-racks-csv -> "rack|tray-csv"  (empty if none free now)
pick_rack() {
  local need=$1 used=",${2}," rack t st trays
  for rack in $(sinfo -N -p batch -h -o "%N %t" 2>/dev/null | awk '$2=="idle"{print $1}' \
                | grep -oE 'nvl72[0-9]+' | sort | uniq -c | sort -rn | awk '$1>='"$need"'{print $2}'); do
    case "$used" in *",$rack,"*) continue;; esac
    trays=""
    for t in $(seq -w 1 18); do
      st=$(scontrol show node "${rack}-T${t}" 2>/dev/null | grep -oE 'State=[A-Za-z+]+' | head -1)
      [ "$st" = "State=IDLE" ] && trays="$trays ${rack}-T${t}"
    done
    if [ "$(echo $trays | wc -w)" -ge "$need" ]; then
      echo "$rack|$(echo $trays | tr ' ' '\n' | head -"$need" | paste -sd,)"; return 0
    fi
  done
  return 1
}

# Build the queue: label|nodes|backend
QUEUE=()
for be in $BACKENDS; do
  for ep in $EP_LIST; do
    QUEUE+=("V_${be}_ep${ep}|$(ep_to_nodes "$ep")|$be")
  done
done
echo "Queue: ${#QUEUE[@]} jobs (${BACKENDS// /,} × EP{${EP_LIST// /,}})"

# Submit ALL jobs nodelist-free. CRITICAL: on this saturated cluster, pinning a
# specific --nodelist races the scheduler — trays flip to IDLE+PLANNED within
# seconds of the pick, so the pinned job dies with ReqNodeNotAvail. Instead let
# SLURM place + queue (PENDING(Resources)), and for multi-node use --switches=1
# so SLURM packs the allocation into ONE topology block = ONE NVL72 rack (the
# single-NVLink-clique requirement) without us naming nodes. Validated: jobs
# 3264087/88/89 placed instantly this way after --nodelist attempts dead-pinned.
SWITCH_WAIT="${SWITCH_WAIT:-00:30:00}"   # max wait for a single-block alloc
for line in "${QUEUE[@]}"; do
  IFS='|' read -r label nodes be <<<"$line"
  [ -f ".sweep_done_vllm/$label" ] && continue
  SW=(); sr=0
  if [ "$nodes" -gt 1 ]; then SW=(--switches="1@${SWITCH_WAIT}"); sr=1; fi
  out=$(env $UNS TMPDIR=/home/harrli/tmp \
    sbatch --no-requeue --nodes="$nodes" "${SW[@]}" \
    --export=ALL,PYXIS_IMAGE="$IMG",REQUIRE_SINGLE_RACK=$sr,ALL2ALL_BACKEND="$be",MOE_BACKEND="$MOE_BACKEND",VLLM_PROFILE_BS="$BS_SWEEP" \
    sbatch_vllm_glm5.sh 2>&1 | tail -1)
  jid=$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
  echo "SUBMIT $label (nodes=$nodes, switches=1) -> ${jid:-PARSE_FAIL($out)}"
  [ -n "$jid" ] && echo "$jid" > ".sweep_done_vllm/$label"
  sleep 2
done
echo "VLLM SWEEP QUEUE EMPTY — all submitted (SLURM queues any that wait for capacity)"