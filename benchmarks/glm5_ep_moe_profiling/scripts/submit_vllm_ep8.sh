#!/usr/bin/env bash
# Submit the EP8 NVFP4 milestone (runbook §9): pin 2 same-rack trays (one NVLink
# clique) and launch sbatch_vllm_glm5.sh. If no fully-IDLE 2-tray block is free
# right now, pin a fully-HEALTHY rack and let SLURM queue PENDING(Resources) —
# the proven sglang strategy (no pin-death on DRAIN/RESERVED trays).
set -uo pipefail
cd "$(dirname "$0")"

SQSH="${PYXIS_IMAGE:-/home/harrli/sglang-moe-images/vllm-glm5-aarch64.sqsh}"
[ -f "$SQSH" ] || { echo "ERROR: image not ready: $SQSH (run vllm_build.sh)"; exit 1; }
NEED="${NEED:-2}"   # EP8 = 2 trays

# Strip inherited SLURM_* so the nested sbatch isn't constrained by the cpu alloc.
UNS="-u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_NODELIST -u SLURM_NNODES -u SLURM_NTASKS -u SLURM_PROCID -u SLURM_TASKS_PER_NODE -u SLURM_GPUS_PER_NODE -u SLURM_NODEID -u SLURM_SUBMIT_DIR -u SLURM_CPUS_PER_TASK -u SLURM_MEM_PER_NODE -u SLURM_JOB_NODELIST"

# Pick a rack with >=NEED bare State=IDLE trays (reject +DRAIN/+RESERVED — those
# pin-death as ReqNodeNotAvail). Falls back to a fully-healthy rack to queue on.
pick_rack() {
  local need=$1
  for rack in $(sinfo -N -p batch -h -o "%N %t" 2>/dev/null | awk '$2=="idle"{print $1}' \
                | grep -oE 'nvl72[0-9]+' | sort | uniq -c | sort -rn | awk '$1>='"$need"'{print $2}'); do
    local trays=""
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

rp="$(pick_rack "$NEED")"
if [ -z "$rp" ]; then
  echo "No same-rack ${NEED}-tray IDLE block free now."
  echo "Re-run this script when capacity frees up, or pass --nodelist manually."
  exit 2
fi
rack="${rp%%|*}"; pick="${rp##*|}"
echo "EP8 -> rack $rack, trays $pick"

out=$(env $UNS TMPDIR=/home/harrli/tmp PYXIS_IMAGE="$SQSH" \
  sbatch --no-requeue --nodes="$NEED" --nodelist="$pick" \
  --export=ALL,PYXIS_IMAGE="$SQSH",VLLM_PROFILE_BS="${VLLM_PROFILE_BS:-8}" \
  sbatch_vllm_glm5.sh 2>&1 | tail -2)
echo "$out"
