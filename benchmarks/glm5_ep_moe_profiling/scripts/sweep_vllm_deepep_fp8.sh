#!/usr/bin/env bash
# Full FP8 DeepEP-LL grid on vLLM: EP8/16/32/64 × bs{1,2,8,32,64,128}.
# Uses the hybrid-ep use_fabric overlay (the cross-node DeepEP fix). This is the
# true reproduction of the sglang HTML DeepEP-LL tables on vLLM.
#
# Each job = one EP size, full bs sweep in one server launch. DeepEP needs a
# SINGLE NVLink clique = ONE NVL72 rack, so all multi-node jobs use --switches=1
# + the launcher's single-rack guard (REQUIRE_SINGLE_RACK=1).
#   EP8=2 trays, EP16=4, EP32=8, EP64=16 — all <=18 trays, fit one rack.
#
# Usage:  ./sweep_vllm_deepep_fp8.sh                 # all EP
#         EP_LIST="8 16" ./sweep_vllm_deepep_fp8.sh
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p .sweep_done_deepep logs vllm_traces

IMG=/home/harrli/sglang-moe-images/vllm-glm5-aarch64.sqsh
OVERLAY=/home/harrli/deepep-hybrid-install
[ -f "$IMG" ] || { echo "ERROR: image missing $IMG"; exit 1; }
[ -d "$OVERLAY" ] || { echo "ERROR: DeepEP use_fabric overlay missing $OVERLAY"; exit 1; }
FP8_SNAP="/moe/hfcache/hub/models--zai-org--GLM-5-FP8/snapshots/4f96cc5eec29dcee5d6ded54f7ffe889438f9516"

EP_LIST="${EP_LIST:-8 16 32 64}"
BS_SWEEP="${BS_SWEEP:-1-2-8-32-64-128}"
SWITCH_WAIT="${SWITCH_WAIT:-01:00:00}"   # wait up to 1h for a single-block alloc

UNS="-u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_NODELIST -u SLURM_NNODES -u SLURM_NTASKS -u SLURM_PROCID -u SLURM_TASKS_PER_NODE -u SLURM_GPUS_PER_NODE -u SLURM_NODEID -u SLURM_SUBMIT_DIR -u SLURM_CPUS_PER_TASK -u SLURM_MEM_PER_NODE -u SLURM_JOB_NODELIST"

echo "FP8 DeepEP-LL grid: EP{${EP_LIST// /,}} × bs{$BS_SWEEP}"
for ep in $EP_LIST; do
  nodes=$(( ep / 4 ))
  label="D_fp8_deepep_ll_ep${ep}"
  [ -f ".sweep_done_deepep/$label" ] && { echo "$label already submitted ($(cat .sweep_done_deepep/$label)), skip"; continue; }
  # --switches=1 packs into 1 topology block = 1 NVL72 rack (DeepEP single-clique).
  out=$(env $UNS TMPDIR=/home/harrli/tmp \
    VLLM_DEEPEP_USE_FABRIC=1 NVSHMEM_CUMEM_HANDLE_TYPE=FABRIC NVSHMEM_DISABLE_CUDA_VMM=0 MC_FORCE_MNNVL=1 \
    sbatch --no-requeue --nodes="$nodes" --switches="1@${SWITCH_WAIT}" \
    --export=ALL,PYXIS_IMAGE=$IMG,REQUIRE_SINGLE_RACK=1,MODEL="$FP8_SNAP",VLLM_QUANT=fp8,ALL2ALL_BACKEND=deepep_low_latency,MOE_BACKEND=auto,VLLM_DEEPEP_USE_FABRIC=1,NVSHMEM_CUMEM_HANDLE_TYPE=FABRIC,NVSHMEM_DISABLE_CUDA_VMM=0,MC_FORCE_MNNVL=1,VLLM_PROFILE_BS="$BS_SWEEP" \
    sbatch_vllm_glm5.sh 2>&1 | tail -1)
  jid=$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
  echo "SUBMIT $label (nodes=$nodes) -> ${jid:-PARSE_FAIL($out)}"
  [ -n "$jid" ] && echo "$jid" > ".sweep_done_deepep/$label"
  sleep 3
done
echo "FP8 DeepEP-LL grid submitted (SLURM queues any waiting for a single-rack block)."
echo "Harvest each when done:  ./harvest_vllm.sh <jid> <ep> deepep_ll_fabric"
echo "  (note: harvest meta tag distinguishes FP8 LL from the NVFP4 sweep rows)"