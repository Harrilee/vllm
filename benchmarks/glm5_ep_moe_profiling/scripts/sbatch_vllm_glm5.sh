#!/usr/bin/env bash
#SBATCH --job-name=vllm-glm5-ep
#SBATCH --partition=batch
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --cpus-per-task=144
#SBATCH --time=01:30:00
#SBATCH --output=logs/vllm_%j.out
#SBATCH --error=logs/vllm_%j.err
#
# vLLM GLM-5 EP MoE profiling launcher (oci-hsg-cs-001, GB200 NVL72, 4 GPU/node).
# vLLM analogue of sbatch_glm5_ep32.sh + sbatch.sh, fused into one launcher.
#
# EP size = nodes x 4:  EP8 = 2 nodes (default), EP16 = 4, EP32 = 8, EP64 = 16.
# Single-rack only (one NVLink clique) — pin --nodelist to same-rack trays.
#
# --mem=0 + cpus-per-task=144: request ALL host RAM, else cgroup OOM on the
# ~450GB GLM-5 weight load (sglang lesson, job 3153355).
#
# Submit (from the cpu alloc, strip inherited SLURM_*):
#   env -u SLURM_JOB_ID -u SLURM_NODELIST -u SLURM_NNODES -u SLURM_NTASKS \
#       -u SLURM_TASKS_PER_NODE -u SLURM_GPUS_PER_NODE -u SLURM_NODEID \
#       -u SLURM_SUBMIT_DIR -u SLURM_CPUS_PER_TASK -u SLURM_MEM_PER_NODE \
#       TMPDIR=/home/harrli/tmp \
#     sbatch --nodes=2 --nodelist=nvl72NNN-T0x,nvl72NNN-T0y sbatch_vllm_glm5.sh

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs vllm_traces

# ---- model (local snapshot dir -> is_local, dodges HF download filelock) ----
export HF_CACHE_HOST="${HF_CACHE_HOST:-/home/harrli/hfcache}"
NVFP4_SNAP="/moe/hfcache/hub/models--nvidia--GLM-5-NVFP4/snapshots/dc54ff55a7e9e71b85db953d8bc22eca894b44c6"
export MODEL="${MODEL:-$NVFP4_SNAP}"
export VLLM_QUANT="${VLLM_QUANT:-modelopt_fp4}"

# ---- container image (shared vLLM sqsh built by vllm_build.sh) ----
PYXIS_IMAGE="${PYXIS_IMAGE:-/home/harrli/sglang-moe-images/vllm-glm5-aarch64.sqsh}"
if [[ ! -f "$PYXIS_IMAGE" ]]; then
    echo "ERROR: vLLM image not found: $PYXIS_IMAGE  (run vllm_build.sh first)" >&2
    exit 1
fi
if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then echo "ERROR: not under SLURM" >&2; exit 1; fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"

# ---- single-rack guard (one NVLink clique = same nvl72NNN prefix) ----
if [[ "${REQUIRE_SINGLE_RACK:-1}" == "1" && "${SLURM_NNODES:-1}" -gt 1 ]]; then
    _racks="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -E 's/-T[0-9]+$//' | sort -u)"
    _nrack="$(echo "$_racks" | grep -c .)"
    if [[ "$_nrack" -ne 1 ]]; then
        echo "ERROR: allocation spans $_nrack NVL72 racks; DeepEP-LL needs ONE." >&2
        echo "  racks: $(echo "$_racks" | tr '\n' ' ')" >&2
        echo "  nodes: $SLURM_JOB_NODELIST" >&2
        echo "  Resubmit with --nodelist=nvl72NNN-T[..] for a single-rack clique." >&2
        exit 1
    fi
    echo "  single-rack OK: $_racks ($SLURM_NNODES trays, one NVLink clique)"
fi

# ---- DP/EP derivation + head node addr ----
_GPN="${SLURM_GPUS_PER_NODE:-4}"; _GPN="${_GPN##*:}"
DP=$(( SLURM_NNODES * ${_GPN:-4} ))
HEAD_NODE_HOSTNAME="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
# Resolve head hostname to a routable IP (vLLM --data-parallel-address wants IP).
HEAD_NODE_IP="$(getent hosts "$HEAD_NODE_HOSTNAME" | awk '{print $1; exit}')"
[[ -z "$HEAD_NODE_IP" ]] && HEAD_NODE_IP="$HEAD_NODE_HOSTNAME"
DP_RPC_PORT="${DP_RPC_PORT:-13345}"
PORT="${PORT:-8000}"

# per-job output dirs (NFS) for logs + traces
OUT_LABEL="vllm_ep${DP}_${SLURM_JOB_ID:-local}"
mkdir -p "$REPO_ROOT/logs/$OUT_LABEL" "$REPO_ROOT/vllm_traces/$OUT_LABEL"

cat <<EOF
===================================================================
vLLM GLM-5 EP MoE profiling
  job:       ${SLURM_JOB_ID:-local}
  nodes:     $SLURM_JOB_NODELIST
  head:      $HEAD_NODE_HOSTNAME ($HEAD_NODE_IP):$DP_RPC_PORT
  image:     $PYXIS_IMAGE
  model:     $MODEL   quant=$VLLM_QUANT
  DP/EP:     $DP across $SLURM_NNODES nodes (local=${_GPN})
  traces ->  $REPO_ROOT/vllm_traces/$OUT_LABEL
===================================================================
EOF

WATCHDOG_SEC="${WATCHDOG_SEC:-5400}"
(
  sleep "$WATCHDOG_SEC"
  echo "[watchdog] cancelling job after ${WATCHDOG_SEC}s"
  scancel --signal=TERM "${SLURM_JOB_ID}" 2>/dev/null || true
) &
WATCHDOG_PID=$!

# ---- optional DeepEP use_fabric overlay (cross-node fix) ----
# When VLLM_DEEPEP_USE_FABRIC=1: bind-mount the hybrid-ep DeepEP build (has
# use_fabric -> CU_MEM_HANDLE_TYPE_FABRIC) at /moe/deepep_overlay and overlay the
# patched vLLM all2all.py (passes use_fabric=True) over the image's copy.
MOUNTS="$REPO_ROOT/logs/$OUT_LABEL:/moe/logs,$REPO_ROOT/vllm_traces/$OUT_LABEL:/moe/vllm_traces,/tmp:/tmp,${HF_CACHE_HOST:-/tmp/hfcache}:/moe/hfcache,$REPO_ROOT/vllm_moe_trace.sh:/moe/vllm_moe_trace.sh"
if [[ "${VLLM_DEEPEP_USE_FABRIC:-0}" == "1" ]]; then
    DEEPEP_OVERLAY="${DEEPEP_OVERLAY:-/home/harrli/deepep-hybrid-install}"
    PATCHED_A2A="${PATCHED_A2A:-/home/harrli/vllm/vllm/distributed/device_communicators/all2all.py}"
    _IMG_A2A=/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/all2all.py
    MOUNTS="$MOUNTS,$DEEPEP_OVERLAY:/moe/deepep_overlay,$PATCHED_A2A:$_IMG_A2A"
fi

# ---- one srun, all nodes, pyxis. Each task derives NODE_RANK from SLURM_NODEID. ----
srun \
    --ntasks-per-node=1 \
    --mpi=pmix \
    --container-image="$PYXIS_IMAGE" \
    --container-mounts="$MOUNTS" \
    --container-workdir=/vllm-workspace \
    --export=ALL,MODEL="$MODEL",VLLM_QUANT="$VLLM_QUANT",DP=$DP,DP_LOCAL=${_GPN},NNODES="$SLURM_NNODES",DP_ADDR="$HEAD_NODE_IP",DP_RPC_PORT="$DP_RPC_PORT",PORT="$PORT",ALL2ALL_BACKEND="${ALL2ALL_BACKEND:-flashinfer_nvlink_one_sided}",MOE_BACKEND="${MOE_BACKEND:-flashinfer_cutlass}",MAX_MODEL_LEN="${MAX_MODEL_LEN:-9600}",VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}",VLLM_PROFILE="${VLLM_PROFILE:-1}",VLLM_PROFILE_BS="${VLLM_PROFILE_BS:-8}",VLLM_PROFILE_DELAY="${VLLM_PROFILE_DELAY:-3}",VLLM_PROFILE_ACTIVE="${VLLM_PROFILE_ACTIVE:-8}",VLLM_LONG_ISL="${VLLM_LONG_ISL:-}",HF_HOME=/moe/hfcache,HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}",MOE_TRACE_DIR=/moe/vllm_traces,VLLM_TORCH_PROFILER_DIR=/moe/vllm_traces/torchprof,VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL="${VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL:-1}",VLLM_DEEPEPLL_NVFP4_DISPATCH="${VLLM_DEEPEPLL_NVFP4_DISPATCH:-0}",VLLM_DEEPEP_USE_FABRIC="${VLLM_DEEPEP_USE_FABRIC:-0}",NVSHMEM_CUMEM_HANDLE_TYPE="${NVSHMEM_CUMEM_HANDLE_TYPE:-}",NVSHMEM_DISABLE_CUDA_VMM="${NVSHMEM_DISABLE_CUDA_VMM:-}",MC_FORCE_MNNVL="${MC_FORCE_MNNVL:-}",IFACE="${IFACE:-}" \
    bash -c '
        export NODE_RANK="${SLURM_NODEID:-0}"
        export SERVER_LOG="/moe/logs/vllm_server.${NODE_RANK}.log"
        exec bash /moe/vllm_moe_trace.sh
    '
RC=$?

kill $WATCHDOG_PID 2>/dev/null || true
echo "vllm job rc=$RC"
N=$(find "$REPO_ROOT/vllm_traces/$OUT_LABEL" -name '*.pt.trace.json*' 2>/dev/null | wc -l)
echo "  traces: $N files in $REPO_ROOT/vllm_traces/$OUT_LABEL/"
exit $RC
