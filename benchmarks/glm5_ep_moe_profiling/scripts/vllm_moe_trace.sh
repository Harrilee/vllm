#!/usr/bin/env bash
# vLLM GLM-5 EP MoE profiling driver (in-container, per-node).
#
# vLLM analogue of moe_trace.sh. Brings up a vLLM OpenAI server with DeepEP
# expert-parallel + DP attention across N GB200 nodes (4 GPU/node), drives the
# native torch profiler via /start_profile + /stop_profile, and captures
# prefill + decode per-kernel traces (*.pt.trace.json.gz) for the EP-MoE study.
#
# Multi-node DP launch (native, non-Ray) follows vLLM's EP deployment doc
# (docs/serving/expert_parallel_deployment.md):
#   - HEAD node (NODE_RANK 0): runs the API server + local DP ranks
#   - WORKER nodes: --headless --data-parallel-start-rank <4*NODE_RANK>
#   Both share --data-parallel-size (global) + --data-parallel-address/-rpc-port.
#
# Profiling uses --profiler-config (this vLLM has NO VLLM_TORCH_PROFILER_DIR):
#   --profiler-config '{"profiler":"torch","torch_profiler_dir":"...",
#                       "delay_iterations":N,"active_iterations":M}'
# /start_profile and /stop_profile take NO JSON body — the dir+schedule come
# only from the launch config. torch.profiler (CUPTI) captures CUDA-graph-
# internal kernels on replay, same mechanism as sglang.

set -uo pipefail

ENGINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- knobs (env-driven from sbatch --export) ----
MODEL="${MODEL:?MODEL must be set (local snapshot dir or HF id)}"
TRACE_DIR="${MOE_TRACE_DIR:-/moe/vllm_traces}"
TP_DIR="${VLLM_TORCH_PROFILER_DIR:-$TRACE_DIR/torchprof}"
SERVER_LOG="${SERVER_LOG:-/moe/logs/vllm_server.0.log}"
PORT="${PORT:-8000}"

# DP/EP: global DP size = total GPUs across the rack. Local = 4 (GB200 4/node).
DP="${DP:?DP (global data-parallel size) must be set}"
DP_LOCAL="${DP_LOCAL:-4}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
DP_ADDR="${DP_ADDR:-}"             # head node IP (required when NNODES>1)
DP_RPC_PORT="${DP_RPC_PORT:-13345}"

# all2all backend + dtype/quant. deepep_low_latency is the decode recipe.
ALL2ALL_BACKEND="${ALL2ALL_BACKEND:-flashinfer_nvlink_one_sided}"
VLLM_QUANT="${VLLM_QUANT:-modelopt_fp4}"   # NVFP4 default; "" for BF16, "fp8" etc.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-9600}"
EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
# MoE runner backend. The image AUTO-selects FLASHINFER_CUTEDSL_BATCHED for NVFP4,
# but that backend's JIT codegen throws a C++ `terminate` in this nightly image —
# "nanobind::builtin_exception: Expected an MLIR object (got ...OpResultList)" — a
# FlashInfer-CUTEDSL ↔ cutlass-MLIR Python-binding version mismatch that hard-kills
# the worker during the first forward (jobs 3257426/3258927 RC; DeepEP itself is
# fine — a standalone deep_ep dispatch+combine probe passed). Force a non-CUTEDSL
# NVFP4 MoE backend. flashinfer_trtllm is the latency-oriented choice; override via
# MOE_BACKEND=flashinfer_cutlass / cutlass if trtllm has its own shape gap.
MOE_BACKEND="${MOE_BACKEND:-flashinfer_cutlass}"

# Profiler schedule.
TP_DELAY="${VLLM_PROFILE_DELAY:-3}"        # skip warmup iters (~sglang start_step=3)
TP_ACTIVE="${VLLM_PROFILE_ACTIVE:-8}"      # active decode iters to capture

mkdir -p "$TRACE_DIR" "$TP_DIR" "$(dirname "$SERVER_LOG")"

# ---- JIT caches to node-local disk (NFS FileLock deadlock otherwise) ----
# Same lesson as sglang: flashinfer/triton/inductor JIT under CUDA-graph capture
# contend a single NFS FileLock across ranks -> Errno 116 -> watchdog kill.
_JITCACHE="${FLASHINFER_WORKSPACE_BASE:-/tmp/jitcache-$USER}"
export FLASHINFER_WORKSPACE_BASE="$_JITCACHE"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$_JITCACHE/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$_JITCACHE/inductor}"
mkdir -p "$_JITCACHE/.cache/flashinfer" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" 2>/dev/null || true
echo "  JIT cache base (node-local): $_JITCACHE"

# ---- HF offline (local snapshot path dodges the multi-rank download filelock) ----
export HF_HOME="${HF_HOME:-/moe/hfcache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# ---- DeepEP / MNNVL transport (GB200 NVLink-fabric, single rack) ----
# vLLM tuning knobs (vllm/envs.py). The MNNVL low-latency dispatch rides the
# NVLink fabric within one clique — the validated single-rack path.
export VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL="${VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL:-1}"
# VLLM_DEEPEPLL_NVFP4_DISPATCH=1 makes vLLM pass `use_nvfp4=True` to
# deep_ep.Buffer.low_latency_dispatch(), but the DeepEP baked into the published
# vllm-openai:nightly-aarch64 image PREDATES that kwarg -> "low_latency_dispatch()
# got an unexpected keyword argument 'use_nvfp4'" and all workers die (job 3258492
# RC). Default it OFF: dispatch then carries fp8/bf16 activations, which is exactly
# what the sglang baseline measured (runbook §7.1, "dispatch 搬 fp8 激活"). Set to 1
# only on an image whose DeepEP commit supports the NVFP4-in-dispatch path.
export VLLM_DEEPEPLL_NVFP4_DISPATCH="${VLLM_DEEPEPLL_NVFP4_DISPATCH:-0}"
# CROSS-NODE DeepEP FIX: the image's stock DeepEP (73b6ea4, allow_mnnvl, no
# use_fabric) fails cross-node at deep_ep.cpp runtime.sync 'invalid resource
# handle'. We overlay a hybrid-ep DeepEP build (d28bd67, has use_fabric ->
# CU_MEM_HANDLE_TYPE_FABRIC) on PYTHONPATH and tell the patched vLLM all2all.py
# to pass use_fabric=True via VLLM_DEEPEP_USE_FABRIC=1. Both the overlay dir and
# the patched all2all.py are bind-mounted by the launcher. Enable with
# VLLM_DEEPEP_USE_FABRIC=1 (default off so non-fabric runs are unaffected).
if [[ "${VLLM_DEEPEP_USE_FABRIC:-0}" == "1" ]]; then
  export PYTHONPATH="/moe/deepep_overlay:${PYTHONPATH:-}"
  echo "  DeepEP use_fabric overlay ON: PYTHONPATH=$PYTHONPATH"
  python3 -c "import inspect,deep_ep; print('  deep_ep use_fabric available:', 'use_fabric' in inspect.signature(deep_ep.Buffer.__init__).parameters)" 2>&1 | head -1
fi
# enroot IMEX/fabric unlock + MNNVL for NCCL (framework-agnostic; from sbatch.sh).
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# DeepEP low_latency uses NVSHMEM even on a SINGLE node, and NVSHMEM defaults to
# trying IBGDA over InfiniBand — which has no device on this NVLink-only GB200
# site, so `nvshmemi_transport_init:282 init failed for transport: IBGDA` kills
# all ranks (jobs 3258586 EP4 / 3257426 EP8 RC). Disable IBGDA and let NVSHMEM
# use the NVLink/native path. Applies to ALL node counts (the sglang sbatch.sh
# lesson). Override VLLM_DEEPEP_FABRIC_ENV=0 to skip.
if [[ "${VLLM_DEEPEP_FABRIC_ENV:-1}" == "1" ]]; then
  export NVSHMEM_IB_ENABLE_IBGDA="${NVSHMEM_IB_ENABLE_IBGDA:-0}"
  export NVSHMEM_IBGDA_ENABLE="${NVSHMEM_IBGDA_ENABLE:-0}"
  export NVSHMEM_DISABLE_CUDA_VMM="${NVSHMEM_DISABLE_CUDA_VMM:-0}"
  if [[ "${NNODES:-1}" -gt 1 ]]; then
    # Cross-node: IMEX-backed FABRIC cuMem heap for the MNNVL symmetric heap
    # (cudaIpc handles don't cross nodes — the EP8 'invalid resource handle' at
    # deep_ep buffer.py:133 runtime.sync). Cross-node LL still needs a remote
    # transport over the NVLink fabric.
    export NVSHMEM_CUMEM_HANDLE_TYPE="${NVSHMEM_CUMEM_HANDLE_TYPE:-FABRIC}"
    export MC_FORCE_MNNVL="${MC_FORCE_MNNVL:-1}"
  else
    # SINGLE node: pure NVLink, NO network. DeepEP LL still allocates RDMA
    # buffers and NVSHMEM tries to bring up the IB/IBGDA transport, which has no
    # device here -> "init failed for transport: IBGDA" / "NET/IB: Ip address ::
    # invalid" -> fatal at the first forward (jobs 3258586/3258859 RC).
    # NVSHMEM_REMOTE_TRANSPORT=none disables network transport entirely so it
    # uses NVLink P2P only.
    export NVSHMEM_REMOTE_TRANSPORT="${NVSHMEM_REMOTE_TRANSPORT:-none}"
  fi
  [[ -z "${NVSHMEM_DEBUG_OFF:-}" ]] && { export NVSHMEM_DEBUG=INFO; export NVSHMEM_DEBUG_SUBSYS=TRANSPORT,INIT; }
  echo "  NVSHMEM env: IBGDA off, VMM on, remote_transport=${NVSHMEM_REMOTE_TRANSPORT:-fabric/mnnvl}"
fi

# Pick the routable iface (default-route owner), pin NCCL+gloo to it.
PICKED_IFACE="${IFACE:-$(awk '$2=="00000000"{print $1; exit}' /proc/net/route 2>/dev/null)}"
[[ -n "$PICKED_IFACE" ]] && { export NCCL_SOCKET_IFNAME="$PICKED_IFACE"; export GLOO_SOCKET_IFNAME="$PICKED_IFACE"; }
echo "  iface=$PICKED_IFACE  NNODES=$NNODES NODE_RANK=$NODE_RANK DP=$DP(local=$DP_LOCAL) addr=${DP_ADDR:-localhost}:$DP_RPC_PORT"

# ---- assemble the vllm serve command ----
ARGS=(
  serve "$MODEL"
  --enable-expert-parallel
  --data-parallel-size "$DP"
  --data-parallel-size-local "$DP_LOCAL"
  --all2all-backend "$ALL2ALL_BACKEND"
  --max-model-len "$MAX_MODEL_LEN"
  --trust-remote-code
  --host 0.0.0.0
  --port "$PORT"
)
[[ -n "$VLLM_QUANT" ]] && ARGS+=(--quantization "$VLLM_QUANT")
[[ -n "$MOE_BACKEND" && "$MOE_BACKEND" != "auto" ]] && ARGS+=(--moe-backend "$MOE_BACKEND")

# Multi-node DP wiring.
IS_HEADLESS=0
if [[ "$NNODES" -gt 1 ]]; then
  [[ -z "$DP_ADDR" ]] && { echo "ERROR: NNODES>1 needs DP_ADDR=<head ip>" >&2; exit 1; }
  ARGS+=(--data-parallel-address "$DP_ADDR" --data-parallel-rpc-port "$DP_RPC_PORT")
  if [[ "$NODE_RANK" -gt 0 ]]; then
    ARGS+=(--headless --data-parallel-start-rank "$(( NODE_RANK * DP_LOCAL ))")
    IS_HEADLESS=1
  fi
fi
# --api-server-count only on a NON-headless node. Default is data_parallel_size_
# local (4/8), spinning up that many API-server procs that each heavy-init + share
# the node; one then fails to bind its ZMQ addr within 600s under the 429GB load
# (job 3258373 RC). One API server suffices for the profiling driver. But
# --headless workers REJECT --api-server-count (job 3264049 RC: "cannot be used
# with --headless") — so add it only when NOT headless.
[[ "$IS_HEADLESS" == "0" ]] && ARGS+=(--api-server-count 1)

# Profiler config (head node only — it owns the API server + /start_profile).
# Workers inherit the profiler via the engine; their worker-side traces still
# land in torch_profiler_dir. We set it on every rank so the dir exists.
PROFILER_JSON="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$TP_DIR\",\"delay_iterations\":$TP_DELAY,\"active_iterations\":$TP_ACTIVE,\"torch_profiler_with_stack\":false}"
if [[ "${VLLM_PROFILE:-1}" == "1" ]]; then
  ARGS+=(--profiler-config "$PROFILER_JSON")
fi

# word-split EXTRA_ARGS intentionally
# shellcheck disable=SC2206
[[ -n "$EXTRA_ARGS" ]] && ARGS+=($EXTRA_ARGS)

echo "=== vllm ${ARGS[*]} ==="
vllm "${ARGS[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "vllm launched (PID=$SERVER_PID), log: $SERVER_LOG"

cleanup() {
  echo "shutting down vllm (PID $SERVER_PID)..."
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---- worker nodes just wait; only the head drives traffic + profiling ----
if [[ "$NODE_RANK" != "0" ]]; then
  echo "NODE_RANK=$NODE_RANK is a headless worker; waiting on server PID $SERVER_PID"
  wait "$SERVER_PID" || true
  exit 0
fi

# ---- wait for /health (head) ----
echo "waiting for /health on :$PORT (up to 40min for weight load + EP init)..."
READY=0
for _ in $(seq 1 480); do
  if curl -s --fail "http://localhost:$PORT/health" >/dev/null 2>&1; then READY=1; break; fi
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "ERROR: server process died early"; break; }
  sleep 5
done
if [[ "$READY" != "1" ]]; then
  echo "ERROR: server not ready; tail of log:"; tail -100 "$SERVER_LOG"; exit 1
fi
echo "server ready"
# Resolve the served model id (for a local snapshot path, vLLM serves it under
# the full path unless --served-model-name was set). The /v1/completions "model"
# field MUST match this id or the request 404s.
MODELS_JSON="$(curl -s "http://localhost:$PORT/v1/models" 2>/dev/null)"
echo "$MODELS_JSON" | head -c 400; echo
SERVED_MODEL="$(echo "$MODELS_JSON" | python3 -c 'import sys,json
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: pass' 2>/dev/null)"
[[ -z "$SERVED_MODEL" ]] && SERVED_MODEL="$MODEL"
echo "  served model id: $SERVED_MODEL"

# Build a batched prompt list of size BS (varied so routing isn't degenerate).
_mk_prompts() { # $1=bs -> JSON list of strings
  python3 - "$1" <<'PYEOF'
import json,sys
n=int(sys.argv[1])
print(json.dumps([f"Write a detailed technical essay number {i} about distributed GPU systems, MoE routing, and all-to-all communication across many ranks." for i in range(n)]))
PYEOF
}

# Output length must exceed delay+active so the decode loop crosses the profiler
# schedule window.
OUTLEN=$(( TP_DELAY + TP_ACTIVE + 8 ))

# ---- DECODE profiling, BATCH-SIZE SWEEP in ONE server launch ----
# vLLM's profiler output dir is FIXED at launch (--profiler-config), and
# /start_profile takes NO body — so to sweep bs without relaunching (each launch
# = ~18min weight-load+capture), we profile one bs, then MOVE its *.pt.trace.json*
# files into a per-bs subdir before the next bs. Accept comma/dash/space lists
# (sbatch --export eats commas, so the queue passes a dash list 1-2-8-32-64-128).
BS_LIST="$(echo "${VLLM_PROFILE_BS:-8}" | tr ',-' '  ')"
echo "=================== vllm DECODE bs-sweep: [$BS_LIST] (out=$OUTLEN) ==================="
for BS in $BS_LIST; do
  PROMPTS=$(_mk_prompts "$BS")
  echo "--- bs=$BS ---"
  # snapshot existing traces so we only move the NEW ones produced for this bs
  _before="$(find "$TP_DIR" -maxdepth 1 -name '*.pt.trace.json*' 2>/dev/null)"
  curl -s -X POST "http://localhost:$PORT/start_profile" -w "[profile bs=$BS] start http=%{http_code}\n" || echo "[profile bs=$BS] start FAILED"
  timeout 600 curl -s "http://localhost:$PORT/v1/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"$SERVED_MODEL\",\"prompt\":$PROMPTS,\"max_tokens\":$OUTLEN,\"temperature\":0}" \
    -o /dev/null -w "[profile bs=$BS] decode http=%{http_code} time=%{time_total}s\n" \
    || echo "[profile bs=$BS] decode FAILED/timeout"
  curl -s -X POST "http://localhost:$PORT/stop_profile" -w "[profile bs=$BS] stop http=%{http_code}\n" || echo "[profile bs=$BS] stop FAILED"
  sleep 25   # profiler flush
  # move the newly-written traces into a per-bs subdir
  mkdir -p "$TP_DIR/bs$BS"
  for f in $(find "$TP_DIR" -maxdepth 1 -name '*.pt.trace.json*' 2>/dev/null); do
    grep -qxF "$f" <<<"$_before" || mv "$f" "$TP_DIR/bs$BS/" 2>/dev/null || true
  done
  echo "[profile bs=$BS] traces -> $TP_DIR/bs$BS: $(find "$TP_DIR/bs$BS" -name '*.pt.trace.json*' | wc -l) files"
done

# ---- PREFILL profiling (optional second pass; long ISL via input_ids list) ----
# Sending `prompt` as a list of token ids forces an exact-length prefill — the
# only way to push expert GEMMs into the compute-bound region (runbook §3, §2.5).
if [[ -n "${VLLM_LONG_ISL:-}" ]]; then
  for ISL in $(echo "$VLLM_LONG_ISL" | tr ',-' '  '); do
    echo "=================== vllm long-ISL PREFILL: ISL=$ISL ==================="
    IDS=$(python3 - "$ISL" <<'PYEOF'
import json,sys
n=int(sys.argv[1])
print(json.dumps([(i*131 + 7) % 120000 + 100 for i in range(n)]))
PYEOF
)
    curl -s -X POST "http://localhost:$PORT/start_profile" -w "[isl $ISL] start http=%{http_code}\n" || true
    timeout 600 curl -s "http://localhost:$PORT/v1/completions" -H "Content-Type: application/json" \
      -d "{\"model\":\"$SERVED_MODEL\",\"prompt\":$IDS,\"max_tokens\":2,\"temperature\":0}" \
      -o /dev/null -w "[isl $ISL] generate http=%{http_code} time=%{time_total}s\n" || true
    curl -s -X POST "http://localhost:$PORT/stop_profile" -w "[isl $ISL] stop http=%{http_code}\n" || true
    sleep 25
  done
fi

# ---- summarize traces ----
echo ""
echo "--- torch profiler traces ($TP_DIR) ---"
find "$TP_DIR" -name '*.pt.trace.json*' 2>/dev/null | sort | head -40
N_TRACES=$(find "$TP_DIR" -name '*.pt.trace.json*' 2>/dev/null | wc -l)
echo "total trace files: $N_TRACES"
if [[ "$N_TRACES" -eq 0 ]]; then
  echo "WARN: no traces produced — tail of server log:"; tail -60 "$SERVER_LOG"
fi
