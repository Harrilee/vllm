#!/usr/bin/env bash
# MoE trace smoke test for vLLM.
#
# Brings up a vllm OpenAI server with MOE_TRACE_ENABLE=1, runs a short
# aiperf burst against it, then prints where the per-rank JSONL traces
# landed.
#
# Target: GB300 NVL72, 4 GPUs (~288 GB HBM3e each).
# Model:  Qwen/Qwen3.5-397B-A17B.
# A2A:    deepep_high_throughput (research-plan row 13).
#
# Per repo AGENTS.md, this script uses uv (not bare pip) for venv +
# engine install.

set -euo pipefail

ENGINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ENGINE_DIR"

VENV_DIR="$ENGINE_DIR/.venv"
TRACE_DIR="${MOE_TRACE_DIR:-$ENGINE_DIR/moe_traces}"
SERVER_LOG="$ENGINE_DIR/vllm_server.log"
MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B}"
TP="${TP:-4}"
PORT="${PORT:-30000}"

# ---------------------------------------------------------------- 1) venv
if ! command -v uv >/dev/null 2>&1; then
  pip install --user uv
  export PATH="$HOME/.local/bin:$PATH"
fi
if [[ ! -d "$VENV_DIR" ]]; then
  uv venv --python 3.12 "$VENV_DIR"
  VLLM_USE_PRECOMPILED=1 uv pip install -e . \
    --python "$VENV_DIR/bin/python" \
    --torch-backend=auto
fi

# ----------------------------------------------------- 2) aiperf for testing
"$VENV_DIR/bin/pip" install --quiet aiperf

# ------------------------------------------------------------- 3) tracer env
mkdir -p "$TRACE_DIR"
rm -f "$TRACE_DIR"/moe_trace_rank*.jsonl
export MOE_TRACE_ENABLE=1
export MOE_TRACE_DIR="$TRACE_DIR"
export MOE_TRACE_FLUSH_EVERY=512

# Select the A2A backend the experiment matrix calls for.
export VLLM_ALL2ALL_BACKEND="${VLLM_ALL2ALL_BACKEND:-deepep_high_throughput}"

# --------------------------------------------------------- 4) launch server
"$VENV_DIR/bin/vllm" serve "$MODEL" \
  --tensor-parallel-size "$TP" \
  --enable-expert-parallel \
  --host 0.0.0.0 \
  --port "$PORT" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "vllm launched (PID=$SERVER_PID), log: $SERVER_LOG"

cleanup() {
  echo "shutting down vllm (PID $SERVER_PID)..."
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ----------------------------------------------------- 5) wait for readiness
echo "waiting for /health on :$PORT (up to 30 min for 397B load)..."
for _ in $(seq 1 360); do
  if curl -s --fail "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "server ready"
    break
  fi
  sleep 5
done
curl -s --fail "http://localhost:$PORT/health" >/dev/null || {
  echo "ERROR: server did not become ready; tail of log:"
  tail -80 "$SERVER_LOG"
  exit 1
}

# -------------------------------------------------------------- 6) aiperf
"$VENV_DIR/bin/aiperf" profile \
  --model "$MODEL" \
  --tokenizer "$MODEL" \
  --service-kind openai \
  --endpoint-type chat \
  --endpoint /v1/chat/completions \
  --url "http://localhost:$PORT" \
  --concurrency 4 \
  --warmup-request-count 2 \
  --request-count 20 \
  --synthetic-input-tokens-mean 1024 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 128 \
  --output-tokens-stddev 0

# --------------------------------------------------- 7) summarize trace output
echo ""
echo "--- trace files ---"
ls -lh "$TRACE_DIR"/moe_trace_rank*.jsonl 2>/dev/null || echo "no trace files produced"
echo ""
echo "--- event counts per kind, per rank ---"
for f in "$TRACE_DIR"/moe_trace_rank*.jsonl; do
  [[ -f "$f" ]] || continue
  echo "==> $(basename "$f")"
  "$VENV_DIR/bin/python" -c "
import json, sys, collections
c = collections.Counter()
for line in open('$f'):
    try: c[json.loads(line)['kind']] += 1
    except Exception: pass
for k, v in sorted(c.items()): print(f'    {k:12s} {v}')
"
done
echo ""
echo "--- first event of each kind (rank 0) ---"
"$VENV_DIR/bin/python" -c "
import json, glob
seen = set()
for line in open(sorted(glob.glob('$TRACE_DIR/moe_trace_rank0.jsonl'))[0]):
    try: e = json.loads(line)
    except Exception: continue
    if e['kind'] in seen: continue
    seen.add(e['kind'])
    print(json.dumps(e, indent=2))
" 2>/dev/null || true
