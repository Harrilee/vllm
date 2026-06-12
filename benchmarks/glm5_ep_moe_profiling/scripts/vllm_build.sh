#!/usr/bin/env bash
#SBATCH --job-name=vllm-glm5-import
#SBATCH --partition=batch
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --cpus-per-task=144
#SBATCH --time=00:40:00
#SBATCH --output=logs/vllm_build_%j.out
#SBATCH --error=logs/vllm_build_%j.err
#
# Import a prebuilt GB200 (aarch64) vLLM image to a SHARED squashfs that
# pyxis/enroot loop-mounts on every compute node.
#
# WHY IMPORT, NOT `docker build`: oci-hsg compute nodes ship ONLY enroot 4.0.1
# (no docker/podman/buildah), and the user has no /etc/subuid range, so rootless
# docker cannot start (`No subuid ranges found` — confirmed job 3256910). This is
# the SAME reason the sglang images on this cluster were imported, never built
# here. The sbatch_build.sh docker-build flow was written for a different site.
#
# EFFICIENCY: DeepEP + NVSHMEM are BAKED INTO the published vllm-openai image as
# system wheels (vllm docker/Dockerfile L721, in the `vllm-base` layer that
# `vllm-openai` is FROM). So this import is once-and-done — no per-run DeepEP
# install. Subsequent EP jobs just loop-mount the .sqsh read-only.
#
# Submit:
#   sbatch vllm_build.sh                       # default tag (nightly-aarch64)
#   VLLM_TAG=v0.22.1-aarch64 sbatch vllm_build.sh

set -uo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"; cd "$REPO_ROOT"; mkdir -p logs

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "ERROR: not aarch64 ($(uname -m)); submit to the batch (GB200) partition" >&2
    exit 1
fi
command -v enroot >/dev/null 2>&1 || { echo "ERROR: enroot not on PATH on this node" >&2; exit 1; }

IMAGE_DIR="${IMAGE_DIR:-/home/harrli/sglang-moe-images}"
# nightly-aarch64 tracks main (matches the ~/vllm checkout's GlmMoeDsa support).
# Pin a stable release tag via VLLM_TAG= if you prefer reproducibility over
# freshest model support.
VLLM_TAG="${VLLM_TAG:-nightly-aarch64}"
# enroot wants registry-host '#'-separated from the path. CRITICAL: use
# registry-1.docker.io, NOT docker.io — the bare `docker.io` 302-redirects to
# www.docker.com (the marketing site), which enroot's `-SsL` follows; the site
# has no www-authenticate header so enroot logs "Permission granted", does an
# anonymous GET, and jq chokes on the 401 error body ("Could not process JSON
# input", jobs 3257048/3257164/3257182 RC). registry-1.docker.io is the real
# Docker Hub v2 endpoint and returns the auth challenge enroot needs.
URI="${VLLM_URI:-docker://registry-1.docker.io#vllm/vllm-openai:${VLLM_TAG}}"
SQSH="${SQSH:-$IMAGE_DIR/vllm-glm5-aarch64.sqsh}"
mkdir -p "$IMAGE_DIR"

echo "==================================================================="
echo "vLLM image import (aarch64 / GB200)"
echo "  job:   ${SLURM_JOB_ID:-local}   node: $(hostname)"
echo "  uri:   $URI"
echo "  sqsh:  $SQSH"
echo "==================================================================="
df -h "$IMAGE_DIR" /tmp 2>&1 | head -3; echo

# The cluster's /etc/enroot/enroot.conf already pins TEMP/CACHE/DATA to node-local
# /raid and caps mksquashfs (-processors 8 -mem 23G), so we DON'T override those
# (a bad override OOM'd sglang's import, job 3151713).
#
# PROXY: /etc/enroot/enroot.conf hardcodes https_proxy to the site ccache, which
# corrupts registry responses. Compute nodes have DIRECT internet to Docker Hub,
# so bypass the proxy for all registry/blob hosts via no_proxy (enroot honors a
# pre-set no_proxy and won't overwrite it). This is the reliable path here.
export no_proxy="${no_proxy:-registry-1.docker.io,auth.docker.io,docker.io,index.docker.io,production.cloudflare.docker.com,.docker.com,.docker.io,.cloudflare.docker.com}"
export NO_PROXY="$no_proxy"

rm -f "$SQSH.tmp"
echo "=== enroot import $URI -> $SQSH ==="
time enroot import -o "$SQSH.tmp" "$URI"
IMPORT_RC=$?
[[ $IMPORT_RC -ne 0 ]] && { echo "ERROR: enroot import failed rc=$IMPORT_RC" >&2; exit $IMPORT_RC; }
mv "$SQSH.tmp" "$SQSH"
ls -lh "$SQSH"

# ---- smoke: run the imported image, confirm vLLM + DeepEP + GLM-5 support ----
echo
echo "=== in-image smoke (vllm + deep_ep + GlmMoeDsa registry) ==="
srun --ntasks=1 --container-image="$SQSH" --container-workdir=/workspace \
    --no-container-mount-home --no-container-entrypoint \
    python3 -c '
import vllm
print("vllm", getattr(vllm, "__version__", "?"))
try:
    import deep_ep; print("deep_ep OK:", deep_ep.__file__)
except Exception as e:
    print("deep_ep IMPORT FAILED:", repr(e))
try:
    from vllm.model_executor.models.registry import ModelRegistry
    archs = ModelRegistry.get_supported_archs()
    print("GlmMoeDsa supported:", "GlmMoeDsaForCausalLM" in archs)
except Exception as e:
    print("registry check failed:", repr(e))
' 2>&1 | grep -vE '^(WARNING|INFO|\s*$)' | tail -20 || echo "WARN: in-image smoke had errors (see above)"

echo
echo "Done. Image squashfs: $SQSH"
echo "Run the EP8 milestone:"
echo "  PYXIS_IMAGE=$SQSH sbatch --nodes=2 --nodelist=<rack-Tx,rack-Ty> sbatch_vllm_glm5.sh"
