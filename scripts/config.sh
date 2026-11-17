#!/usr/bin/env bash
# InkSpark: Inkling 2 Small on two DGX Sparks via TensorFold. Shared settings; override any in scripts/local.sh
# (gitignored), which holds the node addresses.
# shellcheck disable=SC2034

# --- nodes (set in scripts/local.sh) -------------------------------------------------------------------------------
WORKER="${WORKER:-}"                          # rank 1, user@address on the link between the Sparks
MASTER="${MASTER:-}"                          # rank 0's address on that link (rendezvous + NCCL)
MASTER_PORT="${MASTER_PORT:-29591}"

# --- checkpoints ---------------------------------------------------------------------------------------------------
SRC_REPO="thinkingmachines/Inkling-2-Small"   # NVFP4, the native precision
SRC_REV="aac974122acd33ddb0d3e1f54433a9c5dd2249ce"
PACK_REPO="ajensenwaud/inkspark-2dgx"
PACK_REV="${PACK_REV:-aa29f3a011b011896826416f03b8ac55d7ec7c6f}"   # the published pack, 18 Nov 2026
MODELS_ROOT="${MODELS_ROOT:-$HOME/models/inkspark}"
SRC_DIR="${SRC_DIR:-$MODELS_ROOT/Inkling-2-Small}"
MODEL_DIR="${MODEL_DIR:-$MODELS_ROOT/inkspark-2dgx}"   # the same path on both nodes
SERVED_NAME="${SERVED_NAME:-Inkling-2-Small}"

# --- engine --------------------------------------------------------------------------------------------------------
TF_REPO="https://github.com/ashhart/TensorFold"
TF_TAG="v1.2.7"
TF_COMMIT="${TF_COMMIT:-93a2e93ecebff8aa52b88e58c5dd222516e69c5f}"   # v1.2.7
ZIG_TREE="5c4b3f401340846a13651cd41d6cb1e617450342"   # v1.2.7 + patches-zig/ (git tree hash; prepare.sh checks it)
TF_DIR="${TF_DIR:-$HOME/inkspark/tensorfold}"   # the patched tree and its build (zig-out/), the same path on both nodes
EXL3_REPO="https://github.com/turboderp-org/exllamav3"
EXL3_TAG="v1.6.6"

# --- image (docker/Dockerfile.build: builds the engine and runs it) -----------------------------------------------
# CUDA 13.3 base: Mia's GLM-5.3 TensorFold image (NVIDIA PyTorch 26.07), pinned by digest, plus Zig 0.17.0.
BASE_IMAGE="${BASE_IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:14f15591eae5d6a540f09218d3852068962fe5381371bbfefe0e9194cd834529}"
IMAGE="${IMAGE:-inkspark-zig:0.17.0}"
CONTAINER="${CONTAINER:-inkspark}"            # inkspark-r0 on the head, inkspark-r1 on the worker

# --- serving -------------------------------------------------------------------------------------------------------
PORT="${PORT:-8888}"
HOST="${HOST:-0.0.0.0}"                       # the API's listen address; 127.0.0.1 keeps it on the head
SHAPE="${SHAPE:-long}"                        # long: 1 stream at 1M; agents: 8 streams at 128K
case "$SHAPE" in
  long)   PARALLEL="${PARALLEL:-1}"; CONTEXT="${CONTEXT:-1048576}" ;;
  agents) PARALLEL="${PARALLEL:-8}"; CONTEXT="${CONTEXT:-131072}" ;;
esac
DRAFTER="${DRAFTER:-1}"                       # 1: the checkpoint's DSpark drafter
DRAFT_WEIGHTS="${DRAFT_WEIGHTS:-nvfp4}"        # the drafter's projections: nvfp4 (G7: same acceptance, -32 % pass) or bf16
KV="${KV:-fp8}"                               # the global layers' KV cache: fp8 (e4m3, KV8: same GSM8K, 1M decode 1.4x) or bf16
RESERVE_GIB="${RESERVE_GIB:-3}"               # memory each rank leaves free; the rest after weights and streams is prompt cache
MIN_FREE_GIB="${MIN_FREE_GIB:-100}"           # start.sh refuses a node with less memory available than this
GPU_LOCK="${GPU_LOCK:-inkspark/gpu.lock}"     # under $HOME on each node; start.sh writes "serve inkspark" there

# --- fabric (the direct link between the Sparks) ---------------------------------------------------------------------
NCCL_IFNAME="${NCCL_IFNAME:-enp1s0f0np0}"
NCCL_HCA="${NCCL_HCA:-rocep1s0f0}"

[[ -f "$(dirname "${BASH_SOURCE[0]}")/local.sh" ]] && source "$(dirname "${BASH_SOURCE[0]}")/local.sh"

log() { printf '[%s] %s\n' "$(basename "$0")" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }
need_pair() { [[ -n "$WORKER" && -n "$MASTER" ]] || die "set WORKER and MASTER in scripts/local.sh"; }
