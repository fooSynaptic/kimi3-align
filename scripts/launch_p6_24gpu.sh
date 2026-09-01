#!/usr/bin/env bash
# P6 approx MOPD launch · 24 GPUs (no λ / no Effort)
# Requires HEAD_SSH, HEAD_HOST, WORKER_SPECS (see scripts/README.md)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
k3_require_sao
k3_mkdirs


LAUNCH=$K3/scripts/launch_e1_rl_20gpu.sh

MODE=${1:-smoke} # smoke | full
shift || true

case "$MODE" in
  smoke)
    CONFIG=${CONFIG:-$K3/configs/smoke/p6_mopd_p1teacher_24gpu_smoke.yaml}
    LOG=${LOG:-$K3/logs/p6_smoke_$(date +%Y%m%d_%H%M%S).log}
    AUDIT=${K3_MOPD_AUDIT_PATH:-$K3/experiments/checkpoints/root/k3-align-math-rl/p6-mopd-p1teacher-24gpu-smoke/mopd_audit.jsonl}
    : "${K3_MOPD_AUDIT_EVERY:=1}"
    ;;
  full)
    CONFIG=${CONFIG:-$K3/configs/p6_mopd_p1teacher_24gpu.yaml}
    LOG=${LOG:-$K3/logs/p6_full_$(date +%Y%m%d_%H%M%S).log}
    AUDIT=${K3_MOPD_AUDIT_PATH:-$K3/experiments/checkpoints/root/k3-align-math-rl/p6-mopd-p1teacher-24gpu-v4/mopd_audit.jsonl}
    : "${K3_MOPD_AUDIT_EVERY:=8}"
    ;;
  *)
    echo "usage: $0 {smoke|full}" >&2
    exit 2
    ;;
esac

export HEAD_GPUS=${HEAD_GPUS:-8}
export HEAD_CVD=${HEAD_CVD:-0,1,2,3,4,5,6,7}
export K25_TAU_LOG_RATIO_SQ="${K25_TAU_LOG_RATIO_SQ:-0.01}"
export K3_LAMBDA_ENABLED="${K3_LAMBDA_ENABLED:-0}"
export K3_EFFORT_ENABLED="${K3_EFFORT_ENABLED:-0}"
export K3_MOPD_ENABLED="${K3_MOPD_ENABLED:-1}"
export K3_MOPD_RMAX="${K3_MOPD_RMAX:-2.0}"
export K3_MOPD_ALPHA="${K3_MOPD_ALPHA:-0.5}"
export K3_MOPD_LEN_NORM="${K3_MOPD_LEN_NORM:-1}"
export K3_MOPD_ADV_CLIP="${K3_MOPD_ADV_CLIP:-2.0}"
export K3_MOPD_POS_ONLY="${K3_MOPD_POS_ONLY:-1}"
export K3_MOPD_AUDIT_EVERY="${K3_MOPD_AUDIT_EVERY}"
export K3_MOPD_AUDIT_PATH="$AUDIT"
export RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-8589934592}"
export RAY_MEMORY_USAGE_THRESHOLD="${RAY_MEMORY_USAGE_THRESHOLD:-0.995}"
export RAY_STOP_GRACE="${RAY_STOP_GRACE:-20}"

mkdir -p "$(dirname "$AUDIT")" "$K3/logs"

echo "[p6] mode=$MODE config=$CONFIG"
echo "[p6] topology HEAD=$HEAD_SSH gpus=$HEAD_GPUS cvd=$HEAD_CVD workers=$WORKER_SPECS"
echo "[p6] mopd enabled=$K3_MOPD_ENABLED rmax=$K3_MOPD_RMAX alpha=$K3_MOPD_ALPHA len_norm=$K3_MOPD_LEN_NORM adv_clip=$K3_MOPD_ADV_CLIP pos_only=$K3_MOPD_POS_ONLY"
echo "[p6] audit=$K3_MOPD_AUDIT_PATH"

CONFIG="$CONFIG" LOG="$LOG" \
  HEAD_SSH="$HEAD_SSH" HEAD_HOST="$HEAD_HOST" \
  HEAD_GPUS="$HEAD_GPUS" HEAD_CVD="$HEAD_CVD" \
  WORKER_SPECS="$WORKER_SPECS" \
  K25_TAU_LOG_RATIO_SQ="$K25_TAU_LOG_RATIO_SQ" \
  K3_LAMBDA_ENABLED="0" \
  K3_EFFORT_ENABLED="0" \
  K3_MOPD_ENABLED="$K3_MOPD_ENABLED" \
  K3_MOPD_RMAX="$K3_MOPD_RMAX" \
  K3_MOPD_ALPHA="$K3_MOPD_ALPHA" \
  K3_MOPD_LEN_NORM="$K3_MOPD_LEN_NORM" \
  K3_MOPD_ADV_CLIP="$K3_MOPD_ADV_CLIP" \
  K3_MOPD_POS_ONLY="$K3_MOPD_POS_ONLY" \
  K3_MOPD_AUDIT_PATH="$K3_MOPD_AUDIT_PATH" \
  K3_MOPD_AUDIT_EVERY="$K3_MOPD_AUDIT_EVERY" \
  RAY_OBJECT_STORE_MEMORY="$RAY_OBJECT_STORE_MEMORY" \
  RAY_MEMORY_USAGE_THRESHOLD="$RAY_MEMORY_USAGE_THRESHOLD" \
  RAY_STOP_GRACE="$RAY_STOP_GRACE" \
  bash "$LAUNCH"
