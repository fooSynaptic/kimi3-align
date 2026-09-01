#!/usr/bin/env bash
# P4 λ-barrier launch wrapper · 16 GPUs
# Requires HEAD_SSH, HEAD_HOST, WORKER_SPECS
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
k3_require_sao
k3_mkdirs


LAUNCH=$K3/scripts/launch_e1_rl_20gpu.sh

MODE=${1:-smoke} # smoke | full | custom
shift || true

case "$MODE" in
  smoke)
    CONFIG=${CONFIG:-$K3/configs/smoke/p4_h4_lambda05_16gpu_smoke.yaml}
    LOG=${LOG:-$K3/logs/p4_smoke_$(date +%Y%m%d_%H%M%S).log}
    AUDIT=${K3_LAMBDA_AUDIT_PATH:-$K3/experiments/checkpoints/root/k3-align-math-rl/p4-h4-lambda05-16gpu-smoke/lambda_audit.jsonl}
    ;;
  full)
    CONFIG=${CONFIG:-$K3/configs/p4_h4_lambda05_16gpu.yaml}
    LOG=${LOG:-$K3/logs/p4_full_$(date +%Y%m%d_%H%M%S).log}
    AUDIT=${K3_LAMBDA_AUDIT_PATH:-$K3/experiments/checkpoints/root/k3-align-math-rl/p4-h4-lambda05-16gpu/lambda_audit.jsonl}
    ;;
  *)
    echo "usage: $0 {smoke|full}" >&2
    exit 2
    ;;
esac

export HEAD_GPUS=${HEAD_GPUS:-8}
export HEAD_CVD=${HEAD_CVD:-0,1,2,3,4,5,6,7}
export K25_TAU_LOG_RATIO_SQ="${K25_TAU_LOG_RATIO_SQ:-0.01}"
export K3_LAMBDA_ENABLED="${K3_LAMBDA_ENABLED:-1}"
export K3_LAMBDA="${K3_LAMBDA:-0.5}"
export K3_LAMBDA_N="${K3_LAMBDA_N:-168}"
export K3_LAMBDA_K="${K3_LAMBDA_K:-8}"
export K3_LAMBDA_DP="${K3_LAMBDA_DP:-12}"
export K3_LAMBDA_AUDIT_PATH="$AUDIT"
export RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-8589934592}"
export RAY_STOP_GRACE="${RAY_STOP_GRACE:-20}"

mkdir -p "$(dirname "$AUDIT")" "$K3/logs"

echo "[p4] mode=$MODE config=$CONFIG"
echo "[p4] topology HEAD=$HEAD_SSH gpus=$HEAD_GPUS cvd=$HEAD_CVD workers=$WORKER_SPECS"
echo "[p4] lambda enabled=$K3_LAMBDA_ENABLED λ=$K3_LAMBDA N=$K3_LAMBDA_N K=$K3_LAMBDA_K"
echo "[p4] audit=$K3_LAMBDA_AUDIT_PATH"

# launch_e1 must forward K3_LAMBDA_* into the train process env
CONFIG="$CONFIG" LOG="$LOG" \
  HEAD_SSH="$HEAD_SSH" HEAD_HOST="$HEAD_HOST" \
  HEAD_GPUS="$HEAD_GPUS" HEAD_CVD="$HEAD_CVD" \
  WORKER_SPECS="$WORKER_SPECS" \
  K25_TAU_LOG_RATIO_SQ="$K25_TAU_LOG_RATIO_SQ" \
  K3_LAMBDA_ENABLED="$K3_LAMBDA_ENABLED" \
  K3_LAMBDA="$K3_LAMBDA" \
  K3_LAMBDA_N="$K3_LAMBDA_N" \
  K3_LAMBDA_K="$K3_LAMBDA_K" \
  K3_LAMBDA_DP="${K3_LAMBDA_DP:-12}" \
  K3_LAMBDA_AUDIT_PATH="$K3_LAMBDA_AUDIT_PATH" \
  RAY_OBJECT_STORE_MEMORY="$RAY_OBJECT_STORE_MEMORY" \
  RAY_STOP_GRACE="$RAY_STOP_GRACE" \
  bash "$LAUNCH"
