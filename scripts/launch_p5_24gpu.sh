#!/usr/bin/env bash
# P5 Effort launch wrapper · 24 GPUs (no λ)
# Requires HEAD_SSH, HEAD_HOST, WORKER_SPECS
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
k3_require_sao
k3_mkdirs


LAUNCH=$K3/scripts/launch_e1_rl_20gpu.sh

MODE=${1:-smoke} # smoke | stage_a | stage_b
shift || true

case "$MODE" in
  smoke)
    CONFIG=${CONFIG:-$K3/configs/smoke/p5_effort_tau2_24gpu_smoke.yaml}
    LOG=${LOG:-$K3/logs/p5_smoke_$(date +%Y%m%d_%H%M%S).log}
    AUDIT=${K3_EFFORT_AUDIT_PATH:-$K3/experiments/checkpoints/root/k3-align-math-rl/p5-effort-tau2-24gpu-smoke/effort_audit.jsonl}
    TAU=${K3_EFFORT_TAU:-2.0}
    AUDIT_EVERY=${K3_EFFORT_AUDIT_EVERY:-1}
    ;;
  stage_a)
    CONFIG=${CONFIG:-$K3/configs/p5_effort_tau2_24gpu.yaml}
    LOG=${LOG:-$K3/logs/p5_stage_a_$(date +%Y%m%d_%H%M%S).log}
    AUDIT=${K3_EFFORT_AUDIT_PATH:-$K3/experiments/checkpoints/root/k3-align-math-rl/p5-effort-tau2-24gpu/effort_audit.jsonl}
    TAU=${K3_EFFORT_TAU:-2.0}
    AUDIT_EVERY=${K3_EFFORT_AUDIT_EVERY:-8}
    ;;
  stage_b)
    CONFIG=${CONFIG:-$K3/configs/p5_effort_tau1_24gpu.yaml}
    LOG=${LOG:-$K3/logs/p5_stage_b_$(date +%Y%m%d_%H%M%S).log}
    AUDIT=${K3_EFFORT_AUDIT_PATH:-$K3/experiments/checkpoints/root/k3-align-math-rl/p5-effort-tau1-24gpu/effort_audit.jsonl}
    TAU=${K3_EFFORT_TAU:-1.0}
    AUDIT_EVERY=${K3_EFFORT_AUDIT_EVERY:-8}
    ;;
  *)
    echo "usage: $0 {smoke|stage_a|stage_b}" >&2
    exit 2
    ;;
esac

export HEAD_GPUS=${HEAD_GPUS:-8}
export HEAD_CVD=${HEAD_CVD:-0,1,2,3,4,5,6,7}
export K25_TAU_LOG_RATIO_SQ="${K25_TAU_LOG_RATIO_SQ:-0.01}"
export K3_LAMBDA_ENABLED="${K3_LAMBDA_ENABLED:-0}"
export K3_EFFORT_ENABLED="${K3_EFFORT_ENABLED:-1}"
export K3_EFFORT_B0="${K3_EFFORT_B0:-256}"
export K3_EFFORT_TAU="$TAU"
export K3_EFFORT_AUDIT_PATH="$AUDIT"
export K3_EFFORT_AUDIT_EVERY="$AUDIT_EVERY"
export RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-8589934592}"
export RAY_MEMORY_USAGE_THRESHOLD="${RAY_MEMORY_USAGE_THRESHOLD:-0.995}"
export RAY_STOP_GRACE="${RAY_STOP_GRACE:-20}"

mkdir -p "$(dirname "$AUDIT")" "$K3/logs"

echo "[p5] mode=$MODE config=$CONFIG"
echo "[p5] topology HEAD=$HEAD_SSH gpus=$HEAD_GPUS cvd=$HEAD_CVD workers=$WORKER_SPECS"
echo "[p5] effort enabled=$K3_EFFORT_ENABLED b0=$K3_EFFORT_B0 tau=$K3_EFFORT_TAU"
echo "[p5] audit=$K3_EFFORT_AUDIT_PATH every=$K3_EFFORT_AUDIT_EVERY"
echo "[p5] lambda disabled=$K3_LAMBDA_ENABLED"

CONFIG="$CONFIG" LOG="$LOG" \
  HEAD_SSH="$HEAD_SSH" HEAD_HOST="$HEAD_HOST" \
  HEAD_GPUS="$HEAD_GPUS" HEAD_CVD="$HEAD_CVD" \
  WORKER_SPECS="$WORKER_SPECS" \
  K25_TAU_LOG_RATIO_SQ="$K25_TAU_LOG_RATIO_SQ" \
  K3_LAMBDA_ENABLED="0" \
  K3_EFFORT_ENABLED="$K3_EFFORT_ENABLED" \
  K3_EFFORT_B0="$K3_EFFORT_B0" \
  K3_EFFORT_TAU="$K3_EFFORT_TAU" \
  K3_EFFORT_AUDIT_PATH="$K3_EFFORT_AUDIT_PATH" \
  K3_EFFORT_AUDIT_EVERY="$K3_EFFORT_AUDIT_EVERY" \
  RAY_OBJECT_STORE_MEMORY="$RAY_OBJECT_STORE_MEMORY" \
  RAY_MEMORY_USAGE_THRESHOLD="$RAY_MEMORY_USAGE_THRESHOLD" \
  RAY_STOP_GRACE="$RAY_STOP_GRACE" \
  bash "$LAUNCH"
