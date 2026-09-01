#!/usr/bin/env bash
# P1 K2.5-style sync (h0) → success → P2 async (h4). 20 GPU. Teardown uses TERM so CUDA contexts release cleanly.
# SKIP_DIRTY_HEAD=1: when head GPUs hold zombie VRAM, Ray head moves to a clean node and only clean GPU indices are exposed on the dirty host.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"

LAUNCH=$K3/scripts/launch_e1_rl_20gpu.sh
STEPS=${STEPS:-200}
POLL=${POLL:-60}
export K25_TAU_LOG_RATIO_SQ="${K25_TAU_LOG_RATIO_SQ:-0.01}"

# SKIP_DIRTY_HEAD=1: Ray head on a clean node; dirty host contributes only GPUs that pass preflight VRAM check.
if [[ "${SKIP_DIRTY_HEAD:-0}" == "1" ]]; then
  export HEAD_SSH=${HEAD_SSH:?set HEAD_SSH}
  export HEAD_HOST=${HEAD_HOST:?set HEAD_HOST}
  export HEAD_GPUS=8
  export HEAD_CVD=0,1,2,3,4,5,6,7
  export WORKER_SPECS=${WORKER_SPECS:?set WORKER_SPECS}
  echo "[chain] SKIP_DIRTY_HEAD=1 → head=$HEAD_SSH ($HEAD_HOST) workers=$WORKER_SPECS"
fi

wait_train() {
  local label=$1
  local log=$2
  local pidfile=$K3/logs/e1_rl.pid
  echo "[chain] wait $label log=$log tau=$K25_TAU_LOG_RATIO_SQ"
  while true; do
    if [[ -f "$pidfile" ]]; then
      local pid
      pid=$(cat "$pidfile")
      if ! ps -p "$pid" >/dev/null 2>&1; then
        if grep -q "Training complete" "$log" 2>/dev/null; then
          echo "[chain] $label SUCCESS (Training complete)"
          return 0
        fi
        echo "[chain] $label FAIL (driver exited without Training complete)" >&2
        tail -80 "$log" >&2 || true
        return 1
      fi
    fi
    sleep "$POLL"
  done
}

run_one() {
  local phase=$1
  local config=$2
  local log=$K3/logs/${phase}_$(date +%Y%m%d_%H%M%S).log
  echo "[chain] start $phase config=$config"
  CONFIG="$config" LOG="$log" K25_TAU_LOG_RATIO_SQ="$K25_TAU_LOG_RATIO_SQ" \
    HEAD_SSH="${HEAD_SSH:-}" HEAD_HOST="${HEAD_HOST:-}" HEAD_GPUS="${HEAD_GPUS:-}" \
    HEAD_CVD="${HEAD_CVD:-}" WORKER_SPECS="${WORKER_SPECS:-}" \
    bash "$LAUNCH"
  sleep 5
  wait_train "$phase" "$log"
  echo "$log" > "$K3/logs/${phase}_last_log.txt"
}

mkdir -p "$K3/logs" "$K3/docs"
run_one p1_k25_sync "$K3/configs/p1_k25_sync_h0_20gpu.yaml"
run_one p2_k25_async "$K3/configs/p2_k25_async_h4_20gpu.yaml"
echo "[chain] P1+P2 K2.5-style done" | tee "$K3/docs/p1_p2_chain_done.txt"
