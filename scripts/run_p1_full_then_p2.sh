#!/usr/bin/env bash
# Full P1 (200 steps) then P2. Uses 8GiB object store. No kill -9.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"

LAUNCH=$K3/scripts/launch_e1_rl_20gpu.sh
POLL=${POLL:-60}
EXPECTED_STEPS=${EXPECTED_STEPS:-200}
export K25_TAU_LOG_RATIO_SQ="${K25_TAU_LOG_RATIO_SQ:-0.01}"
export RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-8589934592}"
export RAY_STOP_GRACE="${RAY_STOP_GRACE:-20}"

# shellcheck source=/dev/null
source "$K3/scripts/lib/wait_train.sh"

run_one() {
  local phase=$1
  local config=$2
  local log=$K3/logs/${phase}_$(date +%Y%m%d_%H%M%S).log
  echo "[chain] start $phase config=$config"
  # wipe trial name_resolve before each phase (launch also wipes)
  local trial
  trial=$(awk '/^trial_name:/{print $2; exit}' "$config")
  rm -rf "$K3/tmp/name_resolve/k3-align-math-rl/${trial}" || true
  CONFIG="$config" LOG="$log" \
    K25_TAU_LOG_RATIO_SQ="$K25_TAU_LOG_RATIO_SQ" \
    RAY_OBJECT_STORE_MEMORY="$RAY_OBJECT_STORE_MEMORY" \
    RAY_STOP_GRACE="$RAY_STOP_GRACE" \
    bash "$LAUNCH" || return 1
  sleep 8
  wait_train "$phase" "$log" "$EXPECTED_STEPS" "$POLL" || return 1
  echo "$log" > "$K3/logs/${phase}_last_log.txt"
  return 0
}

mkdir -p "$K3/logs" "$K3/docs"
run_one p1_k25_full "$K3/configs/p1_k25_sync_h0_20gpu.yaml"
run_one p2_k25_async "$K3/configs/p2_k25_async_h4_20gpu.yaml"
echo "[chain] P1-full + P2 done" | tee "$K3/docs/p1_p2_chain_done.txt"
