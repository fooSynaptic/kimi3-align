#!/usr/bin/env bash
# Launch P2 only (K2.5-style async h4). Assumes P1 done.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"

export K25_TAU_LOG_RATIO_SQ="${K25_TAU_LOG_RATIO_SQ:-0.01}"
# Prefer smaller object store after P1 OOM on worker node
export RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-8589934592}"
# GPUs cleaned; use default topology unless SKIP_DIRTY_HEAD=1
if [[ "${SKIP_DIRTY_HEAD:-0}" == "1" ]]; then
  export HEAD_SSH=${HEAD_SSH:?set HEAD_SSH}
  export HEAD_HOST=${HEAD_HOST:?set HEAD_HOST}
  export HEAD_GPUS=8
  export HEAD_CVD=0,1,2,3,4,5,6,7
  export WORKER_SPECS=${WORKER_SPECS:?set WORKER_SPECS}
fi
LOG=${LOG:-$K3/logs/p2_k25_async_$(date +%Y%m%d_%H%M%S).log}
CONFIG=$K3/configs/p2_k25_async_h4_20gpu.yaml
echo "[p2] log=$LOG object_store=$RAY_OBJECT_STORE_MEMORY"
CONFIG="$CONFIG" LOG="$LOG" bash "$K3/scripts/launch_e1_rl_20gpu.sh"
echo "[p2] started — tail -f $LOG"
