#!/usr/bin/env bash
# K3-align E1 MATH RL · 20 Hopper (96GB HBM): head vLLM + worker FSDP
# Requires: SAO_ROOT, MODEL_PATH, DATA_PATH, HEAD_SSH, HEAD_HOST, WORKER_SPECS
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
k3_require_train_paths
k3_mkdirs

REPO=${AREAL_REPO}
RAY_TMP=/dev/shm/r
CONFIG=${CONFIG:-$K3/configs/p1_k25_sync_h0_20gpu.yaml}
LOG=${LOG:-$K3/logs/e1_rl_$(date +%Y%m%d_%H%M%S).log}
SMOKE_STEPS=${SMOKE_STEPS:-}
RAY_PORT=${RAY_PORT:-6379}
HEAD_CVD=${HEAD_CVD:-0,1,2,3}
HEAD_GPUS=${HEAD_GPUS:-4}
k3_require_ray_topo
RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-8589934592}
RAY_MEMORY_USAGE_THRESHOLD=${RAY_MEMORY_USAGE_THRESHOLD:-0.98}
TRAIN_PY=$K3/scripts/e1_math_rl_train.py
export K25_TAU_LOG_RATIO_SQ="${K25_TAU_LOG_RATIO_SQ:-0.01}"

export PATH="$VENV/bin:/usr/local/bin:$PATH"
export PYTHONPATH="$K3_PY:$SITE:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HOME="$K3/tmp/home"
export TMPDIR="$K3/tmp"
export HF_HOME="${HF_HOME:-$SAO/tmp/hf-home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$SAO/tmp/hf-datasets}"
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export RAY_ADDRESS="${HEAD_HOST}:${RAY_PORT}"
export RAY_memory_usage_threshold="$RAY_MEMORY_USAGE_THRESHOLD"
export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 TP_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1 NCCL_NET=Socket
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export SWANLAB_MODE=disabled
export WANDB_MODE=disabled

mkdir -p "$K3/logs" "$K3/experiments" "$K3/tmp/home" "$K3/tmp/name_resolve"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[fatal] missing $VENV" >&2
  exit 1
fi

# Wipe stale NFS name_resolve for this config's experiment/trial (avoids NameEntryExistsError).
wipe_name_resolve() {
  local exp trial root
  exp=$(awk '/^experiment_name:/{print $2; exit}' "$CONFIG")
  trial=$(awk '/^trial_name:/{print $2; exit}' "$CONFIG")
  root="$K3/tmp/name_resolve/${exp}/${trial}"
  if [[ -n "$exp" && -n "$trial" && -d "$root" ]]; then
    echo "[e1] wipe name_resolve $root" | tee -a "$LOG"
    rm -rf "$root"
  else
    echo "[e1] name_resolve wipe skip (exp=$exp trial=$trial root=$root)" | tee -a "$LOG"
  fi
}

stop_ray() {
  local host=$1 grace=${RAY_STOP_GRACE:-20}
  echo "[ray] stop $host (grace=${grace}s)" | tee -a "$LOG"
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$host" \
    "export PATH=$VENV/bin:/usr/local/bin:\$PATH; timeout $((grace + 30)) ray stop --grace-period $grace >/dev/null 2>&1 || true" || true
}

start_ray_node() {
  local host=$1 num_gpus=$2 cvd=$3 is_head=$4
  echo "[ray] start $host gpus=$num_gpus cvd=$cvd head=$is_head" | tee -a "$LOG"
  ssh -o BatchMode=yes "$host" "bash -s" <<EOF
set -euo pipefail
export PATH="$VENV/bin:/usr/local/bin:\$PATH"
export PYTHONPATH="$SITE:$REPO"
export HOME="$K3/tmp/home"
export TMPDIR="$K3/tmp"
export RAY_TMPDIR=$RAY_TMP
export RAY_memory_usage_threshold=$RAY_MEMORY_USAGE_THRESHOLD
export MALLOC_ARENA_MAX=2
export CUDA_VISIBLE_DEVICES=$cvd
export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 TP_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1 NCCL_NET=Socket
mkdir -p $RAY_TMP
if [ "$is_head" = "1" ]; then
  ray start --head --node-ip-address=$HEAD_HOST --port=$RAY_PORT \
    --dashboard-host=0.0.0.0 --num-gpus=$num_gpus \
    --object-store-memory=$RAY_OBJECT_STORE_MEMORY \
    --temp-dir=$RAY_TMP --disable-usage-stats
else
  ray start --address='$HEAD_HOST:$RAY_PORT' --num-gpus=$num_gpus \
    --object-store-memory=$RAY_OBJECT_STORE_MEMORY \
    --temp-dir=$RAY_TMP --disable-usage-stats
fi
EOF
}

echo "[e1] log=$LOG config=$CONFIG" | tee "$LOG"
wipe_name_resolve

# collect all hosts for stop
ALL_HOSTS="$HEAD_SSH"
for spec in $WORKER_SPECS; do
  ALL_HOSTS="$ALL_HOSTS ${spec%%:*}"
done
for h in $ALL_HOSTS; do stop_ray "$h"; done
# drop leftover vLLM children on head (TERM only)
ssh -o BatchMode=yes "$HEAD_SSH" \
  'for p in $(ps -eo pid,cmd | awk "/VLLM::|areal_vllm_server/ && !/awk/ {print \$1}"); do kill -TERM $p 2>/dev/null || true; done' \
  || true
sleep 2

start_ray_node "$HEAD_SSH" "$HEAD_GPUS" "$HEAD_CVD" 1
for spec in $WORKER_SPECS; do
  host=${spec%%:*}
  rest=${spec#*:}
  gpus=${rest%%:*}
  cvd=${rest#*:}
  start_ray_node "$host" "$gpus" "$cvd" 0
done
sleep 3
ssh -o BatchMode=yes "$HEAD_SSH" \
  "export PATH='$VENV/bin:/usr/local/bin:\$PATH'; export RAY_ADDRESS='$HEAD_HOST:$RAY_PORT'; ray status" \
  | tee -a "$LOG"

# Hydra overrides: key=value
# SMOKE_STEPS only overrides step count / saver; keeps config trial_name (do not rename to DIS).
EXTRA_ARGS=()
if [[ -n "$SMOKE_STEPS" ]]; then
  EXTRA_ARGS+=("total_train_steps=${SMOKE_STEPS}")
  EXTRA_ARGS+=("total_train_epochs=${SMOKE_STEPS}")
  EXTRA_ARGS+=("saver.freq_steps=null")
  echo "[e1] SMOKE_STEPS=$SMOKE_STEPS (trial_name left as in config)" | tee -a "$LOG"
fi

cd "$REPO"
nohup env \
  PATH="$VENV/bin:/usr/local/bin:$PATH" \
  PYTHONPATH="$K3_PY:$SITE:$REPO" \
  HOME="$K3/tmp/home" \
  TMPDIR="$K3/tmp" \
  HF_HOME="$HF_HOME" \
  HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" \
  TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  RAY_ADDRESS="$HEAD_HOST:$RAY_PORT" \
  RAY_memory_usage_threshold="$RAY_MEMORY_USAGE_THRESHOLD" \
  NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 TP_SOCKET_IFNAME=eth0 \
  NCCL_IB_DISABLE=1 NCCL_NET=Socket \
  MALLOC_ARENA_MAX=2 \
  SWANLAB_MODE=disabled WANDB_MODE=disabled \
  K25_TAU_LOG_RATIO_SQ="${K25_TAU_LOG_RATIO_SQ:-0.01}" \
  K3_LAMBDA_ENABLED="${K3_LAMBDA_ENABLED:-0}" \
  K3_LAMBDA="${K3_LAMBDA:-0.5}" \
  K3_LAMBDA_N="${K3_LAMBDA_N:-168}" \
  K3_LAMBDA_K="${K3_LAMBDA_K:-8}" \
  K3_LAMBDA_DP="${K3_LAMBDA_DP:-12}" \
  K3_LAMBDA_AUDIT_PATH="${K3_LAMBDA_AUDIT_PATH:-}" \
  K3_EFFORT_ENABLED="${K3_EFFORT_ENABLED:-0}" \
  K3_EFFORT_B0="${K3_EFFORT_B0:-256}" \
  K3_EFFORT_TAU="${K3_EFFORT_TAU:-2.0}" \
  K3_EFFORT_AUDIT_PATH="${K3_EFFORT_AUDIT_PATH:-}" \
  K3_EFFORT_AUDIT_EVERY="${K3_EFFORT_AUDIT_EVERY:-1}" \
  K3_MOPD_ENABLED="${K3_MOPD_ENABLED:-0}" \
  K3_MOPD_RMAX="${K3_MOPD_RMAX:-2.0}" \
  K3_MOPD_ALPHA="${K3_MOPD_ALPHA:-0.5}" \
  K3_MOPD_LEN_NORM="${K3_MOPD_LEN_NORM:-1}" \
  K3_MOPD_ADV_CLIP="${K3_MOPD_ADV_CLIP:-2.0}" \
  K3_MOPD_POS_ONLY="${K3_MOPD_POS_ONLY:-0}" \
  K3_MOPD_AUDIT_PATH="${K3_MOPD_AUDIT_PATH:-}" \
  K3_MOPD_AUDIT_EVERY="${K3_MOPD_AUDIT_EVERY:-8}" \
  PYTHONUNBUFFERED=1 \
  "$VENV/bin/python" "$TRAIN_PY" --config "$CONFIG" "${EXTRA_ARGS[@]}" \
  >>"$LOG" 2>&1 &
echo $! > "$K3/logs/e1_rl.pid"
echo "[e1] started pid=$(cat $K3/logs/e1_rl.pid) — tail -f $LOG"
