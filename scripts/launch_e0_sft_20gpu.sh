#!/usr/bin/env bash
# K3-align E0 MATH SFT · 20 Hopper (96GB HBM): head 8 + two workers (8+4)
# Requires: SAO_ROOT, HEAD_SSH, HEAD_HOST, WORKER_HOSTS (two ssh aliases)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
k3_require_sao
k3_mkdirs
k3_require_ray_topo
if [[ -z "${WORKER_HOSTS:-}" ]]; then
  echo "[fatal] Set WORKER_HOSTS to two ssh aliases (8-GPU then 4-GPU worker)." >&2
  exit 1
fi
read -r W02_SSH W03_SSH _rest <<<"$WORKER_HOSTS"
if [[ -z "${W02_SSH:-}" || -z "${W03_SSH:-}" ]]; then
  echo "[fatal] WORKER_HOSTS must list two hosts." >&2
  exit 1
fi


REPO=${AREAL_REPO}
RAY_TMP=/dev/shm/r
CONFIG=${CONFIG:-$K3/configs/e0_math_sft_20gpu.yaml}
LOG=${LOG:-$K3/logs/e0_sft_$(date +%Y%m%d_%H%M%S).log}
RAY_PORT=${RAY_PORT:-6379}
# 20 GPUs: full 8 on 00/02, 4 on 03
HEAD_CVD=${HEAD_CVD:-0,1,2,3,4,5,6,7}
HEAD_GPUS=${HEAD_GPUS:-8}
W02_CVD=${W02_CVD:-0,1,2,3,4,5,6,7}
W02_GPUS=${W02_GPUS:-8}
W03_CVD=${W03_CVD:-0,1,2,3}
W03_GPUS=${W03_GPUS:-4}
RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-42949672960}
RAY_MEMORY_USAGE_THRESHOLD=${RAY_MEMORY_USAGE_THRESHOLD:-0.98}
TRAIN_PY=$K3/scripts/e0_math_sft_train.py
SMOKE_STEPS=${SMOKE_STEPS:-}

export PATH="$VENV/bin:/usr/local/bin:$PATH"
export PYTHONPATH="$SITE:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HOME="$K3/tmp/home"
export TMPDIR="$K3/tmp"
export HF_HOME="$SAO/tmp/hf-home"
export HF_DATASETS_CACHE="$SAO/tmp/hf-datasets"
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export RAY_ADDRESS="${HEAD_HOST}:${RAY_PORT}"
export RAY_memory_usage_threshold="$RAY_MEMORY_USAGE_THRESHOLD"
export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 TP_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1 NCCL_NET=Socket
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}

mkdir -p "$K3/logs" "$K3/experiments" "$K3/tmp/home" "$K3/tmp/name_resolve"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[fatal] missing $VENV" >&2
  exit 1
fi
if [[ ! -d "$K3/data/math_sft_chat" ]]; then
  echo "[fatal] missing data — run: bash $K3/scripts/run_prepare_math_sft.sh" >&2
  exit 1
fi

stop_ray() {
  local host=$1 grace=${RAY_STOP_GRACE:-60}
  echo "[ray] stop $host (grace=${grace}s)" | tee -a "$LOG"
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$host" \
    "export PATH=$VENV/bin:/usr/local/bin:\$PATH; ray stop --grace-period $grace >/dev/null 2>&1 || true" || true
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

echo "[e0] log=$LOG config=$CONFIG" | tee "$LOG"
for h in $HEAD_SSH $WORKER_HOSTS; do stop_ray "$h"; done
start_ray_node "$HEAD_SSH" "$HEAD_GPUS" "$HEAD_CVD" 1
start_ray_node "$W02_SSH" "$W02_GPUS" "$W02_CVD" 0
start_ray_node "$W03_SSH" "$W03_GPUS" "$W03_CVD" 0
sleep 3
ssh -o BatchMode=yes "$HEAD_SSH" \
  "export PATH='$VENV/bin:/usr/local/bin:\$PATH'; export RAY_ADDRESS='$HEAD_HOST:$RAY_PORT'; ray status" \
  | tee -a "$LOG"

# Hydra overrides: key=value (not --key value)
EXTRA_ARGS=()
if [[ -n "$SMOKE_STEPS" ]]; then
  EXTRA_ARGS+=("total_train_steps=${SMOKE_STEPS}")
fi

cd "$REPO"
nohup env \
  PATH="$VENV/bin:/usr/local/bin:$PATH" \
  PYTHONPATH="$SITE:$REPO" \
  HOME="$K3/tmp/home" \
  TMPDIR="$K3/tmp" \
  HF_HOME="$HF_HOME" \
  HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  RAY_ADDRESS="$HEAD_HOST:$RAY_PORT" \
  RAY_memory_usage_threshold="$RAY_MEMORY_USAGE_THRESHOLD" \
  NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 TP_SOCKET_IFNAME=eth0 \
  NCCL_IB_DISABLE=1 NCCL_NET=Socket \
  MALLOC_ARENA_MAX=2 \
  PYTHONUNBUFFERED=1 \
  "$VENV/bin/python" "$TRAIN_PY" --config "$CONFIG" "${EXTRA_ARGS[@]}" \
  >>"$LOG" 2>&1 &
echo $! > "$K3/logs/e0_sft.pid"
echo "[e0] started pid=$(cat $K3/logs/e0_sft.pid) — tail -f $LOG"
