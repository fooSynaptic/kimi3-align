#!/usr/bin/env bash
# Infrastructure smoke: P0 (N=8) → P1 (2 steps, h0) → P2 (2 steps, h4).
# Load uses formal efficiency knobs; smoke gate requires GPU util/mem peak mean ≥ MIN_GPU_UTIL%.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
k3_require_sao
k3_mkdirs


LAUNCH=$K3/scripts/launch_e1_rl_20gpu.sh
# shellcheck source=/dev/null
source "$K3/scripts/lib/wait_train.sh"
# shellcheck source=/dev/null
source "$K3/scripts/lib/sample_gpu_util.sh"

POLL=${POLL:-30}
EXPECTED_STEPS=${EXPECTED_STEPS:-2}
MIN_GPU_UTIL=${MIN_GPU_UTIL:-50}
export K25_TAU_LOG_RATIO_SQ="${K25_TAU_LOG_RATIO_SQ:-0.01}"
export RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-8589934592}"
export RAY_STOP_GRACE="${RAY_STOP_GRACE:-20}"
export PATH="$VENV/bin:/usr/local/bin:$PATH"

REPORT=$K3/docs/smoke_chain_report.md
CHAIN_LOG=$K3/logs/smoke_chain_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$K3/logs" "$K3/docs" "$K3/docs/smoke_p0"

exec > >(tee -a "$CHAIN_LOG") 2>&1

preflight_host_mem() {
  # Ray OOM-kills workers near ~0.98 host RAM; prior orphan spawn can shrink MemAvailable below the smoke gate.
  local avail_gb min_gb=${SMOKE_MIN_HOST_AVAIL_GB:-200}
  avail_gb=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "${HEAD_SSH:?set HEAD_SSH}" \
    "free -g | awk '/^Mem:/{print \$7}'")
  echo "[smoke] HEAD_SSH host MemAvailable≈${avail_gb}GiB (min ${min_gb})"
  if (( avail_gb < min_gb )); then
    echo "[smoke] FAIL host RAM too low on HEAD_SSH (available=${avail_gb}GiB < ${min_gb})" >&2
    echo "[smoke] tip: TERM orphan multiprocessing.spawn workers (ppid=1) holding nvidia fds" >&2
    return 1
  fi
}

preflight_gpus() {
  echo "[smoke] preflight GPUs"
  local bad=0
  for h in ${HEAD_SSH} ${WORKER_HOSTS}; do
    local used
    used=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$h" \
      "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" | awk '{s+=$1} END{print s+0}')
    echo "[smoke] $h memory.used_sum_MiB=$used"
    # allow a few MiB idle; fail if >2GiB total used across GPUs
    if (( used > 2048 )); then
      echo "[smoke] WARN dirty VRAM on $h (sum=${used}MiB)" >&2
      bad=1
    fi
  done
  if (( bad )); then
    echo "[smoke] attempting soft cleanup (ray stop + TERM vLLM)"
    for h in ${HEAD_SSH} ${WORKER_HOSTS}; do
      ssh -o BatchMode=yes "$h" \
        "export PATH=$VENV/bin:\$PATH; timeout 40 ray stop --grace-period 15 >/dev/null 2>&1 || true
         for p in \$(ps -eo pid,cmd | awk '/VLLM::|areal_vllm_server|e1_math_rl_train/ && !/awk/ {print \$1}'); do
           kill -TERM \$p 2>/dev/null || true
         done" || true
    done
    sleep 8
  fi
  preflight_host_mem || return 1
}

run_p0_smoke() {
  local out=$K3/docs/smoke_p0/instruct_n8.json
  local log=$K3/logs/smoke_p0_$(date +%Y%m%d_%H%M%S).log
  echo "[smoke] P0 boxed N=8 log=$log"
  export HOME=$K3/tmp/home TMPDIR=$K3/tmp
  export HF_HOME=$SAO/tmp/hf-home HF_DATASETS_CACHE=$SAO/tmp/hf-datasets
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export CUDA_VISIBLE_DEVICES=${P0_GPU:-0}
  export PYTHONPATH="$VENV/lib/python3.12/site-packages:${PYTHONPATH:-}"
  export PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
  mkdir -p "$K3/docs/smoke_p0" "$TMPDIR" "$HOME"
  "$VENV/bin/python" -u "$K3/scripts/math_boxed_probe.py" \
    --model "${MODEL_PATH:?set MODEL_PATH}" \
    --tag smoke_instruct --out "$out" --n 8 \
    >"$log" 2>&1
  if [[ -f "$out" ]]; then
    echo "[smoke] P0 PASS out=$out"
    echo "$log" > "$K3/logs/smoke_p0_last_log.txt"
    return 0
  fi
  echo "[smoke] P0 FAIL" >&2
  tail -40 "$log" >&2 || true
  return 1
}

stop_between_phases() {
  echo "[smoke] inter-phase cleanup"
  for h in ${HEAD_SSH} ${WORKER_HOSTS}; do
    ssh -o BatchMode=yes -o ConnectTimeout=15 "$h" \
      "export PATH=$VENV/bin:\$PATH; timeout 40 ray stop --grace-period 15 >/dev/null 2>&1 || true
       for p in \$(ps -eo pid,cmd | awk '/VLLM::|areal_vllm_server|e1_math_rl_train/ && !/awk/ {print \$1}'); do
         kill -TERM \$p 2>/dev/null || true
       done" || true
  done
  sleep 5
}

run_rl_smoke() {
  local phase=$1
  local config=$2
  local log=$K3/logs/${phase}_$(date +%Y%m%d_%H%M%S).log
  local util_tsv=$K3/logs/${phase}_gpu_util.tsv
  echo "[smoke] start $phase config=$config min_gpu_util=${MIN_GPU_UTIL}%"
  # NOTE: called under `if`; bash disables set -e there — fallible steps need explicit || return.
  CONFIG="$config" LOG="$log" \
    K25_TAU_LOG_RATIO_SQ="$K25_TAU_LOG_RATIO_SQ" \
    RAY_OBJECT_STORE_MEMORY="$RAY_OBJECT_STORE_MEMORY" \
    RAY_STOP_GRACE="$RAY_STOP_GRACE" \
    bash "$LAUNCH" || return 1
  start_gpu_util_sampler "$util_tsv"
  sleep 8
  local rc=0
  wait_train "$phase" "$log" "$EXPECTED_STEPS" "$POLL" || rc=$?
  stop_gpu_util_sampler "$util_tsv"
  if (( rc != 0 )); then
    return 1
  fi
  check_gpu_util_gate "$util_tsv" "$MIN_GPU_UTIL" || return 1
  echo "$log" > "$K3/logs/${phase}_last_log.txt"
  echo "$util_tsv" > "$K3/logs/${phase}_util_tsv.txt"
  return 0
}

write_report() {
  local p0=$1 p1=$2 p2=$3
  local p1u p2u
  p1u=$(cat "$K3/logs/smoke_p1_gpu_util.tsv.pass" 2>/dev/null || echo "n/a")
  p2u=$(cat "$K3/logs/smoke_p2_gpu_util.tsv.pass" 2>/dev/null || echo "n/a")
  cat > "$REPORT" <<EOF
# Smoke chain report

Generated: $(date -Iseconds)
Chain log: \`$CHAIN_LOG\`
Efficiency: formal knobs (batch=160, n_samples=8, max_concurrent=32, gpu_memory_utilization=0.55); util gate ≥${MIN_GPU_UTIL}%

| Phase | Result | Notes |
|-------|--------|-------|
| P0 boxed N=8 | $p0 | instruct-only probe |
| P1 sync h0 (2 steps) | $p1 | trial \`p1-k25-smoke\` · util peak (SM% MEM%)=\`$p1u\` |
| P2 async h4 (2 steps) | $p2 | trial \`p2-k25-smoke\` · util peak (SM% MEM%)=\`$p2u\` |

P3+ : skipped (not implemented)

## Logs
- P0: \`$(cat $K3/logs/smoke_p0_last_log.txt 2>/dev/null || echo n/a)\`
- P1: \`$(cat $K3/logs/smoke_p1_last_log.txt 2>/dev/null || echo n/a)\`
- P2: \`$(cat $K3/logs/smoke_p2_last_log.txt 2>/dev/null || echo n/a)\`
- P1 util: \`$(cat $K3/logs/smoke_p1_util_tsv.txt 2>/dev/null || echo n/a)\`
- P2 util: \`$(cat $K3/logs/smoke_p2_util_tsv.txt 2>/dev/null || echo n/a)\`
EOF
  echo "[smoke] wrote $REPORT"
}

P0_RES=FAIL
P1_RES=FAIL
P2_RES=FAIL

preflight_gpus
if run_p0_smoke; then P0_RES=PASS; fi
# free P0 GPU before Ray train
stop_between_phases

if [[ "$P0_RES" != PASS ]]; then
  write_report "$P0_RES" "$P1_RES" "$P2_RES"
  exit 1
fi

if run_rl_smoke smoke_p1 "$K3/configs/smoke/p1_k25_sync_h0_smoke.yaml"; then
  P1_RES=PASS
else
  write_report "$P0_RES" "$P1_RES" "$P2_RES"
  exit 1
fi

stop_between_phases
# clear P1/P2 smoke trial name_resolve before P2 async smoke (same NFS hygiene as full runs).
rm -rf "$K3/tmp/name_resolve/k3-align-math-rl/p1-k25-smoke" \
       "$K3/tmp/name_resolve/k3-align-math-rl/p2-k25-smoke" || true

if run_rl_smoke smoke_p2 "$K3/configs/smoke/p2_k25_async_h4_smoke.yaml"; then
  P2_RES=PASS
else
  write_report "$P0_RES" "$P1_RES" "$P2_RES"
  exit 1
fi

stop_between_phases
write_report "$P0_RES" "$P1_RES" "$P2_RES"
echo "[smoke] ALL PASS"
exit 0
