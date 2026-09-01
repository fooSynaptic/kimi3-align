#!/usr/bin/env bash
# P6 full: approx MOPD 200 steps on 24 GPUs.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
k3_require_sao
k3_mkdirs


LAUNCH_P6=$K3/scripts/launch_p6_24gpu.sh
PREFLIGHT=${PREFLIGHT:-$HOME/.cursor/skills/graceful-gpu-shutdown/scripts/preflight_gpus.sh}
GRACEFUL=${GRACEFUL:-$HOME/.cursor/skills/graceful-gpu-shutdown/scripts/graceful_stop_train.sh}
# shellcheck source=/dev/null
source "$K3/scripts/lib/wait_train.sh"

POLL=${POLL:-60}
EXPECTED_STEPS=${EXPECTED_STEPS:-200}
MIN_HOST_AVAIL_GB=${MIN_HOST_AVAIL_GB:-400}
export RAY_MEMORY_USAGE_THRESHOLD="${RAY_MEMORY_USAGE_THRESHOLD:-0.995}"
export PATH="$VENV/bin:/usr/local/bin:$PATH"

CHAIN_LOG=$K3/logs/p6_full_chain_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$K3/logs" "$K3/docs" "$K3/tmp"
exec > >(tee -a "$CHAIN_LOG") 2>&1

cleanup_host_orphans() {
  echo "[p6-full] cleanup orphan spawn on HEAD_SSH (TERM only)"
  local pids
  pids=$(ssh -o BatchMode=yes "${HEAD_SSH:?set HEAD_SSH}" \
    "ps -eo pid,ppid,rss,cmd --sort=-rss | awk '\$2==1 && \$3>500000 && /multiprocessing.spawn/ {print \$1}'" || true)
  if [[ -z "${pids// /}" ]]; then
    echo "[p6-full] no large orphan spawn"
    return 0
  fi
  echo "[p6-full] TERM orphans: $pids"
  ssh -o BatchMode=yes "${HEAD_SSH:?set HEAD_SSH}" "for p in $pids; do kill -TERM \$p 2>/dev/null || true; done" || true
  sleep 12
}

graceful_stop() {
  echo "[p6-full] graceful stop"
  if [[ -x "$GRACEFUL" ]]; then
    bash "$GRACEFUL" --head "${HEAD_SSH:?set HEAD_SSH}" --workers "${WORKER_HOSTS:?set WORKER_HOSTS}" \
      --pattern e1_math_rl_train --wait 90 --no-preflight || true
  fi
  for h in ${HEAD_SSH} ${WORKER_HOSTS}; do
    ssh -o BatchMode=yes "$h" \
      "export PATH=$VENV/bin:\$PATH
       for p in \$(ps -eo pid,cmd | awk '/e1_math_rl_train|VLLM::|areal_vllm_server/ && !/awk/ {print \$1}'); do
         kill -TERM \$p 2>/dev/null || true
       done
       sleep 5
       timeout 60 ray stop --grace-period 25 >/dev/null 2>&1 || true" || true
  done
  sleep 10
  cleanup_host_orphans
}

echo "[p6-full] preflight"
cleanup_host_orphans
avail=$(ssh -o BatchMode=yes "${HEAD_SSH:?set HEAD_SSH}" "free -g | awk '/^Mem:/{print \$7}'")
echo "[p6-full] MemAvailable≈${avail}GiB (min ${MIN_HOST_AVAIL_GB})"
if (( avail < MIN_HOST_AVAIL_GB )); then
  cleanup_host_orphans
  avail=$(ssh -o BatchMode=yes "${HEAD_SSH:?set HEAD_SSH}" "free -g | awk '/^Mem:/{print \$7}'")
  if (( avail < MIN_HOST_AVAIL_GB )); then
    echo "[p6-full] FAIL host RAM too low" >&2
    exit 1
  fi
fi
if [[ -x "$PREFLIGHT" ]]; then
  bash "$PREFLIGHT" ${HEAD_SSH} ${WORKER_HOSTS} || {
    graceful_stop
    bash "$PREFLIGHT" ${HEAD_SSH} ${WORKER_HOSTS}
  }
else
  echo "[p6-full] WARN missing preflight script"
fi

cfg=$K3/configs/p6_mopd_p1teacher_24gpu.yaml
log=$K3/logs/p6_full_$(date +%Y%m%d_%H%M%S).log
audit=$K3/experiments/checkpoints/root/k3-align-math-rl/p6-mopd-p1teacher-24gpu-v4/mopd_audit.jsonl
rm -rf "$K3/tmp/name_resolve/k3-align-math-rl/p6-mopd-p1teacher-24gpu-v4" || true
rm -f "$audit" || true
mkdir -p "$(dirname "$audit")"

echo "[p6-full] === MOPD v4 200 steps (P1 warm-start, pos-only OPD, xccl) ==="
CONFIG="$cfg" LOG="$log" \
  K3_MOPD_AUDIT_PATH="$audit" \
  K3_MOPD_POS_ONLY="${K3_MOPD_POS_ONLY:-1}" \
  RAY_MEMORY_USAGE_THRESHOLD="$RAY_MEMORY_USAGE_THRESHOLD" \
  bash "$LAUNCH_P6" full

wait_train "p6_full" "$log" "$EXPECTED_STEPS" "$POLL"
graceful_stop

echo "[p6-full] DONE log=$log audit=$audit chain=$CHAIN_LOG"
