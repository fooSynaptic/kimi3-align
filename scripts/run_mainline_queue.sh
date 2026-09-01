#!/usr/bin/env bash
# Mainline job queue after smoke green: P2 formal (200) → write status.
# P0/P1 assumed done; P3+ queue entries were not wired in this script (see run_p3_sweep.sh).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
k3_require_sao
k3_mkdirs


LAUNCH=$K3/scripts/launch_e1_rl_20gpu.sh
# shellcheck source=/dev/null
source "$K3/scripts/lib/wait_train.sh"

POLL=${POLL:-60}
EXPECTED_STEPS=${EXPECTED_STEPS:-200}
MIN_HOST_AVAIL_GB=${MIN_HOST_AVAIL_GB:-200}
export K25_TAU_LOG_RATIO_SQ="${K25_TAU_LOG_RATIO_SQ:-0.01}"
export RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-8589934592}"
export RAY_STOP_GRACE="${RAY_STOP_GRACE:-20}"
export PATH="$VENV/bin:/usr/local/bin:$PATH"

QUEUE_LOG=$K3/logs/mainline_queue_$(date +%Y%m%d_%H%M%S).log
STATUS=$K3/docs/JOB_QUEUE.md
mkdir -p "$K3/logs" "$K3/docs"

exec > >(tee -a "$QUEUE_LOG") 2>&1

write_status() {
  local p2=$1
  cat > "$STATUS" <<EOF
# K3-align job queue

Updated: $(date -Iseconds)
Queue log: \`$QUEUE_LOG\`

| Job | Status | Trial / notes |
|-----|--------|---------------|
| Smoke chain | **done** | util≥50% · \`docs/smoke_chain_report.md\` |
| P0 Instruct boxed N=256 | **done** | 65.625% |
| P1 sync h0 · 200 step | **done** | \`p1-k25-sync-h0-full\` |
| P2 async h4 · 200 step | **$p2** | \`p2-k25-async-h4\` · config \`configs/p2_k25_async_h4_20gpu.yaml\` |
| P3 αβτ sweep | **blocked** | not implemented |
| P4 λ-barrier | **blocked** | not implemented |
| P5 Effort schedule | **blocked** | not implemented |
| P6+ GRM/MOPD | **blocked** | out of scope |

## Active
- P2 log: \`$(cat $K3/logs/p2_k25_full_last_log.txt 2>/dev/null || echo pending)\`
- Driver pidfile: \`$K3/logs/e1_rl.pid\`
EOF
  echo "[queue] wrote $STATUS"
}

preflight() {
  echo "[queue] preflight"
  local avail
  avail=$(free -g | awk '/^Mem:/{print $7}')
  echo "[queue] host MemAvailable≈${avail}GiB"
  if (( avail < MIN_HOST_AVAIL_GB )); then
    echo "[queue] FAIL host RAM ${avail}GiB < ${MIN_HOST_AVAIL_GB}" >&2
    exit 1
  fi
  for h in ${HEAD_SSH} ${WORKER_HOSTS}; do
    local used
    used=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$h" \
      "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" | awk '{s+=$1} END{print s+0}')
    echo "[queue] $h vram_sum_MiB=$used"
    if (( used > 2048 )); then
      echo "[queue] FAIL dirty VRAM on $h" >&2
      exit 1
    fi
  done
}

run_p2_full() {
  local config=$K3/configs/p2_k25_async_h4_20gpu.yaml
  local log=$K3/logs/p2_k25_async_full_$(date +%Y%m%d_%H%M%S).log
  local trial
  trial=$(awk '/^trial_name:/{print $2; exit}' "$config")
  echo "[queue] start P2 full trial=$trial expected_steps=$EXPECTED_STEPS"
  # name_resolve cleared here and again in launch — duplicate wipe prevents stale NFS keys on resume.
  if [[ -n "$trial" ]]; then
    local nr=$K3/tmp/name_resolve/k3-align-math-rl/$trial
    if [[ -d "$nr" ]]; then
      echo "[queue] wipe name_resolve $nr"
      find "$nr" -mindepth 1 -delete 2>/dev/null || true
    fi
  fi
  CONFIG="$config" LOG="$log" \
    K25_TAU_LOG_RATIO_SQ="$K25_TAU_LOG_RATIO_SQ" \
    RAY_OBJECT_STORE_MEMORY="$RAY_OBJECT_STORE_MEMORY" \
    RAY_STOP_GRACE="$RAY_STOP_GRACE" \
    bash "$LAUNCH" || return 1
  echo "$log" > "$K3/logs/p2_k25_full_last_log.txt"
  sleep 8
  wait_train p2_k25_full "$log" "$EXPECTED_STEPS" "$POLL" || return 1
  return 0
}

stop_ray_soft() {
  echo "[queue] soft stop ray"
  for h in ${HEAD_SSH} ${WORKER_HOSTS}; do
    ssh -o BatchMode=yes -o ConnectTimeout=15 "$h" \
      "export PATH=$VENV/bin:\$PATH; timeout 50 ray stop --grace-period 20 >/dev/null 2>&1 || true" || true
  done
}

preflight
write_status "running"
if run_p2_full; then
  write_status "done"
  stop_ray_soft
  echo "[queue] P2 FULL PASS"
  # refresh design progress snippet
  cat >> "$K3/docs/P2_DONE.txt" <<EOF
P2 full done at $(date -Iseconds)
log=$(cat $K3/logs/p2_k25_full_last_log.txt)
EOF
  exit 0
fi

write_status "failed"
echo "[queue] P2 FULL FAIL" >&2
exit 1
