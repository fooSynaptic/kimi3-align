#!/usr/bin/env bash
# P3 sweep: fixed H=4, vary τ and/or αβ mask. Baseline = completed P2 (τ=0.01, αβ=0.2/5).
# Arms run sequentially (200 steps each).
#
# Resume examples:
#   SKIP_DONE=1 bash scripts/run_p3_sweep.sh
#   ONLY_ARMS="p3-h4-tau005 p3-h4-mask-tight" bash scripts/run_p3_sweep.sh
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
SKIP_DONE=${SKIP_DONE:-0}
ONLY_ARMS=${ONLY_ARMS:-}
CONTINUE_ON_FAIL=${CONTINUE_ON_FAIL:-0}
export RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-8589934592}"
export RAY_STOP_GRACE="${RAY_STOP_GRACE:-20}"
export PATH="$VENV/bin:/usr/local/bin:$PATH"

QUEUE_LOG=$K3/logs/p3_sweep_$(date +%Y%m%d_%H%M%S).log
STATUS=$K3/docs/JOB_QUEUE.md
REPORT=$K3/docs/P3_SWEEP_REPORT.md
mkdir -p "$K3/logs" "$K3/docs"

exec > >(tee -a "$QUEUE_LOG") 2>&1

# arm_id config tau_for_env
ARMS=(
  "p3-h4-tau0|$K3/configs/p3_h4_tau0_20gpu.yaml|0"
  "p3-h4-tau005|$K3/configs/p3_h4_tau005_20gpu.yaml|0.05"
  "p3-h4-mask-tight|$K3/configs/p3_h4_mask_tight_20gpu.yaml|0.01"
)

declare -A ARM_RES=()
declare -A ARM_LOG=()

# Known-good prior logs (resume)
ARM_RES[p3-h4-tau0]=done
ARM_LOG[p3-h4-tau0]=$K3/logs/p3-h4-tau0_20260808_141418.log

cleanup_host_orphans() {
  echo "[p3] cleanup orphan spawn on HEAD_SSH (TERM only)"
  # Large ppid=1 multiprocessing.spawn leftovers from disk weight sync / vLLM
  local pids
  pids=$(ps -eo pid,ppid,rss,cmd --sort=-rss | awk '$2==1 && $3>500000 && /multiprocessing.spawn/ {print $1}')
  if [[ -z "${pids// /}" ]]; then
    echo "[p3] no large orphan spawn"
    return 0
  fi
  echo "[p3] TERM orphans: $pids"
  for p in $pids; do
    kill -TERM "$p" 2>/dev/null || true
  done
  sleep 12
}

preflight() {
  echo "[p3] preflight"
  cleanup_host_orphans
  local avail
  avail=$(free -g | awk '/^Mem:/{print $7}')
  echo "[p3] host MemAvailable≈${avail}GiB"
  if (( avail < MIN_HOST_AVAIL_GB )); then
    echo "[p3] FAIL host RAM ${avail}GiB < ${MIN_HOST_AVAIL_GB} (retry orphan cleanup once)" >&2
    cleanup_host_orphans
    avail=$(free -g | awk '/^Mem:/{print $7}')
    echo "[p3] host MemAvailable≈${avail}GiB (after retry)"
    if (( avail < MIN_HOST_AVAIL_GB )); then
      echo "[p3] FAIL host RAM still low" >&2
      exit 1
    fi
  fi
  for h in ${HEAD_SSH} ${WORKER_HOSTS}; do
    local used
    used=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$h" \
      "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" | awk '{s+=$1} END{print s+0}')
    echo "[p3] $h vram_sum_MiB=$used"
    if (( used > 2048 )); then
      echo "[p3] WARN dirty VRAM $h — soft ray stop" >&2
      ssh -o BatchMode=yes "$h" \
        "export PATH=$VENV/bin:\$PATH; timeout 40 ray stop --grace-period 15 >/dev/null 2>&1 || true" || true
    fi
  done
}

stop_soft() {
  for h in ${HEAD_SSH} ${WORKER_HOSTS}; do
    ssh -o BatchMode=yes -o ConnectTimeout=15 "$h" \
      "export PATH=$VENV/bin:\$PATH; timeout 50 ray stop --grace-period 20 >/dev/null 2>&1 || true
       for p in \$(ps -eo pid,cmd | awk '/VLLM::|areal_vllm_server|e1_math_rl_train/ && !/awk/ {print \$1}'); do
         kill -TERM \$p 2>/dev/null || true
       done" || true
  done
  sleep 5
  cleanup_host_orphans
}

arm_already_done() {
  local id=$1
  # Prefer final ckpt presence
  local trial
  trial=${id}
  local ckpt_glob=$K3/experiments/checkpoints/root/k3-align-math-rl/${trial}/actor/epoch*globalstep199
  # shellcheck disable=SC2086
  if compgen -G "$ckpt_glob" >/dev/null; then
    return 0
  fi
  # tau0 known done
  if [[ "$id" == "p3-h4-tau0" && -f "${ARM_LOG[p3-h4-tau0]:-}" ]]; then
    grep -q "Train step 200/" "${ARM_LOG[p3-h4-tau0]}" 2>/dev/null && return 0
  fi
  return 1
}

should_run_arm() {
  local id=$1
  if [[ -n "$ONLY_ARMS" ]]; then
    [[ " $ONLY_ARMS " == *" $id "* ]] || return 1
  fi
  if [[ "$SKIP_DONE" == "1" ]] && arm_already_done "$id"; then
    echo "[p3] skip done arm=$id"
    ARM_RES[$id]=done
    return 1
  fi
  return 0
}

write_queue() {
  cat > "$STATUS" <<EOF
# K3-align job queue

Updated: $(date -Iseconds)
P3 log: \`$QUEUE_LOG\`

| Job | Status | Trial / notes |
|-----|--------|---------------|
| Smoke / P0 / P1 | **done** | — |
| P2 async h4 · 200 | **done** | PASS · \`docs/P1_P2_COMPARE.md\` |
| P3 arm tau0 | **${ARM_RES[p3-h4-tau0]:-pending}** | τ=0 · αβ=[0.2,5] |
| P3 arm tau005 | **${ARM_RES[p3-h4-tau005]:-pending}** | τ=0.05 · αβ=[0.2,5] |
| P3 arm mask-tight | **${ARM_RES[p3-h4-mask-tight]:-pending}** | τ=0.01 · αβ=[0.5,2] |
| P3 baseline (P2) | **done** | τ=0.01 · αβ=[0.2,5] · reuse P2 |
| P4–P6 | **blocked** | not implemented |

## Logs
$(for a in p3-h4-tau0 p3-h4-tau005 p3-h4-mask-tight; do echo "- $a: \`${ARM_LOG[$a]:-n/a}\`"; done)
EOF
}

wipe_trial_name_resolve() {
  local trial=$1
  python3 - <<PY
from pathlib import Path
import shutil
p = Path("$K3/tmp/name_resolve/k3-align-math-rl/$trial")
if p.is_dir():
    shutil.rmtree(p)
    print("[p3] wiped", p)
PY
}

run_arm() {
  local id=$1 config=$2 tau=$3
  local log=$K3/logs/${id}_$(date +%Y%m%d_%H%M%S).log
  local trial
  trial=$(awk '/^trial_name:/{print $2; exit}' "$config")
  echo "[p3] start arm=$id trial=$trial tau=$tau"
  ARM_RES[$id]=running
  ARM_LOG[$id]=$log
  write_queue
  if [[ -n "$trial" ]]; then
    wipe_trial_name_resolve "$trial"
  fi
  # host RAM gate again before launch
  local avail
  avail=$(free -g | awk '/^Mem:/{print $7}')
  if (( avail < MIN_HOST_AVAIL_GB )); then
    cleanup_host_orphans
    avail=$(free -g | awk '/^Mem:/{print $7}')
  fi
  if (( avail < MIN_HOST_AVAIL_GB )); then
    echo "[p3] FAIL arm=$id host RAM ${avail}GiB" >&2
    ARM_RES[$id]=failed
    write_queue
    return 1
  fi
  CONFIG="$config" LOG="$log" \
    K25_TAU_LOG_RATIO_SQ="$tau" \
    RAY_OBJECT_STORE_MEMORY="$RAY_OBJECT_STORE_MEMORY" \
    RAY_STOP_GRACE="$RAY_STOP_GRACE" \
    bash "$LAUNCH" || { ARM_RES[$id]=failed; write_queue; return 1; }
  sleep 8
  if wait_train "$id" "$log" "$EXPECTED_STEPS" "$POLL"; then
    ARM_RES[$id]=done
    write_queue
    stop_soft
    return 0
  fi
  ARM_RES[$id]=failed
  write_queue
  stop_soft
  return 1
}

write_report() {
  cat > "$REPORT" <<EOF
# P3 αβτ sweep report

Generated: $(date -Iseconds)
Fixed: \`max_head_offpolicyness=4\` (from P2). Baseline: P2 τ=0.01 αβ=[0.2,5].
Diag: \`docs/P3_FAIL_DIAG.md\`

| Arm | Result | Factor | Log |
|-----|--------|--------|-----|
| P2 baseline | done | τ=0.01 · αβ=[0.2,5] | \`logs/p2_k25_async_full_20260808_005557.log\` |
| tau0 | ${ARM_RES[p3-h4-tau0]:-n/a} | τ=0 | \`${ARM_LOG[p3-h4-tau0]:-}\` |
| tau005 | ${ARM_RES[p3-h4-tau005]:-n/a} | τ=0.05 | \`${ARM_LOG[p3-h4-tau005]:-}\` |
| mask-tight | ${ARM_RES[p3-h4-mask-tight]:-n/a} | αβ=[0.5,2] | \`${ARM_LOG[p3-h4-mask-tight]:-}\` |

Parse rewards/staleness after all arms finish to pick stable region.
EOF
  echo "[p3] wrote $REPORT"
}

preflight
write_queue
fail=0
ran=0
for spec in "${ARMS[@]}"; do
  IFS='|' read -r id config tau <<<"$spec"
  if ! should_run_arm "$id"; then
    continue
  fi
  ran=$((ran + 1))
  if ! run_arm "$id" "$config" "$tau"; then
    fail=1
    if [[ "$CONTINUE_ON_FAIL" == "1" ]]; then
      echo "[p3] arm $id FAILED — CONTINUE_ON_FAIL=1, next arm" >&2
      continue
    fi
    echo "[p3] arm $id FAILED — stopping queue" >&2
    break
  fi
done
write_report
if (( ran == 0 )); then
  echo "[p3] nothing to run (all skipped?)"
  exit 0
fi
if (( fail )); then
  echo "[p3] SWEEP INCOMPLETE" >&2
  exit 1
fi
echo "[p3] SWEEP ALL PASS"
exit 0
