#!/usr/bin/env bash
# P6 smoke on 24 GPUs: util gate + MOPD audit gate.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
k3_require_sao
k3_mkdirs


LAUNCH_P6=$K3/scripts/launch_p6_24gpu.sh
PREFLIGHT=${PREFLIGHT:-$HOME/.cursor/skills/graceful-gpu-shutdown/scripts/preflight_gpus.sh}
# shellcheck source=/dev/null
source "$K3/scripts/lib/wait_train.sh"
# shellcheck source=/dev/null
source "$K3/scripts/lib/sample_gpu_util.sh"

POLL=${POLL:-30}
EXPECTED_STEPS=${EXPECTED_STEPS:-2}
MIN_GPU_UTIL=${MIN_GPU_UTIL:-50}
export GPU_UTIL_HOSTS=${GPU_UTIL_HOSTS:?set GPU_UTIL_HOSTS}
export PATH="$VENV/bin:/usr/local/bin:$PATH"

REPORT=$K3/docs/p6_smoke_report.md
CHAIN_LOG=$K3/logs/p6_smoke_chain_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$K3/logs" "$K3/docs"

exec > >(tee -a "$CHAIN_LOG") 2>&1

cleanup_host_orphans() {
  echo "[p6-smoke] cleanup orphan spawn on HEAD_SSH (TERM only)"
  local pids
  pids=$(ps -eo pid,ppid,rss,cmd --sort=-rss | awk '$2==1 && $3>500000 && /multiprocessing.spawn/ {print $1}')
  if [[ -z "${pids// /}" ]]; then
    echo "[p6-smoke] no large orphan spawn"
    return 0
  fi
  echo "[p6-smoke] TERM orphans: $pids"
  for p in $pids; do kill -TERM "$p" 2>/dev/null || true; done
  sleep 12
}

graceful_stop() {
  echo "[p6-smoke] graceful stop 00/02/03"
  for h in ${HEAD_SSH} ${WORKER_HOSTS}; do
    ssh -o BatchMode=yes "$h" \
      "export PATH=$VENV/bin:\$PATH
       for p in \$(ps -eo pid,cmd | awk '/e1_math_rl_train|VLLM::|areal_vllm_server/ && !/awk/ {print \$1}'); do
         kill -TERM \$p 2>/dev/null || true
       done
       sleep 5
       timeout 50 ray stop --grace-period 20 >/dev/null 2>&1 || true" || true
  done
  sleep 8
  cleanup_host_orphans
}

echo "[p6-smoke] preflight"
cleanup_host_orphans
if [[ -x "$PREFLIGHT" ]]; then
  bash "$PREFLIGHT" ${HEAD_SSH} ${WORKER_HOSTS} || {
    echo "[p6-smoke] WARN preflight dirty — attempting graceful stop then recheck" >&2
    graceful_stop
    bash "$PREFLIGHT" ${HEAD_SSH} ${WORKER_HOSTS}
  }
else
  echo "[p6-smoke] WARN missing $PREFLIGHT — skip"
fi

cfg=$K3/configs/smoke/p6_mopd_p1teacher_24gpu_smoke.yaml
log=$K3/logs/p6_smoke_$(date +%Y%m%d_%H%M%S).log
util_tsv=$K3/logs/p6_smoke_gpu_util.tsv
audit=$K3/experiments/checkpoints/root/k3-align-math-rl/p6-mopd-p1teacher-24gpu-smoke/mopd_audit.jsonl
trial_nr=$K3/tmp/name_resolve/k3-align-math-rl/p6-mopd-p1teacher-24gpu-smoke

rm -rf "$trial_nr" || true
rm -f "$audit" || true
mkdir -p "$(dirname "$audit")"

echo "[p6-smoke] launch config=$cfg"
CONFIG="$cfg" LOG="$log" \
  K3_MOPD_AUDIT_PATH="$audit" \
  bash "$LAUNCH_P6" smoke

start_gpu_util_sampler "$util_tsv"
sleep 8
rc=0
wait_train "p6_smoke" "$log" "$EXPECTED_STEPS" "$POLL" || rc=$?
stop_gpu_util_sampler "$util_tsv"

graceful_stop

if (( rc != 0 )); then
  echo "[p6-smoke] FAIL train" >&2
  exit 1
fi

if [[ ! -f "$audit" ]] || [[ ! -s "$audit" ]]; then
  echo "[p6-smoke] FAIL missing/empty audit $audit" >&2
  tail -40 "$log" >&2 || true
  exit 1
fi
opd_n=$(grep -c '"event": "opd"' "$audit" || true)
echo "[p6-smoke] audit opd_events=$opd_n lines=$(wc -l < "$audit")"
if (( opd_n < 1 )); then
  echo "[p6-smoke] FAIL audit has no opd events (boot-only is not enough)" >&2
  tail -20 "$audit" >&2 || true
  exit 1
fi
# v2 scale gate: suffix-sum stays O(1) per step; v1 without length-norm scaled O(T) and collapsed (~30 |A_opd| vs 0/1 outcome).
max_abs=$(python3 -c "
import json, sys
mx=0.0
for line in open(sys.argv[1]):
    o=json.loads(line)
    if o.get('event')!='opd':
        continue
    mx=max(mx, abs(float(o.get('max_abs_suffix') or 0)))
print(f'{mx:.4f}')
" "$audit")
echo "[p6-smoke] max_abs_suffix=$max_abs (gate <5)"
python3 -c "import sys; v=float(sys.argv[1]); sys.exit(0 if 0 < v < 5 else 1)" "$max_abs" || {
  echo "[p6-smoke] FAIL OPD scale max_abs_suffix=$max_abs (want (0, 5))" >&2
  tail -5 "$audit" >&2 || true
  exit 1
}

pass_util=1
util_peak="n/a"
if [[ -f "$util_tsv" ]]; then
  if check_gpu_util_gate "$util_tsv" "$MIN_GPU_UTIL"; then
    if [[ -f "${util_tsv}.pass" ]]; then
      util_peak=$(tr '\n' ' ' < "${util_tsv}.pass")
    else
      util_peak="ok"
    fi
  else
    pass_util=0
    util_peak="below_gate"
  fi
fi

{
  echo "# P6 smoke report (24 GPU · approx MOPD · P1 teacher)"
  echo
  echo "Generated: $(date -u +%Y-%m-%dT%H:%MZ)"
  echo
  echo "| Tag | steps | audit opd | util | Result |"
  echo "|-----|-------|-----------|------|--------|"
  if (( pass_util )); then
    echo "| smoke | $EXPECTED_STEPS | $opd_n | ${util_peak} | **PASS** |"
  else
    echo "| smoke | $EXPECTED_STEPS | $opd_n | ${util_peak} | FAIL util |"
  fi
  echo
  echo "- log: \`$log\`"
  echo "- audit: \`$audit\`"
  echo "- util tsv: \`$util_tsv\`"
} > "$REPORT"

if (( ! pass_util )); then
  echo "[p6-smoke] FAIL util gate" >&2
  exit 1
fi
echo "[p6-smoke] PASS — see $REPORT"
