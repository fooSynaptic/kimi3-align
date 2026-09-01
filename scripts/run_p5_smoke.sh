#!/usr/bin/env bash
# P5 smoke on 24 GPUs: util gate + effort audit gate.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
k3_require_sao
k3_mkdirs


LAUNCH_P5=$K3/scripts/launch_p5_24gpu.sh
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

REPORT=$K3/docs/p5_smoke_report.md
CHAIN_LOG=$K3/logs/p5_smoke_chain_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$K3/logs" "$K3/docs"

exec > >(tee -a "$CHAIN_LOG") 2>&1

graceful_stop() {
  echo "[p5-smoke] graceful stop 00/02/03"
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
}

echo "[p5-smoke] preflight"
if [[ -x "$PREFLIGHT" ]]; then
  bash "$PREFLIGHT" ${HEAD_SSH} ${WORKER_HOSTS} || {
    echo "[p5-smoke] WARN preflight dirty — attempting graceful stop then recheck" >&2
    graceful_stop
    bash "$PREFLIGHT" ${HEAD_SSH} ${WORKER_HOSTS}
  }
else
  echo "[p5-smoke] WARN missing $PREFLIGHT — skip"
fi

cfg=$K3/configs/smoke/p5_effort_tau2_24gpu_smoke.yaml
log=$K3/logs/p5_smoke_$(date +%Y%m%d_%H%M%S).log
util_tsv=$K3/logs/p5_smoke_gpu_util.tsv
audit=$K3/experiments/checkpoints/root/k3-align-math-rl/p5-effort-tau2-24gpu-smoke/effort_audit.jsonl
trial_nr=$K3/tmp/name_resolve/k3-align-math-rl/p5-effort-tau2-24gpu-smoke

rm -rf "$trial_nr" || true
rm -f "$audit" || true
mkdir -p "$(dirname "$audit")"

echo "[p5-smoke] launch config=$cfg"
CONFIG="$cfg" LOG="$log" \
  K3_EFFORT_AUDIT_PATH="$audit" \
  K3_EFFORT_AUDIT_EVERY=1 \
  bash "$LAUNCH_P5" smoke

start_gpu_util_sampler "$util_tsv"
sleep 8
rc=0
wait_train "p5_smoke" "$log" "$EXPECTED_STEPS" "$POLL" || rc=$?
stop_gpu_util_sampler "$util_tsv"

graceful_stop

if (( rc != 0 )); then
  echo "[p5-smoke] FAIL train" >&2
  exit 1
fi

if [[ ! -f "$audit" ]] || [[ ! -s "$audit" ]]; then
  echo "[p5-smoke] FAIL missing/empty audit $audit" >&2
  exit 1
fi
ok_n=$(grep -c '"event": "ok"' "$audit" || true)
ob_n=$(grep -c '"event": "over_budget"' "$audit" || true)
echo "[p5-smoke] audit ok=$ok_n over_budget=$ob_n lines=$(wc -l < "$audit")"

# util gate via shared helper (SM or mem_pct on active GPUs)
util_peak="n/a"
pass_util=1
if [[ -f "$util_tsv" ]]; then
  if check_gpu_util_gate "$util_tsv" "$MIN_GPU_UTIL"; then
    if [[ -f "${util_tsv}.pass" ]]; then
      # shellcheck disable=SC1090
      source "${util_tsv}.pass"
      util_peak="head_sm=${head_sm:-?} head_mem=${head_mem:-?}"
    else
      util_peak="ok"
    fi
  else
    pass_util=0
    util_peak="below_gate"
  fi
fi
echo "[p5-smoke] util $util_peak (gate ≥${MIN_GPU_UTIL})"

{
  echo "# P5 smoke report (24 GPU · Effort τ=2.0)"
  echo
  echo "Generated: $(date -u +%Y-%m-%dT%H:%MZ)"
  echo
  echo "| Tag | steps | audit ok | over_budget | util | Result |"
  echo "|-----|-------|----------|-------------|------|--------|"
  if (( pass_util )); then
    echo "| smoke | $EXPECTED_STEPS | $ok_n | $ob_n | ${util_peak} | **PASS** |"
  else
    echo "| smoke | $EXPECTED_STEPS | $ok_n | $ob_n | ${util_peak} | FAIL util |"
  fi
  echo
  echo "- log: \`$log\`"
  echo "- audit: \`$audit\`"
  echo "- util tsv: \`$util_tsv\`"
  echo "- note: τ=2.0 budget=512 → over_budget≈0 expected under max_new_tokens=512"
} > "$REPORT"

if (( ! pass_util )); then
  echo "[p5-smoke] FAIL util gate" >&2
  exit 1
fi
echo "[p5-smoke] PASS — see $REPORT"
