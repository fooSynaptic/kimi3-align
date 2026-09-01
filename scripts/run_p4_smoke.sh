#!/usr/bin/env bash
# P4 smoke on 16 GPUs: util gate + λ audit gate + optional efficiency sweep.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
k3_require_sao
k3_mkdirs


LAUNCH_P4=$K3/scripts/launch_p4_16gpu.sh
# shellcheck source=/dev/null
source "$K3/scripts/lib/wait_train.sh"
# shellcheck source=/dev/null
source "$K3/scripts/lib/sample_gpu_util.sh"

POLL=${POLL:-30}
EXPECTED_STEPS=${EXPECTED_STEPS:-4}
MIN_GPU_UTIL=${MIN_GPU_UTIL:-50}
export GPU_UTIL_HOSTS=${GPU_UTIL_HOSTS:?set GPU_UTIL_HOSTS}
export PATH="$VENV/bin:/usr/local/bin:$PATH"

REPORT=$K3/docs/p4_smoke_report.md
CHAIN_LOG=$K3/logs/p4_smoke_chain_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$K3/logs" "$K3/docs"

exec > >(tee -a "$CHAIN_LOG") 2>&1

preflight() {
  echo "[p4-smoke] preflight 00/02"
  local bad=0
  for h in ${HEAD_SSH} ${WORKER_HOSTS}; do
    local used
    used=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$h" \
      "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" | awk '{s+=$1} END{print s+0}')
    echo "[p4-smoke] $h memory.used_sum_MiB=$used"
    if (( used > 2048 )); then
      echo "[p4-smoke] WARN dirty VRAM on $h" >&2
      bad=1
    fi
  done
  if (( bad )); then
    for h in ${HEAD_SSH} ${WORKER_HOSTS}; do
      ssh -o BatchMode=yes "$h" \
        "export PATH=$VENV/bin:\$PATH; timeout 40 ray stop --grace-period 15 >/dev/null 2>&1 || true
         for p in \$(ps -eo pid,cmd | awk '/VLLM::|areal_vllm_server|e1_math_rl_train/ && !/awk/ {print \$1}'); do
           kill -TERM \$p 2>/dev/null || true
         done" || true
    done
    sleep 8
  fi
  local avail_gb min_gb=${SMOKE_MIN_HOST_AVAIL_GB:-200}
  avail_gb=$(ssh -o BatchMode=yes "${HEAD_SSH:?set HEAD_SSH}" "free -g | awk '/^Mem:/{print \$7}'")
  echo "[p4-smoke] HEAD_SSH MemAvailable≈${avail_gb}GiB (min ${min_gb})"
  if (( avail_gb < min_gb )); then
    echo "[p4-smoke] FAIL host RAM too low" >&2
    return 1
  fi
}

check_audit() {
  local audit=$1
  if [[ ! -f "$audit" ]]; then
    echo "[p4-smoke] FAIL missing audit $audit" >&2
    return 1
  fi
  if ! grep -q '"event": "pause"' "$audit"; then
    echo "[p4-smoke] FAIL audit has no pause event" >&2
    tail -20 "$audit" >&2 || true
    return 1
  fi
  echo "[p4-smoke] audit OK pause_count=$(grep -c '"event": "pause"' "$audit" || true)"
  return 0
}

estimate_sec_step() {
  local log=$1
  python3 - <<'PY' "$log"
import re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(errors="ignore")
# Prefer explicit timing lines if present
times = [float(x) for x in re.findall(r"time/step[^\d]*([0-9.]+)", text)]
if len(times) >= 2:
    print(f"{sum(times[-3:]) / len(times[-3:]):.1f}")
    raise SystemExit
# Fallback: wall between first and last Train step
steps = list(re.finditer(r"Train step (\d+)/", text))
ts = list(re.finditer(r"202\d{5}-\d{2}:\d{2}:\d{2}", text))
if len(steps) >= 2 and len(ts) >= 2:
    # crude: use file mtime span / steps — better parse areal timing
    pass
# last resort from "sec" patterns
secs = [float(x) for x in re.findall(r"([0-9]+\.[0-9]+)\s*s/step", text)]
if secs:
    print(f"{secs[-1]:.1f}")
else:
    print("n/a")
PY
}

run_one() {
  local tag=$1
  local batch=$2
  local concurrent=$3
  local gpu_mem=$4
  local max_seqs=$5

  local cfg=$K3/configs/smoke/p4_h4_lambda05_16gpu_smoke.yaml
  local log=$K3/logs/p4_smoke_${tag}_$(date +%Y%m%d_%H%M%S).log
  local util_tsv=$K3/logs/p4_smoke_${tag}_gpu_util.tsv
  local audit=$K3/experiments/checkpoints/root/k3-align-math-rl/p4-h4-lambda05-16gpu-smoke/lambda_audit.jsonl
  local trial_nr=$K3/tmp/name_resolve/k3-align-math-rl/p4-h4-lambda05-16gpu-smoke

  rm -rf "$trial_nr" || true
  rm -f "$audit" || true
  mkdir -p "$(dirname "$audit")"

  echo "[p4-smoke] === $tag batch=$batch concurrent=$concurrent gpu_mem=$gpu_mem max_seqs=$max_seqs ==="

  # temporary config overlay (env-pass avoids bash/python brace clashes)
  local tmp_cfg=$K3/tmp/p4_smoke_${tag}.yaml
  mkdir -p "$K3/tmp"
  BATCH="$batch" CONC="$concurrent" GMEM="$gpu_mem" MSEQS="$max_seqs" \
  SRC="$cfg" DST="$tmp_cfg" \
  python3 - <<'PY'
from pathlib import Path
import os, re
src = Path(os.environ["SRC"]).read_text()
batch, conc, gmem, mseqs = (os.environ[k] for k in ("BATCH", "CONC", "GMEM", "MSEQS"))
src = re.sub(r"(batch_size:\s*)\d+", rf"\g<1>{batch}", src, count=1)
src = re.sub(r"(max_concurrent_rollouts:\s*)\d+", rf"\g<1>{conc}", src, count=1)
src = re.sub(r"(gpu_memory_utilization:\s*)[0-9.]+", rf"\g<1>{gmem}", src, count=1)
src = re.sub(r"(max_num_seqs:\s*)\d+", rf"\g<1>{mseqs}", src, count=1)
src = re.sub(r'K3_LAMBDA_N:\s*"\d+"', f'K3_LAMBDA_N: "{batch}"', src)
Path(os.environ["DST"]).write_text(src)
print("wrote", os.environ["DST"])
PY

  CONFIG="$tmp_cfg" LOG="$log" \
    K3_LAMBDA_N="$batch" \
    K3_LAMBDA_AUDIT_PATH="$audit" \
    bash "$LAUNCH_P4" smoke || return 1

  start_gpu_util_sampler "$util_tsv"
  sleep 8
  local rc=0
  wait_train "p4_$tag" "$log" "$EXPECTED_STEPS" "$POLL" || rc=$?
  stop_gpu_util_sampler "$util_tsv"

  # graceful stop between arms
  for h in ${HEAD_SSH} ${WORKER_HOSTS}; do
    ssh -o BatchMode=yes "$h" \
      "export PATH=$VENV/bin:\$PATH; timeout 40 ray stop --grace-period 15 >/dev/null 2>&1 || true
       for p in \$(ps -eo pid,cmd | awk '/VLLM::|areal_vllm_server|e1_math_rl_train/ && !/awk/ {print \$1}'); do
         kill -TERM \$p 2>/dev/null || true
       done" || true
  done
  sleep 5

  if (( rc != 0 )); then
    echo "[p4-smoke] $tag FAIL train" >&2
    return 1
  fi
  check_gpu_util_gate "$util_tsv" "$MIN_GPU_UTIL" || return 1
  check_audit "$audit" || return 1

  local sec
  sec=$(estimate_sec_step "$log")
  local util_pass
  util_pass=$(cat "${util_tsv}.pass" 2>/dev/null || echo n/a)
  echo "[p4-smoke] $tag PASS sec/step≈$sec util=$util_pass"
  echo "$tag|$batch|$concurrent|$gpu_mem|$max_seqs|$sec|$util_pass|$log|$audit" >> "$K3/logs/p4_smoke_eff_candidates.tsv"
  echo "$log" > "$K3/logs/p4_smoke_last_log.txt"
  echo "$sec" > "$K3/logs/p4_smoke_last_sec_step.txt"
  return 0
}

pick_best_and_report() {
  local best_line
  best_line=$(python3 - <<'PY'
from pathlib import Path
p = Path(__import__("os").environ["K3"]) / "logs" / "p4_smoke_eff_candidates.tsv"
if not p.exists():
    print("")
    raise SystemExit
rows = [ln.strip().split("|") for ln in p.read_text().splitlines() if ln.strip()]
def key(r):
    try:
        return float(r[5])
    except Exception:
        return 1e18
rows.sort(key=key)
print("|".join(rows[0]) if rows else "")
PY
)
  echo "$best_line" > "$K3/logs/p4_smoke_best.txt"
  IFS='|' read -r tag batch conc mem seqs sec util log audit <<<"$best_line"
  {
    echo "# P4 smoke report (16 GPU · λ=0.5)"
    echo
    echo "Generated: $(date -Iseconds)"
    echo "Chain log: \`$CHAIN_LOG\`"
    echo
    echo "| Tag | batch | concurrent | gpu_mem | max_seqs | sec/step | util peak | Result |"
    echo "|-----|-------|------------|---------|----------|----------|-----------|--------|"
    if [[ -f $K3/logs/p4_smoke_eff_candidates.tsv ]]; then
      while IFS='|' read -r t b c m s sec_u u l a; do
        echo "| \`$t\` | $b | $c | $m | $s | $sec_u | \`$u\` | PASS |"
      done < "$K3/logs/p4_smoke_eff_candidates.tsv"
    fi
    echo
    echo "**Selected for full:** tag=\`$tag\` batch=$batch concurrent=$conc gpu_mem=$mem max_seqs=$seqs sec/step≈$sec"
    echo
    echo "Audit gate: pause events required. Best audit: \`$audit\`"
    echo
    echo "## Claim"
    echo "Scheme B approx λ-partial — see \`docs/P4_PLAN.md\`."
  } > "$REPORT"
  echo "[p4-smoke] wrote $REPORT best=$best_line"
}

rm -f "$K3/logs/p4_smoke_eff_candidates.tsv"
preflight

# Primary smoke (formal knobs) then one efficiency upsweep if base passes.
export EXPECTED_STEPS=${EXPECTED_STEPS:-2}
run_one "base" 168 32 0.55 64 || {
  echo "[p4-smoke] base failed" >&2
  exit 1
}
run_one "conc48_mem65" 168 48 0.65 96 || echo "[p4-smoke] conc48_mem65 failed (keep base)"

pick_best_and_report
echo "[p4-smoke] ALL DONE — apply best knobs to full config before 200-step run"
exit 0
