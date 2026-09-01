#!/usr/bin/env bash
# P5 full: Stage A τ=2.0 (100) → graceful stop → Stage B τ=1.0 (100) from A final.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
k3_require_sao
k3_mkdirs


LAUNCH_P5=$K3/scripts/launch_p5_24gpu.sh
PREFLIGHT=${PREFLIGHT:-$HOME/.cursor/skills/graceful-gpu-shutdown/scripts/preflight_gpus.sh}
GRACEFUL=${GRACEFUL:-$HOME/.cursor/skills/graceful-gpu-shutdown/scripts/graceful_stop_train.sh}
# shellcheck source=/dev/null
source "$K3/scripts/lib/wait_train.sh"

POLL=${POLL:-60}
EXPECTED_STEPS=${EXPECTED_STEPS:-100}
MIN_HOST_AVAIL_GB=${MIN_HOST_AVAIL_GB:-400}
# disk weight-sync can leave large ppid=1 spawn trees; MemAvailable threshold raised accordingly.
export RAY_MEMORY_USAGE_THRESHOLD="${RAY_MEMORY_USAGE_THRESHOLD:-0.995}"
export PATH="$VENV/bin:/usr/local/bin:$PATH"

CHAIN_LOG=$K3/logs/p5_full_chain_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$K3/logs" "$K3/docs" "$K3/tmp"
exec > >(tee -a "$CHAIN_LOG") 2>&1

cleanup_host_orphans() {
  echo "[p5-full] cleanup orphan spawn on HEAD_SSH (TERM only)"
  local pids
  pids=$(ssh -o BatchMode=yes "${HEAD_SSH:?set HEAD_SSH}" \
    "ps -eo pid,ppid,rss,cmd --sort=-rss | awk '\$2==1 && \$3>500000 && /multiprocessing.spawn/ {print \$1}'" || true)
  if [[ -z "${pids// /}" ]]; then
    echo "[p5-full] no large orphan spawn"
    return 0
  fi
  echo "[p5-full] TERM orphans: $pids"
  ssh -o BatchMode=yes "${HEAD_SSH:?set HEAD_SSH}" "for p in $pids; do kill -TERM \$p 2>/dev/null || true; done" || true
  sleep 12
}

graceful_stop() {
  echo "[p5-full] graceful stop"
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

find_stage_a_ckpt() {
  python3 - <<'PY'
from pathlib import Path
root = Path(__import__("os").environ["K3"]) / "experiments/checkpoints/root/k3-align-math-rl/p5-effort-tau2-24gpu/actor"
cands = sorted(root.glob("*/model.safetensors"), key=lambda p: p.stat().st_mtime)
if not cands:
    raise SystemExit(f"no Stage A ckpt under {root}")
print(cands[-1].parent)
PY
}

echo "[p5-full] preflight"
cleanup_host_orphans
avail=$(ssh -o BatchMode=yes "${HEAD_SSH:?set HEAD_SSH}" "free -g | awk '/^Mem:/{print \$7}'")
echo "[p5-full] MemAvailable≈${avail}GiB (min ${MIN_HOST_AVAIL_GB})"
if (( avail < MIN_HOST_AVAIL_GB )); then
  cleanup_host_orphans
  avail=$(ssh -o BatchMode=yes "${HEAD_SSH:?set HEAD_SSH}" "free -g | awk '/^Mem:/{print \$7}'")
  echo "[p5-full] MemAvailable≈${avail}GiB after cleanup"
  if (( avail < MIN_HOST_AVAIL_GB )); then
    echo "[p5-full] FAIL host RAM too low" >&2
    exit 1
  fi
fi
if [[ -x "$PREFLIGHT" ]]; then
  bash "$PREFLIGHT" ${HEAD_SSH} ${WORKER_HOSTS} || {
    graceful_stop
    bash "$PREFLIGHT" ${HEAD_SSH} ${WORKER_HOSTS}
  }
else
  echo "[p5-full] WARN missing preflight script"
fi

# --- Stage A ---
cfg_a=$K3/configs/p5_effort_tau2_24gpu.yaml
log_a=$K3/logs/p5_stage_a_$(date +%Y%m%d_%H%M%S).log
audit_a=$K3/experiments/checkpoints/root/k3-align-math-rl/p5-effort-tau2-24gpu/effort_audit.jsonl
rm -rf "$K3/tmp/name_resolve/k3-align-math-rl/p5-effort-tau2-24gpu" || true
rm -f "$audit_a" || true
mkdir -p "$(dirname "$audit_a")"

echo "[p5-full] === Stage A τ=2.0 steps=$EXPECTED_STEPS ==="
CONFIG="$cfg_a" LOG="$log_a" \
  K3_EFFORT_TAU=2.0 \
  K3_EFFORT_AUDIT_PATH="$audit_a" \
  K3_EFFORT_AUDIT_EVERY=32 \
  RAY_MEMORY_USAGE_THRESHOLD="$RAY_MEMORY_USAGE_THRESHOLD" \
  bash "$LAUNCH_P5" stage_a

wait_train "p5_stage_a" "$log_a" "$EXPECTED_STEPS" "$POLL"
graceful_stop
if [[ -x "$PREFLIGHT" ]]; then
  bash "$PREFLIGHT" ${HEAD_SSH} ${WORKER_HOSTS}
fi

ckpt_a=$(find_stage_a_ckpt)
echo "[p5-full] Stage A ckpt=$ckpt_a"

# --- Stage B: patch actor.path ---
cfg_b_src=$K3/configs/p5_effort_tau1_24gpu.yaml
cfg_b=$K3/tmp/p5_effort_tau1_24gpu_live.yaml
CKPT="$ckpt_a" SRC="$cfg_b_src" DST="$cfg_b" python3 - <<'PY'
from pathlib import Path
import os
src = Path(os.environ["SRC"]).read_text()
ckpt = os.environ["CKPT"]
if "PLACEHOLDER_STAGE_A" not in src:
    raise SystemExit("PLACEHOLDER_STAGE_A missing in Stage B config template")
out_lines = []
for line in src.splitlines(True):
    if "PLACEHOLDER_STAGE_A" in line and "path:" in line:
        key = line.split(":", 1)[0]
        out_lines.append(f"{key}: {ckpt}\n")
    else:
        out_lines.append(line)
Path(os.environ["DST"]).write_text("".join(out_lines))
print("wrote", os.environ["DST"], "actor.path=", ckpt)
PY

log_b=$K3/logs/p5_stage_b_$(date +%Y%m%d_%H%M%S).log
audit_b=$K3/experiments/checkpoints/root/k3-align-math-rl/p5-effort-tau1-24gpu/effort_audit.jsonl
rm -rf "$K3/tmp/name_resolve/k3-align-math-rl/p5-effort-tau1-24gpu" || true
rm -f "$audit_b" || true
mkdir -p "$(dirname "$audit_b")"

echo "[p5-full] === Stage B τ=1.0 steps=$EXPECTED_STEPS ==="
cleanup_host_orphans
CONFIG="$cfg_b" LOG="$log_b" \
  K3_EFFORT_TAU=1.0 \
  K3_EFFORT_AUDIT_PATH="$audit_b" \
  K3_EFFORT_AUDIT_EVERY=32 \
  RAY_MEMORY_USAGE_THRESHOLD="$RAY_MEMORY_USAGE_THRESHOLD" \
  bash "$LAUNCH_P5" stage_b

wait_train "p5_stage_b" "$log_b" "$EXPECTED_STEPS" "$POLL"
graceful_stop

echo "[p5-full] DONE"
echo "  stage_a_log=$log_a"
echo "  stage_b_log=$log_b"
echo "  stage_a_ckpt=$ckpt_a"
echo "  chain_log=$CHAIN_LOG"
