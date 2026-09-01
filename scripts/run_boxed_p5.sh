#!/usr/bin/env bash
# Boxed MATH for P5 Stage B final (+ P2 reuse). Eval on HOST_A/HOST_B (see .env).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
k3_require_sao
k3_mkdirs


CK=$K3/experiments/checkpoints/root/k3-align-math-rl
OUT=$K3/docs/boxed_p5
N=${N:-256}
HOST_A=${HOST_A:?set HOST_A}
HOST_B=${HOST_B:?set HOST_B}

P2=$CK/p2-k25-async-h4/actor/epoch4epochstep15globalstep199
P5=$(python3 - <<'PY'
from pathlib import Path
root = Path(__import__("os").environ["K3"]) / "experiments/checkpoints/root/k3-align-math-rl/p5-effort-tau1-24gpu/actor"
cands = sorted(root.glob("*/model.safetensors"), key=lambda p: p.stat().st_mtime)
if not cands:
    raise SystemExit(f"missing P5 ckpt under {root}")
print(cands[-1].parent)
PY
)

for p in "$P2" "$P5"; do
  [[ -f "$p/model.safetensors" ]] || { echo "[fatal] missing $p/model.safetensors" >&2; exit 1; }
done

mkdir -p "$OUT" "$K3/logs" "$K3/tmp/home" "$K3/tmp"
export PATH="$VENV/bin:/usr/local/bin:$PATH"

JOBS=(
  "p2_h4_final|$P2|$HOST_A|0"
  "p5_effort_tau1_final|$P5|$HOST_B|0"
)

run_remote() {
  local tag=$1 model=$2 host=$3 gpu=$4
  local log=$K3/logs/boxed_${tag}_$(date +%Y%m%d_%H%M%S).log
  local outj=$OUT/${tag}.json
  echo "[boxed] start $tag host=$host gpu=$gpu model=$model -> $log"
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$host" "bash -s" <<EOF
set -euo pipefail
export PATH=$VENV/bin:/usr/local/bin:\$PATH
export PYTHONPATH=$SITE:\${PYTHONPATH:-}
export HOME=$K3/tmp/home
export TMPDIR=$K3/tmp TEMP=$K3/tmp TMP=$K3/tmp
export HF_HOME=$SAO/tmp/hf-home
export HF_DATASETS_CACHE=$SAO/tmp/hf-datasets
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=$gpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
mkdir -p "\$HOME" "\$TMPDIR" "$OUT" "$K3/logs"
$VENV/bin/python -u $K3/scripts/math_boxed_probe.py \\
  --model '$model' --tag '$tag' --out '$outj' --n $N \\
  >'$log' 2>&1
EOF
  echo "[boxed] done $tag"
}

pids=()
for spec in "${JOBS[@]}"; do
  IFS='|' read -r tag model host gpu <<<"$spec"
  run_remote "$tag" "$model" "$host" "$gpu" &
  pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

$VENV/bin/python - <<'PY'
import json
from pathlib import Path
out = Path(__import__("os").environ["K3"]) / "docs" / "boxed_p5"
rows = []
for tag in ["p2_h4_final", "p5_effort_tau1_final"]:
    p = out / f"{tag}.json"
    raw = json.loads(p.read_text()) if p.exists() else {"error": "missing"}
    s = raw.get("summary", raw)
    rows.append({
        "tag": tag,
        "n": s.get("n"),
        "correct": s.get("correct"),
        "acc": s.get("acc"),
        "mean_gen_len": s.get("mean_gen_len") or s.get("mean_completion_len"),
    })
p2 = next(r for r in rows if r["tag"] == "p2_h4_final")
p5 = next(r for r in rows if r["tag"] == "p5_effort_tau1_final")
delta = None
if p2.get("acc") is not None and p5.get("acc") is not None:
    delta = (p5["acc"] - p2["acc"]) * 100
compare = {"p2": p2, "p5": p5, "delta_pp": delta, "gate_pass": (delta is not None and delta >= -2.0)}
(out / "compare.json").write_text(json.dumps(compare, indent=2) + "\n")
print(json.dumps(compare, indent=2))
PY

exit "$fail"
