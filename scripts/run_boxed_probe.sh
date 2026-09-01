#!/usr/bin/env bash
# MATH boxed-acc gate: base Instruct vs E0 r2 final ckpt (2 GPUs sequential).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
k3_require_sao
k3_mkdirs

BASE=${MODEL_PATH:?set MODEL_PATH to Instruct checkpoint or HF id}
E0=${E0_CKPT:-$K3/experiments/checkpoints/root/k3-e0-math-sft/qwen3-4b-instruct-20gpu-r2/default/epoch2epochstep45globalstep137}
OUT=$K3/docs/e0_boxed_probe
N=${N:-256}
GPU=${GPU:-0}

export PATH="$VENV/bin:/usr/local/bin:$PATH"
export PYTHONPATH="$SITE:${PYTHONPATH:-}"
export HOME=$K3/tmp/home
export TMPDIR=$K3/tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR
export HF_HOME="${HF_HOME:-$SAO/tmp/hf-home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$SAO/tmp/hf-datasets}"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export CUDA_VISIBLE_DEVICES=$GPU
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

mkdir -p "$OUT" "$K3/logs" "$TMPDIR" "$HOME" "$HF_HOME" "$HF_DATASETS_CACHE"

run_one() {
  local tag=$1 model=$2
  local log=$K3/logs/boxed_probe_${tag}_$(date +%Y%m%d_%H%M%S).log
  echo "[probe] $tag -> $log"
  "$VENV/bin/python" -u "$K3/scripts/math_boxed_probe.py" \
    --model "$model" --tag "$tag" --out "$OUT/${tag}.json" --n "$N" \
    2>&1 | tee "$log"
}

run_one base_instruct "$BASE"
run_one e0_r2_epoch2 "$E0"

"$VENV/bin/python" - <<'PY'
import json
from pathlib import Path
out = Path(__import__("os").environ["K3"]) / "docs" / "e0_boxed_probe"
base = json.loads((out / "base_instruct.json").read_text())["summary"]
e0 = json.loads((out / "e0_r2_epoch2.json").read_text())["summary"]
cmp = {
    "n": base["n"],
    "base_acc": base["acc"],
    "e0_acc": e0["acc"],
    "delta": e0["acc"] - base["acc"],
    "gate": "pass" if e0["acc"] + 1e-9 >= base["acc"] - 0.02 else "warn_drop",
}
(out / "compare.json").write_text(json.dumps(cmp, indent=2))
print(json.dumps(cmp, indent=2))
PY
