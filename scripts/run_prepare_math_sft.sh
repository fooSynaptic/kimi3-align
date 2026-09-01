#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
k3_require_sao
k3_mkdirs

export HF_HOME=$SAO/tmp/hf-home
export HF_DATASETS_CACHE=$SAO/tmp/hf-datasets
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"
"$VENV/bin/python" "$K3/scripts/prepare_math_sft.py" \
  --output "$K3/data/math_sft_chat" \
  --model "${MODEL_PATH:?set MODEL_PATH}" \
  --max-length 2048 \
  --hf-cache "$HF_HOME"
