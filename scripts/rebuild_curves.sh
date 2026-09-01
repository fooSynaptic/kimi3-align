#!/usr/bin/env bash
# Rebuild docs/curves from trial main.logs, then plot + verify claims.
#
# Usage:
#   LOG_ROOT=/path/to/.../k3-align-math-rl bash scripts/rebuild_curves.sh
#   # or with defaults from env.sh / sibling layout:
#   bash scripts/rebuild_curves.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"

OUT="${CURVES_DIR}"
LOG_ROOT="${LOG_ROOT}"
mkdir -p "$OUT"

TRIALS=(
  p1-k25-sync-h0-full
  p2-k25-async-h4
  p3-h4-tau0
  p3-h4-tau005
  p3-h4-mask-tight
  p4-h4-lambda05-16gpu
  p5-effort-tau2-24gpu
  p5-effort-tau1-24gpu
  p6-mopd-p1teacher-24gpu
  p6-mopd-p1teacher-24gpu-v2
  p6-mopd-p1teacher-24gpu-v3
  p6-mopd-p1teacher-24gpu-v4
)

missing=0
for t in "${TRIALS[@]}"; do
  log="$LOG_ROOT/$t/main.log"
  if [[ ! -f "$log" ]]; then
    echo "[skip] missing $log" >&2
    missing=$((missing + 1))
    continue
  fi
  python3 "$SCRIPT_DIR/extract_train_curves.py" --log "$log" --trial "$t" --out "$OUT"
done

python3 "$SCRIPT_DIR/plot_train_curves.py" --curves-dir "$OUT"
python3 "$SCRIPT_DIR/verify_reported_results.py" --update-hashes
python3 "$SCRIPT_DIR/verify_reported_results.py"

echo "[ok] curves in $OUT (skipped_missing=$missing)"
