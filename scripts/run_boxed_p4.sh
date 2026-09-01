#!/usr/bin/env bash
# Boxed MATH for P4 final (+ optional P2 reuse). Eval on HOST_A/HOST_B (see .env).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
k3_require_sao
k3_mkdirs


CK=$K3/experiments/checkpoints/root/k3-align-math-rl
OUT=$K3/docs/boxed_p4
N=${N:-256}
HOST_A=${HOST_A:?set HOST_A}
HOST_B=${HOST_B:?set HOST_B}

P2=$CK/p2-k25-async-h4/actor/epoch4epochstep15globalstep199
P4=$CK/p4-h4-lambda05-16gpu/actor/epoch4epochstep23globalstep199

for p in "$P2" "$P4"; do
  [[ -f "$p/model.safetensors" ]] || { echo "[fatal] missing $p/model.safetensors" >&2; exit 1; }
done

mkdir -p "$OUT" "$K3/logs" "$K3/tmp/home" "$K3/tmp"
export PATH="$VENV/bin:/usr/local/bin:$PATH"

JOBS=(
  "p2_h4_final|$P2|$HOST_A|0"
  "p4_lambda05_final|$P4|$HOST_B|0"
)

run_remote() {
  local tag=$1 model=$2 host=$3 gpu=$4
  local log=$K3/logs/boxed_${tag}_$(date +%Y%m%d_%H%M%S).log
  local outj=$OUT/${tag}.json
  echo "[boxed] start $tag host=$host gpu=$gpu -> $log"
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
out = Path(__import__("os").environ["K3"]) / "docs" / "boxed_p4"
rows = []
for tag in ["p2_h4_final", "p4_lambda05_final"]:
    p = out / f"{tag}.json"
    s = json.loads(p.read_text())["summary"] if p.exists() else {"tag": tag, "error": "missing"}
    if "summary" in s:
        s = s["summary"]
    rows.append({"tag": tag, "n": s.get("n"), "correct": s.get("correct"), "acc": s.get("acc"), **{k:s.get(k) for k in ()}})
p2 = next(r for r in rows if r["tag"]=="p2_h4_final")
p4 = next(r for r in rows if r["tag"]=="p4_lambda05_final")
delta = None
if "acc" in p2 and "acc" in p4:
    delta = round(p4["acc"] - p2["acc"], 6)
gate = "pass" if delta is not None and p4["acc"] + 1e-9 >= p2["acc"] - 0.02 else "fail_collapse"
cmp = {"rows": rows, "delta_p4_vs_p2": delta, "p4_vs_p2_gate": gate}
(out / "compare.json").write_text(json.dumps(cmp, indent=2))
md = ["# Boxed MATH · P4 vs P2", "", f"N={p2.get('n')} · hosts 02/03", "",
      "| Tag | Acc | Correct/N |", "|-----|-----|-----------|"]
for r in rows:
    md.append(f"| `{r['tag']}` | **{r.get('acc', 'ERR'):.4f}** | {r.get('correct')}/{r.get('n')} |" if "acc" in r else f"| `{r['tag']}` | ERR | — |")
md += ["", f"- P4 − P2: `{delta}` · gate `{gate}`", ""]
(out / "COMPARE.md").write_text("\n".join(md))
print(json.dumps(cmp, indent=2))
print("wrote", out / "COMPARE.md")
PY

if (( fail )); then
  echo "[boxed] SOME FAILED" >&2
  exit 1
fi
echo "[boxed] ALL PASS — see $OUT/COMPARE.md"
