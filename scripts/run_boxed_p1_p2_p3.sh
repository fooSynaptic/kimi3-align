#!/usr/bin/env bash
# Boxed MATH gate for P1 / P2 / P3-tau0 finals (+ Instruct baseline).
# Prefer eval hosts via HOST_A / HOST_B (leave the train head free).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
k3_require_sao
k3_mkdirs


BASE=${MODEL_PATH:?set MODEL_PATH to Instruct checkpoint or HF id}
CK=$K3/experiments/checkpoints/root/k3-align-math-rl
OUT=$K3/docs/boxed_p1_p2_p3
N=${N:-256}
# hosts that may hold GPUs (00 intentionally unused by default)
HOST_A=${HOST_A:?set HOST_A}
HOST_B=${HOST_B:?set HOST_B}

export PATH="$VENV/bin:/usr/local/bin:$PATH"

mkdir -p "$OUT" "$K3/logs" "$K3/tmp/home" "$K3/tmp"

P1=$CK/p1-k25-sync-h0-full/actor/epoch4epochstep15globalstep199
P2=$CK/p2-k25-async-h4/actor/epoch4epochstep15globalstep199
P3=$CK/p3-h4-tau0/actor/epoch4epochstep15globalstep199

for p in "$P1" "$P2" "$P3"; do
  [[ -f "$p/model.safetensors" ]] || { echo "[fatal] missing $p/model.safetensors" >&2; exit 1; }
done

# tag|model|ssh_host|local_gpu
JOBS=(
  "base_instruct|$BASE|$HOST_A|0"
  "p1_h0_final|$P1|$HOST_A|1"
  "p2_h4_final|$P2|$HOST_B|0"
  "p3_tau0_final|$P3|$HOST_B|1"
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

# launch all in parallel (00 unused)
pids=()
for spec in "${JOBS[@]}"; do
  IFS='|' read -r tag model host gpu <<<"$spec"
  run_remote "$tag" "$model" "$host" "$gpu" &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

# summarize
$VENV/bin/python - <<'PY'
import json
from pathlib import Path
out = Path(__import__("os").environ["K3"]) / "docs" / "boxed_p1_p2_p3"
order = ["base_instruct", "p1_h0_final", "p2_h4_final", "p3_tau0_final"]
rows = []
for tag in order:
    p = out / f"{tag}.json"
    if not p.exists():
        rows.append({"tag": tag, "error": "missing"})
        continue
    s = json.loads(p.read_text())
    # support both summary-wrapped and flat
    if "summary" in s:
        s = s["summary"]
    rows.append({
        "tag": tag,
        "n": s.get("n"),
        "correct": s.get("correct"),
        "acc": s.get("acc"),
    })

base = next((r for r in rows if r["tag"] == "base_instruct" and "acc" in r), None)
p1 = next((r for r in rows if r["tag"] == "p1_h0_final" and "acc" in r), None)
p2 = next((r for r in rows if r["tag"] == "p2_h4_final" and "acc" in r), None)
p3 = next((r for r in rows if r["tag"] == "p3_tau0_final" and "acc" in r), None)

def delta(a, b):
    if a is None or b is None or "acc" not in a or "acc" not in b:
        return None
    return round(b["acc"] - a["acc"], 6)

cmp = {
    "n": base["n"] if base else None,
    "rows": rows,
    "delta_p1_vs_base": delta(base, p1),
    "delta_p2_vs_p1": delta(p1, p2),
    "delta_p3_vs_p2": delta(p2, p3),
    "delta_p2_vs_base": delta(base, p2),
    "delta_p3_vs_base": delta(base, p3),
}
# soft gates from experiment design spirit
if p1 and p2 and "acc" in p1 and "acc" in p2:
    # P2 should not collapse vs P1 beyond ~2pp absolute (stricter than train-reward proxy)
    cmp["p2_vs_p1_gate"] = "pass" if p2["acc"] + 1e-9 >= p1["acc"] - 0.02 else "fail_collapse"
if p2 and p3 and "acc" in p2 and "acc" in p3:
    cmp["p3_vs_p2_gate"] = "pass" if p3["acc"] + 1e-9 >= p2["acc"] - 0.02 else "fail_worse"

(out / "compare.json").write_text(json.dumps(cmp, indent=2))

md = ["# Boxed MATH gate · P1 / P2 / P3-tau0", "", f"N={cmp['n']} · greedy · max_tokens=1024 · same batch", "",
      "| Tag | Acc | Correct/N |", "|-----|-----|-----------|"]
for r in rows:
    if "acc" in r:
        md.append(f"| `{r['tag']}` | **{r['acc']:.4f}** | {r['correct']}/{r['n']} |")
    else:
        md.append(f"| `{r['tag']}` | ERR | — |")
md += ["", "## Deltas",
       f"- P1 − base: `{cmp.get('delta_p1_vs_base')}`",
       f"- P2 − P1: `{cmp.get('delta_p2_vs_p1')}` · gate `{cmp.get('p2_vs_p1_gate')}`",
       f"- P3-tau0 − P2: `{cmp.get('delta_p3_vs_p2')}` · gate `{cmp.get('p3_vs_p2_gate')}`",
       f"- P2 − base: `{cmp.get('delta_p2_vs_base')}`",
       f"- P3 − base: `{cmp.get('delta_p3_vs_base')}`",
       ""]
(out / "COMPARE.md").write_text("\n".join(md))
print(json.dumps(cmp, indent=2))
print("wrote", out / "COMPARE.md")
PY

if (( fail )); then
  echo "[boxed] SOME JOBS FAILED" >&2
  exit 1
fi
echo "[boxed] ALL PASS — see $OUT/COMPARE.md"
exit 0
