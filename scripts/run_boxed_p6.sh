#!/usr/bin/env bash
# Boxed MATH for P6 v4 final vs P1/P2 (same batch N=256). Prefer 00/02/03.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
k3_require_sao
k3_mkdirs


CK=$K3/experiments/checkpoints/root/k3-align-math-rl
OUT=$K3/docs/boxed_p6
N=${N:-256}
HOST_A=${HOST_A:?set HOST_A}
HOST_B=${HOST_B:?set HOST_B}
HOST_C=${HOST_C:?set HOST_C}

P1=$CK/p1-k25-sync-h0-full/actor/epoch4epochstep15globalstep199
P2=$CK/p2-k25-async-h4/actor/epoch4epochstep15globalstep199
P6=$CK/p6-mopd-p1teacher-24gpu-v4/actor/epoch4epochstep15globalstep199

for p in "$P1" "$P2" "$P6"; do
  [[ -f "$p/model.safetensors" ]] || { echo "[fatal] missing $p/model.safetensors" >&2; exit 1; }
done

mkdir -p "$OUT" "$K3/logs" "$K3/tmp/home" "$K3/tmp"
export PATH="$VENV/bin:/usr/local/bin:$PATH"

JOBS=(
  "p1_sync_h0_final|$P1|$HOST_A|0"
  "p2_h4_final|$P2|$HOST_B|0"
  "p6_mopd_v4_final|$P6|$HOST_C|0"
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
export HF_HUB_CACHE=$SAO/tmp/hf-home/hub
export HF_ENDPOINT=\${HF_ENDPOINT:-https://hf-mirror.com}
# Prefer cache; allow mirror fetch if cache missing.
export HF_HUB_OFFLINE=\${HF_HUB_OFFLINE:-0}
export HF_DATASETS_OFFLINE=\${HF_DATASETS_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=1
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
out = Path(__import__("os").environ["K3"]) / "docs" / "boxed_p6"
tags = ["p1_sync_h0_final", "p2_h4_final", "p6_mopd_v4_final"]
rows = {}
for tag in tags:
    p = out / f"{tag}.json"
    raw = json.loads(p.read_text()) if p.exists() else {"error": "missing"}
    s = raw.get("summary", raw)
    rows[tag] = {
        "tag": tag,
        "n": s.get("n"),
        "correct": s.get("correct"),
        "acc": s.get("acc"),
        "mean_gen_len": s.get("mean_gen_len") or s.get("mean_completion_len"),
        "error": raw.get("error"),
    }
p1, p2, p6 = rows["p1_sync_h0_final"], rows["p2_h4_final"], rows["p6_mopd_v4_final"]
accs = [r.get("acc") for r in (p1, p2, p6)]
gate = None
delta_vs_max = None
if all(a is not None for a in accs):
    base = max(p1["acc"], p2["acc"])
    delta_vs_max = (p6["acc"] - base) * 100
    gate = delta_vs_max >= -2.0
v3 = None
v3p = out / "p6_mopd_v3_final.json"
if v3p.exists():
    s = json.loads(v3p.read_text()).get("summary", {})
    v3 = {"tag": "p6_mopd_v3_final", "acc": s.get("acc"), "correct": s.get("correct"), "n": s.get("n"), "mean_gen_len": s.get("mean_gen_len")}
compare = {
    "p1": p1,
    "p2": p2,
    "p6": p6,
    "p6_v3_ref": v3,
    "max_p1_p2_acc": max((a for a in (p1.get("acc"), p2.get("acc")) if a is not None), default=None),
    "delta_pp_vs_max_p1_p2": delta_vs_max,
    "gate_pass": gate,
}
(out / "compare_v4.json").write_text(json.dumps(compare, indent=2) + "\n")
(out / "compare.json").write_text(json.dumps(compare, indent=2) + "\n")
md = [
    "# Boxed MATH · P6 v4 vs P1/P2",
    "",
    "N=256 · greedy · max_tokens=1024 · same batch",
    "",
    "| Tag | Acc | Correct/N | mean_gen_len |",
    "|-----|-----|-----------|--------------|",
]
for tag, r in rows.items():
    acc = r.get("acc")
    acc_s = f"**{acc:.4f}**" if isinstance(acc, float) else str(acc)
    md.append(
        f"| `{tag}` | {acc_s} | {r.get('correct')}/{r.get('n')} | {r.get('mean_gen_len')} |"
    )
if v3 and v3.get("acc") is not None:
    md.append(
        f"| `p6_mopd_v3_final` (prev batch ref) | **{v3['acc']:.4f}** | {v3.get('correct')}/{v3.get('n')} | {v3.get('mean_gen_len')} |"
    )
md += [
    "",
    f"- max(P1,P2) acc: `{compare['max_p1_p2_acc']}`",
    f"- P6 v4 − max(P1,P2): `{delta_vs_max}` pp · gate `{'pass' if gate else 'fail'}` (need ≥ −2pp)",
    "",
]
(out / "COMPARE_v4.md").write_text("\n".join(md) + "\n")
(out / "COMPARE.md").write_text("\n".join(md) + "\n")
print(json.dumps(compare, indent=2))
PY

exit "$fail"
