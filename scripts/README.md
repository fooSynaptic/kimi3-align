# Scripts

Flat entrypoints + `lib/` helpers. Read this map before running anything.

```text
scripts/
  README.md                 ← you are here
  lib/
    env.sh                  shared roots / env (source me)
    wait_train.sh           poll train logs for step completion
    sample_gpu_util.sh      smoke GPU util sampler / gate
  verify.sh                 Tier-1 claim check (C0–C7)
  verify_reported_results.py
  rebuild_curves.sh         extract all trials → plot → verify
  extract_train_curves.py
  plot_train_curves.py
  math_boxed_probe.py       MATH boxed N=256 probe (eval core)
  e1_math_rl_train.py       AReaL PPO entry
  e0_math_sft_train.py      optional SFT entry
  prepare_math_sft.py
  launch_e1_rl_20gpu.sh     Ray bring-up + train (core launcher)
  launch_p4_16gpu.sh | launch_p5_24gpu.sh | launch_p6_24gpu.sh
  launch_e0_sft_20gpu.sh
  run_*_smoke.sh / run_*_full.sh / run_*_chain.sh
  run_boxed_*.sh            same-batch capability gates
```

## 0. Environment

Every bash wrapper should start with:

```bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/env.sh"
```

| Variable | Meaning | Default |
|----------|---------|---------|
| `K3_ROOT` / `K3` | this repo | auto: checkout root |
| `SAO_ROOT` / `SAO` | [SAO](https://github.com/fooSynaptic/Single-rollout-async-Optimization) checkout (AReaL + venv bundle) | sibling path if present; else **required** |
| `AREAL_REPO` | AReaL checkout | `$SAO_ROOT/vendor/AReaL` or `$SAO_ROOT/repo/AReaL` |
| `VENV` | python env with vLLM/AReaL | `$SAO_ROOT/tmp/phase2-runtime` |
| `MODEL_PATH` | Instruct / eval model | unset |
| `DATA_PATH` | `gsm8k_hard` etc. | unset |
| `CK_ROOT` | checkpoints root | `$K3/experiments/checkpoints/root/k3-align-math-rl` |
| `LOG_ROOT` | trial `main.log` root | `$K3/experiments/logs/root/k3-align-math-rl` |
| `HEAD_SSH` / `HEAD_HOST` / `WORKER_SPECS` | Ray topology | **required** for train launch (no baked-in hosts) |
| `MODEL_PATH` | Instruct / eval model | **required** for train |
| `DATA_PATH` | `gsm8k_hard` | **required** for train |

Always export paths and Ray topology before train/eval.

```bash
export K3_ROOT=/path/to/k3-align
export SAO_ROOT=/path/to/Single-rollout-async-Optimization
export MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507
export DATA_PATH=/path/to/gsm8k_hard
export HEAD_SSH=head-alias HEAD_HOST=<ray-head-ip>
export WORKER_SPECS='worker-a:8:0,1,2,3,4,5,6,7 worker-b:8:0,1,2,3,4,5,6,7'
```

## 1. Verify & curves (no / light GPU)

| Script | Role |
|--------|------|
| `./scripts/verify.sh` | Claims C0–C7 vs `docs/` artifacts |
| `./scripts/rebuild_curves.sh` | Re-extract curves from `LOG_ROOT/*/main.log`, plot, verify |
| `extract_train_curves.py` | One trial → CSV/JSON |
| `plot_train_curves.py` | JSON → SVG + `docs/curves/COMPARE.md` |

```bash
./scripts/verify.sh
LOG_ROOT=/path/to/logs/.../k3-align-math-rl ./scripts/rebuild_curves.sh
```

## 2. Eval (boxed MATH)

| Script | Role |
|--------|------|
| `math_boxed_probe.py` | Core probe (seed=1, N=256, greedy) |
| `run_boxed_probe.sh` | P0 / Instruct vs SFT |
| `run_boxed_p1_p2_p3.sh` | batch_p1_p2_p3 (C0–C3) |
| `run_boxed_p4.sh` | batch_p4 (C4) |
| `run_boxed_p5.sh` | batch_p5 (C5) |
| `run_boxed_p6.sh` | batch_p6_v4 (C6) |

Protocol details: `docs/REPRODUCIBILITY.md`.

## 3. Train launch

| Script | Role |
|--------|------|
| `launch_e1_rl_20gpu.sh` | **Core**: Ray head/workers + `e1_math_rl_train.py` |
| `launch_p4_16gpu.sh` | P4 λ wrapper → e1 |
| `launch_p5_24gpu.sh` | P5 Effort wrapper → e1 |
| `launch_p6_24gpu.sh` | P6 MOPD wrapper → e1 |
| `launch_e0_sft_20gpu.sh` | Optional E0 SFT |
| `run_p*_smoke.sh` / `run_p*_full.sh` | Smoke / formal orchestration |
| `run_p1_p2_chain.sh` / `run_p1_full_then_p2.sh` / `run_p2_only.sh` | Mainline chains |
| `run_p3_sweep.sh` | P3 τ/mask sweep |
| `run_smoke_chain.sh` / `run_mainline_queue.sh` | Queue helpers |

Phase knobs live in `configs/`; boot hooks in `pythonpath/`.

## 4. Typical flow

```text
smoke  →  full train (launch_p*)  →  run_boxed_*  →  rebuild_curves  →  verify
```

1. Point `K3_ROOT` / `SAO_ROOT` (and topology env).
2. Smoke: `bash scripts/run_p6_smoke.sh` (example).
3. Full: `bash scripts/launch_p6_24gpu.sh full`.
4. Same-batch boxed: `bash scripts/run_boxed_p6.sh`.
5. Curves + claims: `bash scripts/rebuild_curves.sh && bash scripts/verify.sh`.

## 5. Lib helpers

| File | Role |
|------|------|
| `lib/env.sh` | Roots, `k3_require_sao`, `k3_mkdirs` |
| `lib/wait_train.sh` | `wait_train <label> <log> <steps>` |
| `lib/sample_gpu_util.sh` | Background util sampling for smokes |

Hardware notes in comments use **Hopper (96GB HBM)**; host aliases in launch defaults are operator-local — override for your cluster.
