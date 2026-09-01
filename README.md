# k3-align

Open **one-factor ablation lab** for K3-§4.1-style RL knobs on AReaL (vLLM rollout + FSDP actor).

> Inspired by Kimi K3 (arXiv:2607.24653) §4.1. **Not a K3 reproduction.**

| Doc | Role |
|-----|------|
| [`docs/TECH_REPORT_P0_P6.md`](docs/TECH_REPORT_P0_P6.md) | Full design + results |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | **Claim→artifact map; Tier-1/2 repro** |
| [`docs/curves/COMPARE.md`](docs/curves/COMPARE.md) | Train convergence curves (CSV + SVG) |
| [`LICENSE`](LICENSE) | MIT |

## What this is

- Same model/data/recipe base; **change one factor per phase** (P0–P6).
- Auditable **approximate** mechanisms: async head `H`, `approx λ-partial`, `approx Effort`, `approx MOPD` (single teacher).
- First-class **negative results** (especially MOPD v1–v3 failure path).
- Checked-in **per-step reward / seq_len curves** for every full trial.

## Claims (contributions)

1. **Ablation suite** — protocol + smoke/boxed gates for isolating §4.1-style knobs on an open stack.
2. **Auditable approx implementations** — JSONL audits and explicit Diff vs paper namesakes.
3. **OPD failure modes + repair** — scale blow-up, disk NFS sync timeouts, negative log-ratio bias; repair via length-norm, xccl, P1 warm-start, pos-only.
4. **Multi-node async RL ops notes** — weight-sync choice when actor/rollout are not colocated; eval conventions (same-batch MATH boxed N=256).
5. **Negative-result artifacts** — VOID/FAIL trials kept in docs, not only the winning checkpoint.
6. **Transparent convergence** — extractable StatsLogger series + overlay SVGs under `docs/curves/`.

## Elevator pitch

**k3-align** is an open, one-factor ablation lab for K3-§4.1-style RL knobs on AReaL: auditable *approximate* async-head, λ-stop, Effort gate, and single-teacher OPD—plus documented failure modes and train curves. It is **not** a K3 reproduction.

## Phase map (short)

Same-batch MATH boxed N=256 unless noted. Do not mix batches when comparing absolute %.

| Phase | One change | Metrics | Status |
|-------|------------|---------|--------|
| P0 | Cold-start gate | boxed **65.625%** (168/256) | Instruct pass (no RL curve) |
| P1 | Sync + recipe R, H=0 | boxed **78.125%**; train reward 0.38→0.47 | Sync ceiling |
| P2 | H=4 only | boxed **75.000%** (−3.125pp vs P1); throughput **+31.9%** steps/h | Throughput pass; boxed soft-fail |
| P3 | τ_R / mask sweep | boxed **75.781%** (τ_R=0; +0.78pp vs P2) | ≈P2 |
| P4 | approx λ=0.5 | boxed **74.219%** (−1.95pp vs same-batch P2 76.172%) | Mechanism pass |
| P5 | approx Effort | boxed **76.953%** (+0.78pp vs same-batch P2); gen len **−2.1%** | Pass vs P2 |
| P6 | approx MOPD | v3 boxed **72.266%**; **v4 79.297%** (+0.78pp vs same-batch P1 78.516%) | v1–v3 fail; **v4 pass** |

## Convergence (preview)

Main metric: `ppo_actor/task_reward/avg`.

![mainline](docs/curves/figures/overlay_task_reward_mainline.svg)

![p6](docs/curves/figures/overlay_task_reward_p6.svg)

Rebuild from a trial `main.log`:

```bash
python3 scripts/extract_train_curves.py \
  --log path/to/<trial>/main.log --trial <trial> --out docs/curves/
python3 scripts/plot_train_curves.py --curves-dir docs/curves
```

## Repo layout

```text
configs/          # per-phase YAML (+ smoke/) — see configs/README.md
pythonpath/       # boot hooks (λ / Effort / MOPD / recipe R)
scripts/          # entrypoints — see scripts/README.md
  lib/env.sh      # shared K3_ROOT / SAO_ROOT
  verify.sh       # Tier-1 claims C0–C7
  rebuild_curves.sh
  launch_*.sh     # Ray + train
  run_boxed_*.sh  # same-batch MATH gates
docs/             # report, REPRODUCIBILITY, curves, boxed
```

## Quickstart (outline)

1. **Runtime (SAO + AReaL).** Train/eval use the AReaL + vLLM + FSDP env from [Single-rollout-async-Optimization](https://github.com/fooSynaptic/Single-rollout-async-Optimization) (SAO). k3-align does not vendor that stack. Follow SAO’s Quick start through the AReaL bootstrap (`INSTALL=1 INFERENCE_BACKEND=vllm bash scripts/bootstrap_areal.sh`), then export:

   ```bash
   export SAO_ROOT=/path/to/Single-rollout-async-Optimization
   export AREAL_REPO=$SAO_ROOT/vendor/AReaL          # after SAO bootstrap
   export VENV=$SAO_ROOT/tmp/phase2-runtime          # or the env used for INSTALL=1
   export MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507
   export DATA_PATH=$SAO_ROOT/data/gsm8k_hard
   export HEAD_SSH=head-alias HEAD_HOST=<ray-head-ip>
   export WORKER_SPECS='worker-a:8:0,1,2,3,4,5,6,7 worker-b:8:0,1,2,3,4,5,6,7'
   ```

   YAML keys resolve via Hydra `oc.env` (`K3_ROOT`, `MODEL_PATH`, `DATA_PATH`, `AREAL_REPO`, `VENV`, `SITE`). Variable table: [`scripts/README.md`](scripts/README.md) §0. YAML path keys: [`configs/README.md`](configs/README.md).

2. **Smoke → full → boxed → curves → verify:**

```bash
export K3_ROOT=$PWD
# SAO_ROOT / AREAL_REPO / VENV / MODEL_PATH / DATA_PATH / HEAD_SSH / HEAD_HOST / WORKER_SPECS as above
bash scripts/run_p6_smoke.sh          # example
bash scripts/launch_p6_24gpu.sh full
bash scripts/run_boxed_p6.sh
bash scripts/rebuild_curves.sh
bash scripts/verify.sh
```

Launch wrappers require `HEAD_SSH`, `HEAD_HOST`, and `WORKER_SPECS` (no baked-in cluster hosts). Checked-in **docs** omit internal hostnames.

This repo has no `requirements.txt`; Python deps live in the SAO AReaL venv. Hardware for published numbers: multi-GPU **Hopper (96GB HBM)** (actor/rollout split).

## Citation / relation to K3

This project is **inspired by** Kimi K3 §4.1 training ideas. Cite the K3 paper for the original methods; cite this repo only for the open ablation protocol, approximate mechanisms, failure notes, and artifacts.
