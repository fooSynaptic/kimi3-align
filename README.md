# k3-align

Open-source one-factor-at-a-time ablation testbed built on ARealL for exploring RL hyperparameters inspired by ideas in K3 Section 4.1. This repository implements approximate variants, **not the original official K3 algorithm**.
> Inspired by Kimi K3 (arXiv:2607.24653) §4.1.

## What this is
- Same model/data/recipe base; **change one factor per phase** (P0–P6).
- Auditable **approximate** mechanisms: async head `H`, `approx λ-partial`, `approx Effort`, `approx MOPD` (single teacher).
- First-class recording of **negative results** (especially the full MOPD v1–v3 failure path).
- Checked-in **per-step reward / seq_len curves** for every completed trial.

## Contributions
1. **Ablation suite** — protocol + smoke/boxed evaluation gates for isolating §4.1-style RL knobs on a fully open stack.
2. **Auditable approximate implementations** — JSONL audit traces and explicit diffs against the mechanisms described in the K3 paper.
3. **OPD failure modes + remediation** — scale blow-up, NFS disk sync timeouts, negative log-ratio bias; mitigated via length normalization, XCCL communication, P1 warm-start, positive-only clipping.
4. **Multi-node async RL operational notes** — weight synchronization strategies when actors and rollout workers are not colocated; standardized evaluation protocol (same-batch MATH boxed, N=256).
5. **Negative-result artifacts** — VOID/FAILED trials preserved in documentation, rather than only retaining successful checkpoints.
6. **Transparent convergence tracking** — exportable StatsLogger time series + overlay SVGs stored under `docs/curves/`.

## Elevator pitch
**k3-align** is an open, one-factor ablation testbed for K3-§4.1-style RL knobs on AReaL: auditable *approximate* async-head, λ-stop, Effort gate, and single-teacher OPD—plus documented failure modes and training curves. It is **not an official K3 reproduction**.

> Terminology Convention:
> 1. Every abbreviation/acronym is expanded on its **first appearance** within the document.
> 2. Core reference baselines:
> - K3: Kimi K3 alignment research framework
> - ARealL: Async Reinforcement Learning for LLM alignment benchmark
> - OPD: Objective Policy Distillation

## Phase map (short)
All evaluations use same-batch MATH boxed with N=256 unless explicitly noted. Do not mix evaluation batches when comparing absolute pass-rate percentages.

| Phase | One change | Metrics | Status |
|-------|------------|---------|--------|
| P0 | Cold-start gate | boxed **65.625%** (168/256) | Instruct pass (no RL curve) |
| P1 | Sync + recipe R, H=0 | boxed **78.125%**; train reward 0.38→0.47 | Sync ceiling baseline |
| P2 | H=4 only | boxed **75.000%** (−3.125pp vs P1); throughput **+31.9%** steps/h | Throughput gain; boxed soft-failure |
| P3 | τ_R / mask sweep | boxed **75.781%** (τ_R=0; +0.78pp vs P2) | Performance comparable to P2 |
| P4 | approx λ=0.5 | boxed **74.219%** (−1.95pp vs same-batch P2 76.172%) | Mechanism validated |
| P5 | approx Effort | boxed **76.953%** (+0.78pp vs same-batch P2); generation length **−2.1%** | Improvement over P2 |
| P6 | approx MOPD | v3 boxed **72.266%**; **v4 79.297%** (+0.78pp vs same-batch P1 78.516%) | v1–v3 exhibit failure; **v4 achieves pass** |

| Document | Purpose |
|-----|------|
| [`docs/TECH_REPORT_P0_P6.md`](docs/TECH_REPORT_P0_P6.md) | Full technical design & complete results |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | **Claim ↔ artifact mapping; Tier-1 / Tier-2 reproducibility scope** |
| [`docs/curves/COMPARE.md`](docs/curves/COMPARE.md) | Training convergence curves (raw CSV + rendered SVG) |
| [`LICENSE`](LICENSE) | MIT License |

## Convergence (preview)
Primary tracking metric: `ppo_actor/task_reward/avg`.

![mainline](docs/curves/figures/overlay_task_reward_mainline.svg)
![p6](docs/curves/figures/overlay_task_reward_p6.svg)

Rebuild visualizations from trial `main.log`:
```bash
python3 scripts/extract_train_curves.py \
  --log path/to/<trial>/main.log --trial <trial> --out docs/curves/
python3 scripts/plot_train_curves.py --curves-dir docs/curves
```

## Repo layout
```text
configs/          # per-phase Hydra YAML (+ smoke subdirectory) — see configs/README.md
pythonpath/       # runtime boot hooks (λ / Effort / MOPD / recipe R)
scripts/          # experiment entrypoints — see scripts/README.md
  lib/env.sh      # shared environment variables (K3_ROOT / SAO_ROOT)
  verify.sh       # Tier-1 claim validator C0–C7
  rebuild_curves.sh
  launch_*.sh     # Ray distributed training launchers
  run_boxed_*.sh  # same-batch MATH boxed evaluation gates
docs/             # technical report, reproducibility docs, curves, boxed evaluation logs
```

## Quickstart (outline)
1. **Runtime stack (SAO + AReaL).** Training and evaluation rely on the AReaL + vLLM + FSDP runtime from [Single-rollout-async-Optimization](https://github.com/fooSynaptic/Single-rollout-async-Optimization) (SAO).
> k3-align does **not** vendor this dependency stack.
Follow SAO’s Quickstart to bootstrap AReaL:
```bash
INSTALL=1 INFERENCE_BACKEND=vllm bash scripts/bootstrap_areal.sh
```
After bootstrap export required environment variables:
```bash
export SAO_ROOT=/path/to/Single-rollout-async-Optimization
export AREAL_REPO=$SAO_ROOT/vendor/AReaL
export VENV=$SAO_ROOT/tmp/phase2-runtime
export MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507
export DATA_PATH=$SAO_ROOT/data/gsm8k_hard
export HEAD_SSH=head-alias HEAD_HOST=<ray-head-ip>
export WORKER_SPECS='worker-a:8:0,1,2,3,4,5,6,7 worker-b:8:0,1,2,3,4,5,6,7'
```
YAML configuration resolves variables via Hydra `oc.env` (`K3_ROOT`, `MODEL_PATH`, `DATA_PATH`, `AREAL_REPO`, `VENV`, `SITE`). Variable definitions: [`scripts/README.md`](scripts/README.md) §0. Path keys within YAML: [`configs/README.md`](configs/README.md).

2. **Workflow: Smoke → full training → boxed eval → curve rendering → verification**
```bash
export K3_ROOT=$PWD
# SAO_ROOT / AREAL_REPO / VENV / MODEL_PATH / DATA_PATH / HEAD_SSH / HEAD_HOST / WORKER_SPECS must already be defined
bash scripts/run_p6_smoke.sh          # sanity smoke test example
bash scripts/launch_p6_24gpu.sh full
bash scripts/run_boxed_p6.sh
bash scripts/rebuild_curves.sh
bash scripts/verify.sh
```
All launch wrappers require `HEAD_SSH`, `HEAD_HOST`, and `WORKER_SPECS`; no hardcoded cluster hostnames are baked in. Checked-in documentation removes all internal hostnames.

> This repository ships no standalone `requirements.txt`. All Python dependencies reside within the SAO + AReaL virtual environment. Published results are collected on multi-GPU **Hopper (96GB HBM)** hardware with actor/rollout workers split across nodes.

## Citation
This project draws inspiration from the training ideas described in Kimi K3 §4.1.
- Cite the original K3 paper for the official algorithm formulation.
- Cite **this repository** only when referencing the open ablation protocol, approximate mechanisms, failure analysis notes, and experimental artifacts.

```bibtex
@misc{k3-align,
  title={k3-align: Open One-Factor Ablation Testbed for K3-Style RL Mechanisms on AReaL},
  author={fooSynaptic},
  year={2026},
  howpublished={GitHub repository, \url{https://github.com/fooSynaptic/k3-align}},
  note={Not an official K3 reproduction}
}
