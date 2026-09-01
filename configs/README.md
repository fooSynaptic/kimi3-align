# Configs

Phase YAMLs target AReaL. Paths are **Hydra `oc.env` lookups** — set these before launch:

| Env | YAML use |
|-----|----------|
| `K3_ROOT` | `fileroot`, audits, `pythonpath` |
| `MODEL_PATH` | Instruct / student `actor.path` (most arms) |
| `DATA_PATH` | `train_dataset.path` (`gsm8k_hard`) |
| `AREAL_REPO` / `VENV` / `SITE` | RPC `PYTHONPATH` / `PATH` |
| `P1_CKPT` | P6 student/ref (defaults in `scripts/lib/env.sh`) |
| `P5_STAGE_A_CKPT` | P5-B actor path (set after Stage A) |
| `SFT_DATA_PATH` | E0 SFT data |

```bash
export K3_ROOT=/path/to/k3-align
export SAO_ROOT=/path/to/Single-rollout-async-Optimization
export MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507
export DATA_PATH=/path/to/gsm8k_hard
```

Runtime: [SAO](https://github.com/fooSynaptic/Single-rollout-async-Optimization) AReaL bootstrap (`INSTALL=1 INFERENCE_BACKEND=vllm bash scripts/bootstrap_areal.sh`).

Canonical mainline YAMLs: `p1_k25_sync_h0_20gpu.yaml`, `p2_k25_async_h4_20gpu.yaml`, `p3_h4_*.yaml`, `p4_h4_lambda05_16gpu.yaml`, `p5_effort_tau*.yaml`, `p6_mopd_p1teacher_24gpu.yaml`.  
Legacy (do not use): `p1_sync_h0_20gpu.yaml`, `p2_async_h4_20gpu.yaml`, `e1_partial_rl_20gpu.yaml`.

Smoke configs live under `smoke/` (few steps, same knobs). GPU counts in filenames are topology hints.
