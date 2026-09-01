# K3-align E0 — MATH light SFT

## Locked

- Scope: **B** light SFT (MATH chat trajectories)
- Model: `Qwen3-4B-Instruct-2507`
- Nodes: 3-node Ray · **20×Hopper (96GB HBM)**
- Backend: AReaL `SFTTrainer` · `fsdp:d20p1t1` · Ray
- WS: ``<cluster-root>/...``
- Data: `data/math_sft_chat` · train 7490 / test 4999 · max_length 2048

## Runs

- Smoke 5 step OK · loss ≈ 2.20 → 1.31
- r1 (`qwen3-4b-instruct-20gpu`): 1 epoch / 46 steps · loss ≈ 2.21 → 0.60 · **no HF ckpt** (`freq_steps=100` never hit)
- r2: `total_train_epochs=3` · `saver.freq_epochs=1` · new trial `qwen3-4b-instruct-20gpu-r2`

## Gate (before E1)

1. Train finishes without NaN / host OOM
2. Valid loss decreases vs step 0
3. Freeze HF ckpt under `experiments/.../checkpoints`
4. Optional: short MATH boxed-acc probe vs base Instruct

## Commands

```bash
bash scripts/run_prepare_math_sft.sh
SMOKE_STEPS=5 bash scripts/launch_e0_sft_20gpu.sh
bash scripts/launch_e0_sft_20gpu.sh   # r2: 3 epochs + save each epoch
```
