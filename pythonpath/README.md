# pythonpath — train boot hooks

Imported via `PYTHONPATH` on the driver / RPC workers (see `configs/*.yaml` `scheduling_spec` / env blocks).

| Module | Phase | Role |
|--------|-------|------|
| `k25_boot.py` | P1+ | Recipe R: τ_R · (Δlog π)² via `K25_TAU_LOG_RATIO_SQ` |
| `p4_boot.py` / `lambda_barrier.py` | P4 | approx λ-partial stop-admit (`K3_LAMBDA_*`) |
| `effort_boot.py` / `effort_reward.py` | P5 | approx Effort length gate (`K3_EFFORT_*`) |
| `mopd_boot.py` / `p6_boot.py` | P6 | approx MOPD / OPD (`K3_MOPD_*`, `K3_MOPD_POS_ONLY`) |

Entry train script: `scripts/e1_math_rl_train.py`.  
Do not claim these are full K3 λ / Effort / nine-expert MOPD.
