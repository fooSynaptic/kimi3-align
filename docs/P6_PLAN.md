# P6 approx MOPD Plan (MVP)

## Scope and Claim

- **Included**: relative to P2, **only add** on-policy distillation—use P1 final as the teacher and add the clipped log-ratio from K3 Eq.15 to the advantage as a **dense token reward**, while retaining the `gsm8k` outcome.
- **Excluded (MVP)**: nine experts / multi-domain × multi-effort routing, GRM tournament, λ-barrier, and Effort schedule.
- **Reporting**: `approx MOPD` (single teacher = P1); **not equivalent to** the paper's complete Multi-Teacher OPD.

Disable AReaL's built-in `distill_loss_weight` RKL (`0.0`) to avoid double-counting it with the Eq.15 reward.

```text
r_opd = clip(sg(log π_P1 − log π_θ), −R_max, R_max)
# v4+: optional pos-only → max(r_opd, 0)
A ← A_gsm8k + suffix_sum(α · r_opd / T)
```

## Optimization Path (Failure Record · Four Versions)

| Version | Trial | Change from Previous Version | Result | Assessment |
|---------|-------|------------------------------|--------|------------|
| **v1** | `p6-mopd-p1teacher-24gpu` | Instruct cold start · H=4 · α=0.05 · **no** length-norm · `weight_update=disk` | 151/200; train reward→0 after ~100 steps; disk 500 | **VOID** (scale collapse + synchronization) |
| **v2** | `…-v2` | length-norm on · α=0.5 · still disk | 19/200; GPFS shard-write tail latency causes >120s NFS handshake | **VOID** (synchronization) |
| **v3** | `…-v3` | **xccl** (allowed by separate actor/rollout GPUs) · otherwise identical to v2 | 200/200 · ~2.95h · no 500; boxed **72.27%** vs same-batch P1 78.52% / P2 75.39% (−6.25pp vs max) | **FAIL** (completed training, missed capability gate) |
| **v4** | `…-v4` | Diagnosis-driven: ① student **warm-started from P1 final** (ref=P1); ② `K3_MOPD_POS_ONLY=1` injects only `r_opd>0`; ③ retain xccl + len-norm α=0.5 | 200/200 · boxed **79.30%** (same-batch P1 78.52%, **+0.78pp**) · gate **pass** · see `P6_COMPARE.md` | **PASS** |

### v3 Diagnosis (Motivating v4)

- OPD audit: `mean_raw_log_ratio ≈ −6.1→−6.7`, `clip_frac≈0.73–0.75`, `mean_suffix≈−0.39` → with an Instruct cold start and H=4, student token log probabilities are systematically above the P1 teacher's, turning Eq.15 into a **stable negative bias**.
- Same-batch errors: P1✓P6✗ = 19, versus only 3 in the opposite direction; P6 outputs are longer on failed examples (~990 vs P1 ~741).
- Final training `task_reward`: P1 0.468 > P2 0.447 > P6 0.431, matching the boxed ordering.
- Mainline fact: none of P2–P5 surpasses P1 boxed; P6 is the only run that falls **below P2**.

### Fixed Items (Shared by v1–v4 Unless Noted in the Table)

| Item | Value |
|------|-------|
| Teacher | `p1-k25-sync-h0-full` final · FSDP colocated with actor |
| Recipe R / H | K2.5-style · `max_head_offpolicyness=4` · no λ / no Effort |
| \(R_{\max}\) / adv clip | 2.0 / 2.0 |
| Topology | 24 GPU: `vllm:d4` + `fsdp:d20` · batch=160 |
| Full run | 200 steps |

### v4 Knobs (Relative to v3)

| Item | v3 | v4 |
|------|----|----|
| Student / vLLM init | Instruct | **P1 final** |
| ref | Instruct | **P1 final** |
| OPD | Bidirectional clip | **pos-only** (negative log-ratio→0) |
| α / length-norm / sync | 0.5 / on / xccl | Same as v3 |
| trial | `…-v3` | `…-v4` |

## Code Entry Points

- [`pythonpath/mopd_boot.py`](../pythonpath/mopd_boot.py) — `K3_MOPD_POS_ONLY`
- [`pythonpath/p6_boot.py`](../pythonpath/p6_boot.py)
- Launch: `scripts/launch_p6_24gpu.sh {smoke|full}`
- Smoke / full: `scripts/run_p6_smoke.sh`, `scripts/run_p6_full.sh`
- Boxed: `scripts/run_boxed_p6.sh` → `docs/boxed_p6/`

## Success Criteria

1. Smoke: 2 steps; util≥50%; audit contains `opd` (for v4, expect `mean_suffix≥0` scale and auditable `pos_frac`).
2. Full: 200 steps; final boxed ≥ max(P2, P1)−2pp (same-batch N=256).
3. Reporting: use only approx MOPD and the paths above; do not claim complete K3 MOPD. v4 may be reported as “slightly above P1 in the same batch,” but must also state that pos_frac remains low and that this is not equivalent to fully activated strong distillation.
