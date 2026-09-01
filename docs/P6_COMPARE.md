# P6 v4 Results (approx MOPD · P1 Warm Start · pos-only)

Generated: 2026-08-12

## Configuration Recap

| Item | v4 |
|------|----|
| Changes from v3 | Student/ref **warm-started from P1 final**; `K3_MOPD_POS_ONLY=1`; retain xccl + len-norm α=0.5 |
| Trial | `p6-mopd-p1teacher-24gpu-v4` |
| Topology | 24 GPU · `vllm:d4` + `fsdp:d20` · H=4 · recipe R |
| Log | `logs/p6_full_20260811_235637.log` |

## Training

| Metric | v4 | v3 (control) |
|--------|----|--------------|
| Completed | **200/200** | 200/200 |
| Wall time | ~2.81 h (10113 s) | ~2.95 h |
| HTTP 500 | **0** | 0 |
| `task_reward`, final 20 | **0.489** | 0.431 |
| `task_reward`, first 20 | **0.486** | 0.377 |
| `seq_len`, final 20 | 456.5 | 476.5 |
| `update_weights` | ~7.1 s (stable xccl) | ~6.7 s |

OPD audit (1000 events, `pos_only=true`):

| Stage | mean_suffix | pos_frac | raw log-ratio |
|-------|-------------|----------|---------------|
| early | **+0.0027** | 0.051 | −6.48 |
| late | **+0.0022** | 0.047 | −6.65 |

Compared with v3's `mean_suffix≈−0.39`, negative advantage has been eliminated. Effective positive OPD remains sparse (~5% of tokens), but no longer imposes a systematic penalty.

## Boxed (N=256 · same batch · greedy)

See `docs/boxed_p6/COMPARE_v4.md`.

| Tag | Acc | Correct | mean_gen_len |
|-----|-----|---------|--------------|
| P1 sync | 78.52% | 201/256 | 540.5 |
| P2 async H=4 | 75.78% | 194/256 | 598.1 |
| **P6 v4** | **79.30%** | **203/256** | **522.9** |
| P6 v3 (previous-batch reference) | 72.27% | 185/256 | 610.2 |

- Gate: ≥ max(P1,P2)−2pp → floor 76.52% → **pass** (actual result: **+0.78pp vs P1**)
- vs v3: **+7.03pp**
- Length: **3.3%** shorter than P1 and **12.6%** shorter than P2

## Assessment

**P6 v4 PASS (capability gate + slight gain over P1)**

1. The run completes fully, xccl produces no 500 errors, and reward is above v3 while close to or slightly above historical P1 training levels.  
2. In same-batch boxed evaluation, it **surpasses P1 for the first time** (+0.78pp / +2 problems); P2–P5 and P6 v3 did not.  
3. Removing v3's stable negative bias with pos-only is critical; the P1 warm start provides a high-capability starting point.  
4. Effective OPD remains sparse (pos_frac≈5%). The gain is better characterized as “H=4 RL near P1 without being dragged down by negative OPD,” not as “strong distillation is fully activated.”

**Reporting scope:** `approx MOPD` (single teacher=P1, pos-only) — the paper's nine-expert routing and domain GRM stack is a separate, larger system.
