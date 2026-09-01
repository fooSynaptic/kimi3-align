# K3-align Experiment Design

Based on Kimi K3 (arXiv:2607.24653) §4.1. Principle: keep the model, data, and topology fixed, and change only one factor at a time.

**Locked**

| Item | Value |
|------|-------|
| Model | `Qwen3-4B-Instruct-2507` |
| Topology | Historical mainline: 20 GPU; **P5+ default: 24 GPU** (`vllm:d4` + `fsdp:d20`) |
| Data (RL) | `gsm8k_hard` |
| G/K | `n_samples=8` |
| Recipe R | **K2.5-style**: group mean advantage `(r−r̄)` + token ratio mask `[α,β]` + `τ(log-ratio)²` (not claimed as SAO GRPO+DIS) |
| α / β / τ | `0.2` / `5.0` / `0.01` (placeholders; the paper does not disclose exact values) |
| Steps | 200 / arm |

**Reporting**: the `max_head_offpolicyness` ablation is not equivalent to K3 approx λ-partial; this recipe is not equivalent to complete K2.5/K3 (it lacks the λ-sandbox, MuonClip, and exact hyperparameters).

---

## Phase Table

| Phase | Name | Only Change | Success Criteria |
|-------|------|-------------|------------------|
| **P0** | Cold-start gate | Initialization: Instruct **or** SFT that passes the gate | MATH boxed N≥256; SFT must not fall below Instruct−2pp, otherwise discard it |
| **P1** | Sync baseline | `max_head_offpolicyness=0`, recipe R, initialization that passed P0 | Complete the specified steps; record accuracy/reward/staleness≈0 |
| **P2** | Async-head | **Only** `max_head_offpolicyness=4` | Relative to P1: improve throughput or tail latency; keep accuracy above the collapse threshold; shift staleness right |
| **P3** | αβτ sweep | Fix H; sweep mask / τ | Stable interval; boxed ≈ P2 |
| **P4** | approx λ-partial | Stop-admit at λ=0.5 | Auditable λ scheduling logs; boxed within −2pp of P2 |
| **P5** | approx Effort | `b0`, `τ_E` schedule | Lower length without an accuracy collapse |
| **P6** | approx MOPD | Single-teacher Eq.15 | v1–v3 fail path documented; v4 pass vs P1 |

**Default mainline**: P0 = Instruct (do not use the degraded E0) → P1 → P2.

**Legacy trials** (retired): `partial-grpo-dis-e0warm`, `p1-sync-h0` (SAO/DIS framing).

---

# Smoke Chain (Orchestration First)

Entry point: `scripts/run_smoke_chain.sh` (P0 N=8 → P1 2-step h0 → P2 2-step h4).
The workload matches the production knobs (`batch=160` / `n_samples=8` / `max_concurrent=32` / `gpu_memory_utilization=0.55`) and only shortens the step count. The peak GPU utilization gate is ≥50% (SM or memory utilization, averaged over active GPUs).
Report: `docs/smoke_chain_report.md`. All smoke checks must pass before full experiments begin.

## Current Progress

| Phase | Status |
|-------|--------|
| Smoke | **pass** · production knobs + util≥50% · see `docs/smoke_chain_report.md` |
| P0 | **pass** · Instruct boxed **65.625%** (168/256) |
| P1 | **pass** · `p1-k25-sync-h0-full` · 200 steps · final boxed **78.125%** |
| P2 | **pass (throughput) / soft fail (boxed)** · `p2-k25-async-h4` · throughput +32% · stale→4 · final boxed **75.0%** (−3.1pp vs P1, beyond the −2pp gate; +9.4pp vs Instruct) · see `docs/P1_P2_COMPARE.md`, `docs/boxed_p1_p2_p3/COMPARE.md` |
| P3 | **pass** · fixed H=4 · τ0 / τ0.05 / mask-tight all run for 200 steps · τ0 final boxed **75.78%** (≈P2, +0.8pp) · see `docs/boxed_p1_p2_p3/COMPARE.md` |
| P4 | **pass (mechanism)** · Scheme B λ=0.5 · 16 GPU · 200 steps · final-50 reward=0.432 (≈P2) · audit 200×pause · stale→0 · see `docs/P4_COMPARE.md`, `docs/P4_PLAN.md` |
| P5 | **pass (accuracy) / slight length reduction** · approx Effort · 24 GPU · A τ=2.0 + B τ=1.0, 100 steps each · boxed **76.95%** vs same-batch P2 76.17% (+0.78pp) · mean_gen_len 585.7 vs 598.4 (−2.1%) · B over_budget 97% · see `docs/P5_COMPARE.md`, `docs/P5_PLAN.md` |
| P6 | **v1–v3 FAIL** · **v4 PASS** boxed **79.30%** (+0.78pp vs same-batch P1) · see `docs/P6_COMPARE.md` / `P6_PLAN.md` |

**Boxed gate (N=256)**: Instruct 65.6% · P1 78.1% (same-batch recheck 78.5%) · P2 75.0% (rechecks 75.4–76.2%) · P3-τ0 75.8% · P4 74.2% (−1.95pp vs same-batch P2, pass) · **P5 77.0%** (+0.78pp vs same-batch P2, pass) · P6 v3 72.3% (fail) · **P6 v4 79.3%** (**+0.78pp** vs same-batch P1, pass; the first mainline run to surpass P1).

**Internal run notes**: the recipe has switched to K2.5-style; the launch phase wipes the current trial's `name_resolve`; the host must have sufficient MemAvailable before smoke tests (clean up orphaned `ppid=1` spawn processes). P5+ training defaults to **24 GPU**.
