# P4 vs P2 Assessment (λ=0.5 Soft Barrier · Scheme B)

Generated: 2026-08-10

Comparison:
- P2: `p2-k25-async-h4` · 20 GPU · H=4 · no λ-barrier · see `docs/P1_P2_COMPARE.md`
- P4: `p4-h4-lambda05-16gpu` · **16 GPU** · H=4 · λ=0.5 · batch=168 · `docs/P4_PLAN.md`

| Metric | P2 (20gpu) | P4 (16gpu · λ0.5) | Notes |
|--------|------------|-------------------|-------|
| Completed steps | 200 | 200 | Pass |
| Wall time | 4.62 h | **6.04 h** | Fewer GPUs + λ admission pause |
| sec/step | 83.1 | **108.7** | Throughput is not directly comparable |
| Reward mean, full run | 0.417 | 0.414 | Tied |
| Reward, final 50 steps | 0.440 | **0.432** | −1.8% vs P2 (no collapse) |
| `staleness_theta` last50 | ~3.94 | **0.0** | The λ soft barrier narrows the window, pushing staleness back toward synchronous behavior |
| `n_seqs` / step | 1280 | **864** | ≈ DP alignment after the λ window + inflight work (~108×8) |
| λ audit | — | pause/train/reset/resume **200 each** | The gate triggers every step |

## Conclusion

**P4 PASS (Scheme B mechanism + no reward collapse)**

1. The λ soft barrier is auditable and triggers every step (200× pause/train).
2. Late-stage reward is only about 1.8% below P2, which does not constitute a collapse.
3. Side effects: the effective batch is smaller (864 vs 1280 sequences), and `staleness_theta→0` (closer to “wait for enough λ completion, then train” than a pure async-head profile). The 16-GPU run has a longer wall time than the 20-GPU P2 run, so **this run cannot prove or disprove a throughput benefit from λ**.

A fair throughput comparison requires the same topology (recommended: **24 GPU**) and a new control arm.

## Boxed (N=256)

| Tag | Acc |
|-----|-----|
| P2 final (same-batch recheck) | **76.17%** |
| P4 final | **74.22%** |

P4 − P2 = **−1.95pp** · gate **pass** (within the −2pp threshold). See `docs/boxed_p4/COMPARE.md` for details.
