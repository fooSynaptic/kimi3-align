# P1 vs P2 Assessment

Generated: 2026-08-08

Comparison logs:
- P1: `logs/p1_k25_full_20260807_130934.log` · trial `p1-k25-sync-h0-full` · H=0
- P2: `logs/p2_k25_async_full_20260808_005557.log` · trial `p2-k25-async-h4` · H=4

Success criteria (design): relative to P1, improve throughput or tail latency; avoid an accuracy/reward collapse; shift staleness to the right.

| Metric | P1 (H=0) | P2 (H=4) | Assessment |
|--------|----------|----------|------------|
| Completed steps | 200 | 200 | Pass |
| Wall time | 6.09 h | 4.62 h | P2 is faster |
| sec/step | 109.6 | 83.1 | **Throughput +31.9%** |
| steps/h | 32.8 | 43.3 | Favorable |
| `staleness_theta` mean/p50 | 0.0 / 0.0 | 3.94 / 4.0 | **Shifted right to H** |
| Reward mean, full run | 0.428 | 0.417 | −2.7% |
| Reward mean, final 50 steps | 0.457 | 0.440 | **−3.8%** (no collapse) |
| gen tok/s p50 | 3127 | 3275 | Slight increase |

## Conclusion

**P2 PASS (throughput / train reward)**

1. Async-head provides a clear throughput gain (about +32% steps/h).
2. Staleness shifts from 0 to about 4, consistent with `max_head_offpolicyness=4`.
3. Late-stage reward is about 3.8% below P1, which does not constitute a collapse (threshold: relative decrease <10%).

## Boxed Recheck (2026-08-10)

N=256 · final checkpoint · see `docs/boxed_p1_p2_p3/COMPARE.md`

| Tag | Acc |
|-----|-----|
| Instruct | 65.625% |
| P1 final | **78.125%** |
| P2 final | **75.000%** |
| P3-τ0 final | 75.781% |

- P2 − P1 = **−3.125pp** → a **soft fail** against the design criterion of “no accuracy drop beyond ~2pp” (still well above Instruct: +9.4pp).
- P3-τ0 − P2 = **+0.78pp** → effectively tied with P2; τ=0 does not hurt accuracy.
- Training `rollout/reward` supports the “no collapse” interpretation. When boxed evaluation is treated as the final capability measure, **synchronous P1 is stronger**; P2 trades a small amount of accuracy for throughput.
