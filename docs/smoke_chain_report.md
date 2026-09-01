# Smoke chain report

Generated: 2026-08-08T00:13:46+00:00
Chain log: ``<cluster-root>/...``
Efficiency: formal knobs (batch=160, n_samples=8, max_concurrent=32, gpu_memory_utilization=0.55); util gate ≥50%

| Phase | Result | Notes |
|-------|--------|-------|
| P0 boxed N=8 | PASS | instruct-only probe |
| P1 sync h0 (2 steps) | PASS | trial `p1-k25-smoke` · util peak (SM% MEM%)=`head_sm=100 head_mem=56 single_sm=100 single_mem=56` |
| P2 async h4 (2 steps) | PASS | trial `p2-k25-smoke` · util peak (SM% MEM%)=`head_sm=100 head_mem=56 single_sm=100 single_mem=57` |

P3+ : skipped (not implemented)

## Logs
- P0: ``<cluster-root>/...``
- P1: ``<cluster-root>/...``
- P2: ``<cluster-root>/...``
- P1 util: ``<cluster-root>/...``
- P2 util: ``<cluster-root>/...``
