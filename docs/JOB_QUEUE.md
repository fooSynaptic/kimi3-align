# K3-align job queue

Updated: 2026-08-12

| Job | Status | Trial / notes |
|-----|--------|---------------|
| P0–P5 | **done** | see `EXPERIMENT_DESIGN.md` |
| P6 smoke | **PASS** | 2 steps · OPD events OK |
| P6 full v1 | **VOID** | 151/200 · α=0.05 without /T · reward→0 · disk 500 |
| P6 full v2 | **VOID** | 19/200 · len-norm α=0.5 · disk NFS 120s timeout |
| P6 full v3 | **FAIL (boxed)** | 200/200 xccl · boxed 72.27% vs P1 78.52% (−6.25pp) |
| P6 full v4 | **PASS** | 200/200 · boxed **79.30%** (+0.78pp vs P1) · `P6_COMPARE.md` · log `p6_full_20260811_235637.log` |
| Future training topology | **24 GPU** | prefer a clean 3-node split when head cards are clean |
