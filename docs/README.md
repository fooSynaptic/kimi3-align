# Docs index

| Path | Content |
|------|---------|
| [`TECH_REPORT_P0_P6.md`](TECH_REPORT_P0_P6.md) | End-to-end report in English (alignment, knobs, curves, boxed, claims) |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Claim IDs C0–C7; Tier-1 verify; Tier-2 retrain protocol |
| [`manifests/results_manifest.json`](manifests/results_manifest.json) | Machine-readable expects + artifact hashes |
| [`curves/COMPARE.md`](curves/COMPARE.md) | Train convergence overlays + per-trial SVGs |
| [`curves/README.md`](curves/README.md) | How to extract/rebuild curves |
| [`EXPERIMENT_DESIGN.md`](EXPERIMENT_DESIGN.md) | Phase design table |
| [`P1_P2_COMPARE.md`](P1_P2_COMPARE.md) / [`P4_COMPARE.md`](P4_COMPARE.md) / [`P5_COMPARE.md`](P5_COMPARE.md) / [`P6_COMPARE.md`](P6_COMPARE.md) | Per-phase compare write-ups |
| [`P4_PLAN.md`](P4_PLAN.md) / [`P5_PLAN.md`](P5_PLAN.md) / [`P6_PLAN.md`](P6_PLAN.md) | Mechanism plans |
| [`boxed_p1_p2_p3/`](boxed_p1_p2_p3/) / [`boxed_p4/`](boxed_p4/) / [`boxed_p5/`](boxed_p5/) / [`boxed_p6/`](boxed_p6/) | MATH boxed N=256 artifacts |
| [`JOB_QUEUE.md`](JOB_QUEUE.md) | Historical queue notes (ops) |

Public docs avoid internal hostnames and absolute cluster paths.

Verify locally:

```bash
python3 scripts/verify_reported_results.py
```

