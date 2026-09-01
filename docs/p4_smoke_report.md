# P4 smoke report (16 GPU · λ=0.5)

Generated: 2026-08-10

| Tag | batch | concurrent | gpu_mem | max_seqs | sec/step | util peak | Result |
|-----|-------|------------|---------|----------|----------|-----------|--------|
| `base` | 168 | 32 | 0.55 | 64 | **≈101.4** | head_sm=96 head_mem=46 | PASS |
| `conc48_mem65` | 168 | 48 | 0.65 | 96 | ≈106.5 | head_sm=50 head_mem=52 | PASS |

**Selected for full:** `base` (faster sec/step + higher head SM util).

Audit gate: `pause` / `train` / `reset` / `resume` observed (λ=0.5 · target_groups=84 · DP-aligned train batches of 120 groups after inflight drain).

## Notes

- FSDP `d12` requires `#groups % 12 == 0` → batch **168** (not 160).
- Scheme B wait fix: collect results 1-at-a-time under λ barrier (upstream full-batch wait deadlocks).
- Claim: approx λ-partial — see `docs/P4_PLAN.md`.
