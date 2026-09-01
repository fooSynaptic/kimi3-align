# Boxed MATH gate · P1 / P2 / P3-tau0

N=256 · greedy · max_tokens=1024 · same batch

| Tag | Acc | Correct/N |
|-----|-----|-----------|
| `base_instruct` | **0.6562** | 168.0/256 |
| `p1_h0_final` | **0.7812** | 200.0/256 |
| `p2_h4_final` | **0.7500** | 192.0/256 |
| `p3_tau0_final` | **0.7578** | 194.0/256 |

## Deltas
- P1 − base: `0.125`
- P2 − P1: `-0.03125` · gate `fail_collapse`
- P3-tau0 − P2: `0.007812` · gate `pass`
- P2 − base: `0.09375`
- P3 − base: `0.101562`
