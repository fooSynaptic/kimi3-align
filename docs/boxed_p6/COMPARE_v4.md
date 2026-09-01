# Boxed MATH · P6 v4 vs P1/P2

N=256 · greedy · max_tokens=1024 · same batch

| Tag | Acc | Correct/N | mean_gen_len |
|-----|-----|-----------|--------------|
| `p1_sync_h0_final` | **0.7852** | 201.0/256 | 540.53125 |
| `p2_h4_final` | **0.7578** | 194.0/256 | 598.08984375 |
| `p6_mopd_v4_final` | **0.7930** | 203.0/256 | 522.859375 |
| `p6_mopd_v3_final` (prev batch ref) | **0.7227** | 185.0/256 | 610.15234375 |

- max(P1,P2) acc: `0.78515625`
- P6 v4 − max(P1,P2): `0.78125` pp · gate `pass` (need ≥ −2pp)

