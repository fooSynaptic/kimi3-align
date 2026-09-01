# P5 vs P2 Assessment (approx Effort · Global b0=256 · τ=2.0→1.0)

Generated: 2026-08-11

Comparison:
- P2: `p2-k25-async-h4` · 20 GPU · H=4 · no Effort · see `docs/P1_P2_COMPARE.md`
- P5: Stage A `p5-effort-tau2-24gpu` (100 steps, τ=2.0) → Stage B `p5-effort-tau1-24gpu` (100 steps, τ=1.0, warm-started from A) · **24 GPU** · `docs/P5_PLAN.md`

Claim: `approx Effort` (hard gate \(T(y)>\tau\cdot b_0\) → reward=−1). **Not equivalent to** the paper's complete per-prompt \(b_0(x)\) / multi-expert suite.

| Metric | P2 (20gpu) | P5-A τ=2.0 | P5-B τ=1.0 | Notes |
|--------|------------|------------|------------|-------|
| Completed steps | 200 | 100 | 100 | Pass |
| Wall time | 4.62 h | ~2.11 h | ~2.12 h | ~4.2 h of training in total |
| sec/step | 83.1 | ~76.8 | ~76.9 | Slightly faster with 24 GPUs |
| `n_seqs` / step | 1280 | 1280 | 1280 | Full batch |
| Train `correct_n_seqs` mean | — | **508.8** (39.8%) | **201.6** (15.8%) | Correct-sequence count collapses in B because over-budget samples receive −1 |
| Train weighted seq_len | — | **505** | **487** | Slightly shorter during training |
| Effort audit over_budget | — | **0 / 16625 (0%)** | **106419 / 109728 (97.0%)** | In A, threshold=512=max_new_tokens, so the gate almost never triggers |
| Audit mean \(T\) | — | 418 | 446 | Both p50 values reach the 512 cap |

## Boxed (N=256 · greedy · max_tokens=1024)

| Tag | Acc | Correct | mean_gen_len |
|-----|-----|---------|--------------|
| P2 final (same batch) | **76.17%** | 195/256 | **598.4** |
| P5-B final | **76.95%** | 197/256 | **585.7** |

P5 − P2 accuracy = **+0.78pp** · gate **pass** (≥ P2−2pp).  
Length: 585.7 vs 598.4 (**−2.1%**), satisfying the requirement to decrease relative to P2, though by only a small margin.

## Conclusion

**P5 PASS (accuracy gate + slight length reduction) · substantial mechanism side effects**

1. The hard gate is auditable: A applies almost no penalties (budget equals the generation cap); in B, 97% of samples receive reward=−1, consistent with a loose-to-tight schedule.
2. Boxed capability does not collapse and is slightly above same-batch P2.
3. The length gain is weak: boxed outputs are only about 13 tokens shorter; incorrect training outputs remain long (~532).
4. Stage B's correct training sequence rate falls from ~40% to ~16%, a direct consequence of assigning a hard −1 to over-budget samples. **Do not use train correctness as a proxy for boxed results.**

Reporting constraint: report only approx Effort and the values above; do not claim the complete K3 Reasoning Effort method.

To reduce length materially, decouple the \(b_0\) or τ schedule from `max_new_tokens` (for example, set A's threshold below 512), or implement per-prompt \(b_0(x)\).
