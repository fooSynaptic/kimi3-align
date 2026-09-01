# Reproducibility

This repo separates **two tiers**. Conclusions in `TECH_REPORT_P0_P6.md` are written so both tiers are checkable.

## Tier 1 — Artifact consistency (no GPU)

Proves the report numbers still match checked-in boxed JSON + train curves.

```bash
./scripts/verify.sh
# equivalent: python3 scripts/verify_reported_results.py
```

Rebuild curves from trial logs (then verify):

```bash
LOG_ROOT=/path/to/.../k3-align-math-rl ./scripts/rebuild_curves.sh
```

Script map: [`scripts/README.md`](../scripts/README.md).

Machine-readable map of claims → trials → artifacts:

- [`docs/manifests/results_manifest.json`](manifests/results_manifest.json)

| Claim | Statement (short) | Primary artifacts |
|-------|-------------------|-------------------|
| **C0** | Instruct P0 gate 65.625% | `p0_gate/gate.json`, `boxed_p1_p2_p3/` |
| **C1** | P1 sync ceiling ~78% | `boxed_p1_p2_p3/compare.json` |
| **C2** | H=4 ≈+32% throughput; −3.125pp boxed vs P1 | `P1_P2_COMPARE.md`, same boxed batch |
| **C3** | P3 τ_R=0 ≈ P2 | `boxed_p1_p2_p3/compare.json` |
| **C4** | P4 λ=0.5 −1.95pp vs same-batch P2 · pass | `boxed_p4/compare.json` |
| **C5** | P5 Effort +0.78pp vs same-batch P2; len −2.1% | `boxed_p5/compare.json` |
| **C6** | P6 v3 fail / v4 +0.78pp vs same-batch P1 | `boxed_p6/compare_v4.json` + curves |
| **C7** | §6 curve table matches `docs/curves/*.json` | curve JSON summaries + CSV hashes |

### Same-batch eval context

Absolute boxed % is defined relative to a fixed eval batch (`batch_*` in the manifest). Comparing absolute % across batches mixes different probe runs and checkpoint context — each claim therefore cites one batch id.

Examples of P1 across batches (both valid, serving different claim ids):

| Batch | P1 acc | Used for |
|-------|--------|----------|
| `batch_p1_p2_p3` | 78.125% (200/256) | C1/C2/C3 |
| `batch_p6_v4` | 78.516% (201/256) | C6 deltas vs P6-v4 |

## Tier 2 — End-to-end retrain + reeval (cluster)

Runtime: install [SAO](https://github.com/fooSynaptic/Single-rollout-async-Optimization) at the pinned revision, bootstrap AReaL, then set `SAO_ROOT` / `AREAL_REPO` / `VENV` as in the root README Quickstart.

**Upstream pins** (branch + commit): [`manifests/upstream_pins.json`](manifests/upstream_pins.json)

| Layer | Branch | Commit |
|-------|--------|--------|
| SAO | `main` | `ac2728149041b5f15ceb75413f467ce0659179cf` |
| AReaL | detached | `3cf0dfbd2b0fbeabd6977184980e189d1567747a` (SAO `bootstrap_areal.sh`) |

```bash
git clone https://github.com/fooSynaptic/Single-rollout-async-Optimization "$SAO_ROOT"
git -C "$SAO_ROOT" checkout ac2728149041b5f15ceb75413f467ce0659179cf
cd "$SAO_ROOT" && INSTALL=1 INFERENCE_BACKEND=vllm bash scripts/bootstrap_areal.sh
```

### Fixed protocol

**Train (defaults unless a phase overrides):**

| Knob | Value |
|------|-------|
| seed | `1` |
| model | `Qwen3-4B-Instruct-2507` |
| data | `gsm8k_hard` |
| steps | 200 / arm (P5 A/B staged; see notes) |
| recipe R | G=8, mask `[0.2,5.0]`, τ_R=`0.01`, Adam `1e-6` |
| train gen | `max_new_tokens=512`, `temperature=1.0` |

**Eval (all boxed claims):**

```bash
python3 scripts/math_boxed_probe.py \
  --model <ckpt_or_hf> \
  --tag <tag> \
  --out docs/boxed_<phase>/<tag>.json \
  --n 256 --seed 1 --max-tokens 1024
```

- Dataset: `DigitalLearningGmbH/MATH-lighteval` / `test`
- Sampling: greedy (`temperature=0`)
- Grader: `math_verify` + `last_boxed` fallback (same as script)

### Per-phase one-factor configs

| Phase | Config | Trial name | One change |
|-------|--------|------------|------------|
| P1 | `configs/p1_k25_sync_h0_20gpu.yaml` | `p1-k25-sync-h0-full` | RL + R + H=0 |
| P2 | `configs/p2_k25_async_h4_20gpu.yaml` | `p2-k25-async-h4` | H=0→4 |
| P3 | `configs/p3_h4_tau0_20gpu.yaml` etc. | `p3-h4-*` | τ_R / mask |
| P4 | `configs/p4_h4_lambda05_16gpu.yaml` | `p4-h4-lambda05-16gpu` | λ=0.5 stop-admit |
| P5 | `configs/p5_effort_tau{2,1}_24gpu.yaml` | `p5-effort-tau*` | Effort gate |
| P6 | `configs/p6_mopd_p1teacher_24gpu.yaml` | `p6-mopd-*-v{1..4}` | OPD knobs (see `P6_PLAN.md`) |

Launch wrappers: `scripts/launch_*.sh`, `scripts/run_*_full.sh`. Override cluster paths via env — see `configs/README.md`.

### Checkpoints

Final actor dirs follow:

```text
experiments/checkpoints/root/k3-align-math-rl/<trial>/actor/*globalstep199*
```

P6-v4 additionally requires P1 final as student/ref warm-start and `K3_MOPD_POS_ONLY=1`.

### Curves after retrain

```bash
python3 scripts/extract_train_curves.py \
  --log path/to/<trial>/main.log --trial <trial> --out docs/curves/
python3 scripts/plot_train_curves.py --curves-dir docs/curves
python3 scripts/verify_reported_results.py --update-hashes
```

Tier-2 success = same-batch deltas and gates match claim expect fields (within N=256 binomial noise); bit-identical reward curves are out of scope because async RL and hardware vary.

## Known non-clean runs (still part of the record)

| Trial | Issue | How conclusions treat it |
|-------|-------|---------------------------|
| `p3-h4-tau005` | n=237 (resume/extra) | Recipe sweep evidence; not a clean 200-step arm |
| `p5-effort-tau2-24gpu` | n=147 | Stage A as logged; capability judged on Stage B boxed |
| `p6-*-v1/v2` | VOID | Failure-mode artifacts; required for C6 narrative |

## What “reproduced” means here

1. **C0–C7 pass** `verify_reported_results.py` on a clean checkout.
2. Optional Tier-2: retrain with listed configs/seeds; reeval with the fixed probe; same-batch gates still pass.
3. “Reproduced” here means this ablation protocol and artifact gates — not numeric parity with K3 paper tables.
