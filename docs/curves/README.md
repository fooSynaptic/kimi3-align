# Train convergence curves

Extracted from AReaL `main.log` StatsLogger tables for each full trial.

## Contents

| Path | What |
|------|------|
| `*.csv` | Per-step metrics (`task_reward`, `rollout/reward`, `seq_len`, timeperf, …) |
| `*.json` | Same series + first20/last20 summary (paths sanitized) |
| `figures/*.svg` | Per-trial and overlay plots (stdlib SVG, no matplotlib) |
| `COMPARE.md` | Summary table + embedded figures |

## Rebuild

From repo root (after placing `main.log` files or pointing `--log`):

```bash
python3 scripts/extract_train_curves.py \
  --log path/to/<trial>/main.log --trial <trial> --out docs/curves/

python3 scripts/plot_train_curves.py --curves-dir docs/curves
```

## Canonical trials

| Label | Trial name |
|-------|------------|
| P1 | `p1-k25-sync-h0-full` |
| P2 | `p2-k25-async-h4` |
| P3 | `p3-h4-tau0`, `p3-h4-tau005`, `p3-h4-mask-tight` |
| P4 | `p4-h4-lambda05-16gpu` |
| P5 | `p5-effort-tau2-24gpu`, `p5-effort-tau1-24gpu` |
| P6 | `p6-mopd-p1teacher-24gpu` (+ `-v2`/`-v3`/`-v4`) |

P0 has no RL train curve (Instruct cold-start gate only).
