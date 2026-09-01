#!/usr/bin/env python3
"""Extract per-step train metrics from AReaL main.log ascii tables.

Usage:
  python3 scripts/extract_train_curves.py \\
    --log /path/to/main.log --trial p1-k25-sync-h0-full --out docs/curves/

Writes:
  <out>/<trial>.csv
  <out>/<trial>.json  (summary + series)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


DEFAULT_KEYS = (
    "ppo_actor/task_reward/avg",
    "rollout/reward",
    "ppo_actor/seq_len/avg",
    "timeperf/train_step",
    "timeperf/update_weights",
    "timeperf/rollout",
    "correct_n_seqs",
)


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def parse_series(text: str, keys: tuple[str, ...]) -> dict[str, list[float]]:
    # Table cells: │ key │ value │
    pairs = re.findall(
        r"│\s*([a-zA-Z0-9_./+-]+)\s*│\s*([-+]?\d+\.\d+(?:e[+-]?\d+)?)\s*",
        text,
    )
    series: dict[str, list[float]] = {k: [] for k in keys}
    for k, v in pairs:
        if k in series:
            series[k].append(float(v))
    return series


def summarize(series: dict[str, list[float]]) -> dict:
    out = {}
    for k, xs in series.items():
        if not xs:
            out[k] = None
            continue
        n = len(xs)
        out[k] = {
            "n": n,
            "first": xs[0],
            "last": xs[-1],
            "mean_first20": sum(xs[:20]) / min(20, n),
            "mean_last20": sum(xs[-20:]) / min(20, n),
            "min": min(xs),
            "max": max(xs),
        }
    return out


def write_csv(path: Path, series: dict[str, list[float]]) -> int:
    keys = [k for k, xs in series.items() if xs]
    if not keys:
        path.write_text("step\n")
        return 0
    n = max(len(series[k]) for k in keys)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step"] + keys)
        for i in range(n):
            row = [i + 1]
            for k in keys:
                xs = series[k]
                row.append(xs[i] if i < len(xs) else "")
            w.writerow(row)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keys", nargs="*", default=list(DEFAULT_KEYS))
    args = ap.parse_args()

    text = strip_ansi(args.log.read_text(errors="ignore"))
    keys = tuple(args.keys)
    series = parse_series(text, keys)
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / f"{args.trial}.csv"
    json_path = args.out / f"{args.trial}.json"
    n = write_csv(csv_path, series)
    payload = {
        "trial": args.trial,
        "log": str(args.log),
        "n_steps_inferred": n,
        "summary": summarize(series),
        "series": series,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] {args.trial} steps={n} -> {csv_path}")


if __name__ == "__main__":
    main()
