#!/usr/bin/env python3
"""Render train-curve SVGs + COMPARE.md from docs/curves/*.json (stdlib only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


COLORS = [
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#be185d",
    "#4b5563",
    "#65a30d",
]


def svg_polyline(
    xs: list[float],
    ys: list[float],
    *,
    width: int = 720,
    height: int = 280,
    pad: int = 44,
    stroke: str = "#2563eb",
    title: str = "",
    ylabel: str = "",
) -> str:
    if not xs or not ys or len(xs) != len(ys):
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<text x="20" y="40">no data</text></svg>'
        )
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if abs(ymax - ymin) < 1e-12:
        ymax = ymin + 1.0
    if abs(xmax - xmin) < 1e-12:
        xmax = xmin + 1.0

    def px(x: float) -> float:
        return pad + (x - xmin) / (xmax - xmin) * (width - 2 * pad)

    def py(y: float) -> float:
        return height - pad - (y - ymin) / (ymax - ymin) * (height - 2 * pad)

    pts = " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in zip(xs, ys))
    yticks = [ymin + i * (ymax - ymin) / 4 for i in range(5)]
    xticks = [xmin + i * (xmax - xmin) / 4 for i in range(5)]
    grid: list[str] = []
    for y in yticks:
        yy = py(y)
        grid.append(
            f'<line x1="{pad}" y1="{yy:.2f}" x2="{width - pad}" y2="{yy:.2f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{pad - 8}" y="{yy + 4:.2f}" text-anchor="end" '
            f'font-size="11" fill="#6b7280">{y:.3g}</text>'
        )
    for x in xticks:
        xx = px(x)
        grid.append(
            f'<text x="{xx:.2f}" y="{height - pad + 16}" text-anchor="middle" '
            f'font-size="11" fill="#6b7280">{int(x)}</text>'
        )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{width / 2}" y="22" text-anchor="middle" font-size="14" font-family="ui-sans-serif,system-ui" fill="#111827">{title}</text>
  <text x="14" y="{height / 2}" text-anchor="middle" font-size="11" fill="#6b7280" transform="rotate(-90 14 {height / 2})">{ylabel}</text>
  {"".join(grid)}
  <polyline fill="none" stroke="{stroke}" stroke-width="2" points="{pts}"/>
  <text x="{width / 2}" y="{height - 8}" text-anchor="middle" font-size="11" fill="#6b7280">step</text>
</svg>
'''


def multi_svg(
    curves: list[tuple[str, list[float], list[float], str]],
    *,
    title: str,
    ylabel: str,
    width: int = 800,
    height: int = 320,
    pad: int = 48,
) -> str:
    if not curves:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="120">'
            '<text x="20" y="40">no data</text></svg>'
        )
    all_x = [x for _, xs, _, _ in curves for x in xs]
    all_y = [y for _, _, ys, _ in curves for y in ys]
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    if abs(ymax - ymin) < 1e-12:
        ymax = ymin + 1.0
    if abs(xmax - xmin) < 1e-12:
        xmax = xmin + 1.0

    def px(x: float) -> float:
        return pad + (x - xmin) / (xmax - xmin) * (width - 2 * pad)

    def py(y: float) -> float:
        return height - pad - (y - ymin) / (ymax - ymin) * (height - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="22" text-anchor="middle" font-size="14" '
        f'font-family="ui-sans-serif,system-ui" fill="#111827">{title}</text>',
        f'<text x="14" y="{height / 2}" text-anchor="middle" font-size="11" fill="#6b7280" '
        f'transform="rotate(-90 14 {height / 2})">{ylabel}</text>',
    ]
    for y in [ymin + i * (ymax - ymin) / 4 for i in range(5)]:
        yy = py(y)
        parts.append(
            f'<line x1="{pad}" y1="{yy:.2f}" x2="{width - pad}" y2="{yy:.2f}" stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{pad - 8}" y="{yy + 4:.2f}" text-anchor="end" '
            f'font-size="11" fill="#6b7280">{y:.3g}</text>'
        )
    for x in [xmin + i * (xmax - xmin) / 4 for i in range(5)]:
        xx = px(x)
        parts.append(
            f'<text x="{xx:.2f}" y="{height - pad + 16}" text-anchor="middle" '
            f'font-size="11" fill="#6b7280">{int(x)}</text>'
        )
    legend_y = 40
    for name, xs, ys, color in curves:
        pts = " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in zip(xs, ys))
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>'
        )
        parts.append(
            f'<rect x="{width - pad - 150}" y="{legend_y - 10}" width="12" height="3" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{width - pad - 132}" y="{legend_y}" font-size="11" fill="#374151">{name}</text>'
        )
        legend_y += 16
    parts.append(
        f'<text x="{width / 2}" y="{height - 8}" text-anchor="middle" '
        f'font-size="11" fill="#6b7280">step</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


NOTES = {
    "P0": "No RL train curve (cold-start gate only); see boxed Instruct.",
    "P1": "Sync ceiling: reward climbs ~0.38→0.47 (first20→last20).",
    "P2": "H=4: similar early rise; slightly lower late reward than P1.",
    "P3-tau0": "τ_R=0 ≈ P2 shape (recipe τ not the P2 drop cause).",
    "P3-tau005": "τ_R=0.05 sweep arm (longer log; compare to P2).",
    "P3-mask": "Tighter ratio mask sweep; same H=4 base.",
    "P4": "λ=0.5 pause/train: reward plateaus with more step noise.",
    "P5-A": "τ_E=2.0: mild length pressure; reward still rising at stop (147 steps).",
    "P5-B": "τ_E=1.0: task_reward deeply negative (over_budget≈97%); boxed still held.",
    "P6-v1": "VOID: reward collapses to ~0 (scale / disk path).",
    "P6-v2": "VOID: aborted early (~19 steps, NFS disk timeout).",
    "P6-v3": "FAIL: trains to 200 but late reward below P1; negative OPD bias.",
    "P6-v4": "PASS: P1 warm-start + pos-only; starts high (~0.49) and stays.",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves-dir", type=Path, default=Path("docs/curves"))
    ap.add_argument("--reward-key", default="ppo_actor/task_reward/avg")
    ap.add_argument("--seq-key", default="ppo_actor/seq_len/avg")
    args = ap.parse_args()
    root = args.curves_dir
    fig = root / "figures"
    fig.mkdir(parents=True, exist_ok=True)

    catalog = [
        ("P1", "p1-k25-sync-h0-full"),
        ("P2", "p2-k25-async-h4"),
        ("P3-tau0", "p3-h4-tau0"),
        ("P3-tau005", "p3-h4-tau005"),
        ("P3-mask", "p3-h4-mask-tight"),
        ("P4", "p4-h4-lambda05-16gpu"),
        ("P5-A", "p5-effort-tau2-24gpu"),
        ("P5-B", "p5-effort-tau1-24gpu"),
        ("P6-v1", "p6-mopd-p1teacher-24gpu"),
        ("P6-v2", "p6-mopd-p1teacher-24gpu-v2"),
        ("P6-v3", "p6-mopd-p1teacher-24gpu-v3"),
        ("P6-v4", "p6-mopd-p1teacher-24gpu-v4"),
    ]

    rows = []
    reward_curves = []
    seq_curves = []
    for i, (label, trial) in enumerate(catalog):
        jp = root / f"{trial}.json"
        if not jp.exists():
            rows.append((label, trial, None))
            continue
        data = json.loads(jp.read_text(encoding="utf-8"))
        # sanitize absolute paths if present
        if "log" in data and ("/" in str(data.get("log", "")) and not str(data.get("log", "")).startswith("experiments/")):
            data["log"] = f"experiments/logs/.../{trial}/main.log"
            jp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        ys = data.get("series", {}).get(args.reward_key, [])
        seq = data.get("series", {}).get(args.seq_key, [])
        color = COLORS[i % len(COLORS)]
        if ys:
            xs = list(range(1, len(ys) + 1))
            (fig / f"{trial}_task_reward.svg").write_text(
                svg_polyline(
                    xs,
                    ys,
                    stroke=color,
                    title=f"{label} · {args.reward_key}",
                    ylabel="task_reward",
                ),
                encoding="utf-8",
            )
            reward_curves.append((label, xs, ys, color))
        if seq:
            xs = list(range(1, len(seq) + 1))
            (fig / f"{trial}_seq_len.svg").write_text(
                svg_polyline(
                    xs,
                    seq,
                    stroke=color,
                    title=f"{label} · {args.seq_key}",
                    ylabel="seq_len",
                ),
                encoding="utf-8",
            )
            if label in ("P1", "P2", "P4", "P5-A", "P5-B", "P6-v4"):
                seq_curves.append((label, xs, seq, color))
        rows.append((label, trial, (data.get("summary") or {}).get(args.reward_key)))

    mainline = [
        c
        for c in reward_curves
        if c[0] in ("P1", "P2", "P3-tau0", "P4", "P5-A", "P6-v4")
    ]
    p6 = [c for c in reward_curves if c[0].startswith("P6")]
    if mainline:
        (fig / "overlay_task_reward_mainline.svg").write_text(
            multi_svg(
                mainline,
                title="task_reward/avg · mainline arms",
                ylabel="task_reward",
            ),
            encoding="utf-8",
        )
    if p6:
        (fig / "overlay_task_reward_p6.svg").write_text(
            multi_svg(p6, title="task_reward/avg · P6 versions", ylabel="task_reward"),
            encoding="utf-8",
        )
    if seq_curves:
        (fig / "overlay_seq_len_mainline.svg").write_text(
            multi_svg(
                seq_curves,
                title="seq_len/avg · selected arms",
                ylabel="seq_len",
            ),
            encoding="utf-8",
        )

    lines = [
        "# Train convergence curves",
        "",
        "Source: AReaL `main.log` StatsLogger ascii tables.",
        f"Primary metric: `{args.reward_key}`.",
        f"Secondary: `{args.seq_key}` (Effort / length story).",
        "",
        "P0 has **no** RL train curve (Instruct cold-start gate only).",
        "",
        "## How to read",
        "",
        "- **Mainline overlay**: P1 / P2 / P3-τ0 / P4 / P5-A / P6-v4 on one canvas.",
        "- **P6 overlay**: v1 collapse, v2 abort, v3 underperform, v4 warm high plateau.",
        "- **P5-B**: negative `task_reward` is expected under hard Effort gate; judge by boxed, not train reward.",
        "",
        "## Overlay figures",
        "",
        "### Mainline `task_reward`",
        "",
        "![mainline reward](figures/overlay_task_reward_mainline.svg)",
        "",
        "### P6 versions `task_reward`",
        "",
        "![p6 reward](figures/overlay_task_reward_p6.svg)",
        "",
        "### Selected `seq_len`",
        "",
        "![seq len](figures/overlay_seq_len_mainline.svg)",
        "",
        "## Summary table",
        "",
        "| Label | Trial | n | first20 | last20 | last | Note |",
        "|-------|-------|---|---------|--------|------|------|",
    ]
    for label, trial, s in rows:
        note = NOTES.get(label, "")
        if not s:
            lines.append(f"| {label} | `{trial}` | — | — | — | — | {note} |")
            continue
        lines.append(
            "| {} | `{}` | {} | {:.4f} | {:.4f} | {:.4f} | {} |".format(
                label,
                trial,
                s.get("n"),
                s.get("mean_first20", float("nan")),
                s.get("mean_last20", float("nan")),
                s.get("last", float("nan")),
                note,
            )
        )

    lines += ["", "## Per-trial plots", ""]
    for label, trial, s in rows:
        reward_svg = fig / f"{trial}_task_reward.svg"
        if reward_svg.exists():
            lines.append(f"### {label} (`{trial}`)")
            lines.append("")
            lines.append(NOTES.get(label, ""))
            lines.append("")
            lines.append(f"![reward](figures/{trial}_task_reward.svg)")
            if (fig / f"{trial}_seq_len.svg").exists():
                lines.append("")
                lines.append(f"![seq_len](figures/{trial}_seq_len.svg)")
            lines.append("")

    (root / "COMPARE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote figures + {root / 'COMPARE.md'}")


if __name__ == "__main__":
    main()
