#!/usr/bin/env python3
"""Verify checked-in artifacts still match docs/manifests/results_manifest.json.

Tier-1 reproducibility (no GPU): proves reported conclusions match local JSON/CSV.

Usage (repo root):
  python3 scripts/verify_reported_results.py
  python3 scripts/verify_reported_results.py --update-hashes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MANIFEST = DOCS / "manifests" / "results_manifest.json"


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def approx(a: float, b: float, atol: float) -> bool:
    return abs(float(a) - float(b)) <= atol


def load_json(rel: str) -> dict:
    return json.loads((DOCS / rel).read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-hashes", action="store_true")
    args = ap.parse_args()

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures: list[str] = []
    atol = float(man.get("curve_summary_expect", {}).get("atol", 5e-4))

    # --- hashes ---
    hashes = dict(man.get("artifact_sha256_16") or {})
    if args.update_hashes:
        for rel in list(hashes):
            p = DOCS / rel
            if p.exists():
                hashes[rel] = sha16(p)
        # include boxed_p4 compare if present
        p4 = "boxed_p4/compare.json"
        if (DOCS / p4).exists():
            hashes[p4] = sha16(DOCS / p4)
        man["artifact_sha256_16"] = hashes
        MANIFEST.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
        print(f"[ok] updated hashes in {MANIFEST.relative_to(ROOT)}")

    for rel, expect in hashes.items():
        p = DOCS / rel
        if not p.exists():
            failures.append(f"missing artifact: {rel}")
            continue
        got = sha16(p)
        if got != expect:
            failures.append(f"hash mismatch {rel}: got {got} expect {expect}")

    # --- C0 ---
    gate = load_json("p0_gate/gate.json")
    if not approx(gate["acc"], 0.65625, 1e-9):
        failures.append(f"C0 gate acc {gate['acc']} != 0.65625")

    # --- C1/C2/C3 batch_p1_p2_p3 ---
    c123 = load_json("boxed_p1_p2_p3/compare.json")
    by_tag = {r["tag"]: r for r in c123["rows"]}
    if not approx(by_tag["base_instruct"]["acc"], 0.65625, 1e-9):
        failures.append("C0/C1 base_instruct mismatch")
    if not approx(by_tag["p1_h0_final"]["acc"], 0.78125, 1e-9):
        failures.append("C1 p1 acc mismatch")
    if not approx(by_tag["p2_h4_final"]["acc"], 0.75, 1e-9):
        failures.append("C2 p2 acc mismatch")
    if not approx(by_tag["p3_tau0_final"]["acc"], 0.7578125, 1e-9):
        failures.append("C3 p3 acc mismatch")
    if not approx(c123["delta_p2_vs_p1"] * 100, -3.125, 1e-6):
        failures.append("C2 delta_pp mismatch")

    # --- C4 ---
    c4 = load_json("boxed_p4/compare.json")
    if not c4.get("gate_pass"):
        failures.append("C4 gate_pass false")
    if not approx(c4["p4"]["acc"], 0.7421875, 1e-9):
        failures.append("C4 p4 acc mismatch")
    if not approx(c4["delta_pp"], -1.953125, 1e-6):
        failures.append("C4 delta_pp mismatch")

    # --- C5 ---
    c5 = load_json("boxed_p5/compare.json")
    if not c5.get("gate_pass"):
        failures.append("C5 gate_pass false")
    if not approx(c5["p5"]["acc"], 0.76953125, 1e-9):
        failures.append("C5 p5 acc mismatch")
    if not approx(c5["delta_pp"], 0.78125, 1e-6):
        failures.append("C5 delta_pp mismatch")
    p5b = load_json("curves/p5-effort-tau1-24gpu.json")
    last20 = p5b["summary"]["ppo_actor/task_reward/avg"]["mean_last20"]
    if last20 >= 0:
        failures.append(f"C5 expected negative P5-B task_reward last20, got {last20}")

    # --- C6 ---
    c6 = load_json("boxed_p6/compare_v4.json")
    if not c6.get("gate_pass"):
        failures.append("C6 gate_pass false")
    if not approx(c6["p6"]["acc"], 0.79296875, 1e-9):
        failures.append("C6 v4 acc mismatch")
    if not approx(c6["p1"]["acc"], 0.78515625, 1e-9):
        failures.append("C6 same-batch p1 mismatch")
    delta_pp = (c6["p6"]["acc"] - c6["p1"]["acc"]) * 100
    if not approx(delta_pp, 0.78125, 1e-6):
        failures.append(f"C6 delta_pp {delta_pp} != 0.78125")
    if not approx(c6["p6_v3_ref"]["acc"], 0.72265625, 1e-9):
        failures.append("C6 v3 ref acc mismatch")
    v1 = load_json("curves/p6-mopd-p1teacher-24gpu.json")
    if not approx(v1["summary"]["ppo_actor/task_reward/avg"]["last"], 0.0, 1e-9):
        failures.append("C6 v1 last reward not 0")

    # --- C7 curves ---
    expect_rows = man["curve_summary_expect"]["rows"]
    metric = man["curve_summary_expect"]["metric"]
    for trial, exp in expect_rows.items():
        data = load_json(f"curves/{trial}.json")
        s = data["summary"][metric]
        for k in ("n", "mean_first20", "mean_last20", "last"):
            if k == "n":
                if int(s["n"]) != int(exp["n"]):
                    failures.append(f"C7 {trial} n {s['n']} != {exp['n']}")
            elif not approx(s[k], exp[k], atol):
                failures.append(f"C7 {trial} {k} {s[k]} != {exp[k]} (±{atol})")

    if failures:
        print("[FAIL] verify_reported_results")
        for f in failures:
            print(" -", f)
        return 1
    print("[ok] all claims C0–C7 match checked-in artifacts")
    print("     (Tier-1: artifact consistency. Tier-2: retrain+reeval needs cluster + ckpts.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
