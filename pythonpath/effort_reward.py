"""P5 approx Effort: hard -1 when generation length exceeds τ·b0.

Env:
  K3_EFFORT_ENABLED=1
  K3_EFFORT_B0=256
  K3_EFFORT_TAU=2.0
  K3_EFFORT_AUDIT_PATH=...   # optional JSONL
  K3_EFFORT_AUDIT_EVERY=1    # audit every N calls (1=all)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any

_AUDIT_LOCK = Lock()
_CALL_N = 0
_CALL_LOCK = Lock()


def _enabled() -> bool:
    return os.environ.get("K3_EFFORT_ENABLED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if raw is None or str(raw).strip() == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def _gen_len(completion_ids: Any) -> int:
    if completion_ids is None:
        return 0
    try:
        return int(len(completion_ids))
    except TypeError:
        return 0


def _audit(event: str, **fields: Any) -> None:
    path = os.environ.get("K3_EFFORT_AUDIT_PATH", "").strip()
    if not path:
        return
    every = max(1, _env_int("K3_EFFORT_AUDIT_EVERY", 1))
    global _CALL_N
    with _CALL_LOCK:
        _CALL_N += 1
        n = _CALL_N
    if event != "over_budget" and (n % every) != 0:
        return
    rec = {"ts": time.time(), "event": event, "n": n, **fields}
    line = json.dumps(rec, ensure_ascii=False)
    p = Path(path)
    with _AUDIT_LOCK:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def effort_gsm8k_reward_fn(
    prompt,
    completions,
    prompt_ids,
    completion_ids,
    answer,
    **kwargs,
) -> float:
    from areal.reward.gsm8k import gsm8k_reward_fn

    base = float(
        gsm8k_reward_fn(
            prompt,
            completions,
            prompt_ids,
            completion_ids,
            answer,
            **kwargs,
        )
    )
    if not _enabled():
        return base

    b0 = _env_float("K3_EFFORT_B0", 256.0)
    tau = _env_float("K3_EFFORT_TAU", 2.0)
    budget = tau * b0
    t = _gen_len(completion_ids)
    if t > budget:
        _audit(
            "over_budget",
            t=t,
            budget=budget,
            tau=tau,
            b0=b0,
            base_reward=base,
            reward=-1.0,
        )
        return -1.0
    _audit(
        "ok",
        t=t,
        budget=budget,
        tau=tau,
        b0=b0,
        base_reward=base,
        reward=base,
    )
    return base
