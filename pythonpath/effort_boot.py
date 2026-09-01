"""P5 boot: mark Effort env presence for driver/RPC workers.

Import from driver (e1_math_rl_train) and RPC scheduling_spec cmd.
Safe no-op when K3_EFFORT_ENABLED is off. Does not import torch.
"""
from __future__ import annotations

import os


def enabled() -> bool:
    return os.environ.get("K3_EFFORT_ENABLED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def apply() -> None:
    if not enabled():
        return
    # Ensure audit dir exists early (optional).
    path = os.environ.get("K3_EFFORT_AUDIT_PATH", "").strip()
    if path:
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)


# Apply on import so `python -c "import effort_boot; ..."` works for RPC workers.
apply()
