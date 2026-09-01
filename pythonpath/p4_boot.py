"""P4 boot: apply λ soft barrier when K3_LAMBDA_ENABLED=1.

Import from driver (e1_math_rl_train) and RPC scheduling_spec cmd.
Safe no-op when disabled. Does not import torch before CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import os


def apply() -> None:
    if os.environ.get("K3_LAMBDA_ENABLED", "0").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    import lambda_barrier

    lambda_barrier.apply()


# Apply on import so `python -c "import p4_boot; ..."` works for RPC workers.
apply()
