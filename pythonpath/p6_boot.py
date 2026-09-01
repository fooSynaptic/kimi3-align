"""P6 boot: apply approx MOPD when K3_MOPD_ENABLED=1.

Import from driver (e1_math_rl_train) and RPC scheduling_spec cmd.
Safe no-op when disabled. Does not import torch before CUDA_VISIBLE_DEVICES
except inside mopd_boot.apply() after enable check.
"""
from __future__ import annotations

import os


def apply() -> None:
    if os.environ.get("K3_MOPD_ENABLED", "0").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    import mopd_boot

    mopd_boot.apply()


apply()
