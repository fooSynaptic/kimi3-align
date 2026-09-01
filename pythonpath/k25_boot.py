"""K2.5 τ boot helper.

Uses in-repo AReaL ``functional.py`` ``k25_tau_patch`` (reads
``K25_TAU_LOG_RATIO_SQ`` at loss time). Importing torch/areal from
sitecustomize runs before Ray sets ``CUDA_VISIBLE_DEVICES``, which
triggers NCCL ``Duplicate GPU`` on multi-node launches — boot stays
import-light until workers start.
"""
from __future__ import annotations

import os

_TAU = float(os.environ.get("K25_TAU_LOG_RATIO_SQ", "0") or "0")


def apply() -> None:
    if _TAU <= 0:
        return
    print(
        f"[k25_boot] τ={_TAU} (loss-time patch in areal.utils.functional)",
        flush=True,
    )
