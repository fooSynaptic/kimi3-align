"""K2.5 τ boot helper.

Prefer the in-repo AReaL ``functional.py`` ``k25_tau_patch`` (reads
``K25_TAU_LOG_RATIO_SQ`` at loss time). Do NOT import torch/areal from
sitecustomize — that initializes CUDA before Ray sets CUDA_VISIBLE_DEVICES
and causes NCCL ``Duplicate GPU``.
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
