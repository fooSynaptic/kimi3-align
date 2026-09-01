"""P6 approx MOPD: Eq.15 clipped log-ratio as dense token reward.

K3 §4.1.3:
  r_opd = clip(sg(log π_teacher − log π_student), −R_max, R_max)

This patch adds a suffix-sum of α·r_opd into PPO advantages
(values=0, γ=λ_gae=1 ⇒ GAE of per-token OPD). Outcome gsm8k reward is
unchanged. Distill KL loss stays 0 because Eq.15 already injects dense OPD
into advantage (enabling RKL would double-count the same signal).

v1 (α=0.05, no length-norm) let suffix-sum scale with T≈500 → |A_opd|~30
vs outcome 0/1 and collapsed train reward to 0 after ~100 steps. v2 divides
per-sequence valid length so |A_opd| is O(α·Rmax), not O(α·Rmax·T).

Env:
  K3_MOPD_ENABLED=1
  K3_MOPD_RMAX=2.0
  K3_MOPD_ALPHA=0.5
  K3_MOPD_LEN_NORM=1
  K3_MOPD_ADV_CLIP=2.0
  K3_MOPD_POS_ONLY=0   # v4+: keep only r_opd>0 (drop negative teacher−student)
  K3_MOPD_AUDIT_PATH=...
  K3_MOPD_AUDIT_EVERY=1
"""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from threading import Lock
from typing import Any

_PATCHED = False
_AUDIT_LOCK = Lock()
_STEP_N = 0


def enabled() -> bool:
    return os.environ.get("K3_MOPD_ENABLED", "0").strip().lower() in (
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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "")
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _audit(event: str, **fields: Any) -> None:
    path = os.environ.get("K3_MOPD_AUDIT_PATH", "").strip()
    if not path:
        return
    rec = {"ts": time.time(), "event": event, **fields}
    try:
        line = json.dumps(rec, ensure_ascii=False, default=str)
        p = Path(path)
        with _AUDIT_LOCK:
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as exc:  # noqa: BLE001 — audit failures are logged; training continues
        print(f"[mopd_boot] audit write failed: {exc}", flush=True)


def _align_last_dim(t, s, mask, adv):
    import torch

    if t.shape == s.shape:
        return t, s, mask, adv
    if t.dim() != s.dim():
        return None
    if t.shape[:-1] != s.shape[:-1]:
        return None
    L = min(int(t.shape[-1]), int(s.shape[-1]), int(mask.shape[-1]), int(adv.shape[-1]))
    return t[..., :L], s[..., :L], mask[..., :L], adv[..., :L]


def inject_opd(data: dict[str, Any]) -> dict[str, Any]:
    """Add Eq.15 dense OPD into `data['advantages']`. Used from AReaL PPOActor.

    Ray workers ignore scheduling_spec.cmd, so the in-tree actor.py hook calls
    this after GAE. No-op when K3_MOPD_ENABLED is off.
    """
    global _STEP_N
    if not enabled() or not isinstance(data, dict):
        return data
    import torch

    rmax = _env_float("K3_MOPD_RMAX", 2.0)
    alpha = _env_float("K3_MOPD_ALPHA", 0.5)
    len_norm = _env_bool("K3_MOPD_LEN_NORM", True)
    adv_clip = _env_float("K3_MOPD_ADV_CLIP", 2.0)
    pos_only = _env_bool("K3_MOPD_POS_ONLY", False)
    every = max(1, _env_int("K3_MOPD_AUDIT_EVERY", 8))
    tlogp = data.get("teacher_logp")
    slogp = data.get("logprobs")
    loss_mask = data.get("loss_mask")
    adv = data.get("advantages")
    _STEP_N += 1
    do_audit = ((_STEP_N - 1) % every) == 0
    if tlogp is None or slogp is None or loss_mask is None or adv is None:
        if do_audit:
            _audit(
                "skip",
                n=_STEP_N,
                reason="missing_tensor",
                has_teacher=tlogp is not None,
                has_logprobs=slogp is not None,
                has_mask=loss_mask is not None,
                has_adv=adv is not None,
                pid=os.getpid(),
            )
        return data
    try:
        t = tlogp.detach()
        s = slogp.detach()
        aligned = _align_last_dim(t, s, loss_mask.float(), adv)
        if aligned is None:
            if do_audit:
                _audit(
                    "skip",
                    n=_STEP_N,
                    reason="shape",
                    t_shape=list(t.shape),
                    s_shape=list(s.shape),
                    m_shape=list(loss_mask.shape),
                    a_shape=list(adv.shape),
                    pid=os.getpid(),
                )
            return data
        t, s, mask, adv_a = aligned
        t = torch.roll(t, shifts=-1, dims=-1)
        raw = t - s
        tok = raw.clamp(-rmax, rmax) * mask
        if pos_only:
            tok = tok.clamp(min=0.0)
        if len_norm:
            seq_len = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            opd = alpha * tok / seq_len
        else:
            opd = alpha * tok
        suffix = torch.cumsum(opd.flip(dims=[-1]), dim=-1).flip(dims=[-1])
        if adv_clip > 0:
            suffix = suffix.clamp(-adv_clip, adv_clip)
        if suffix.shape == adv.shape:
            data["advantages"] = adv + suffix
        else:
            patched = adv.clone()
            patched[..., : suffix.shape[-1]] = adv[..., : suffix.shape[-1]] + suffix
            data["advantages"] = patched
        data["mopd_opd"] = opd
        denom = mask.sum().clamp(min=1.0)
        if do_audit:
            pos_frac = float(((raw > 0).float() * mask).sum().detach().cpu() / float(denom))
            _audit(
                "opd",
                n=_STEP_N,
                mean_opd=float((opd.sum() / denom).detach().cpu()),
                mean_suffix=float((suffix * mask).sum().detach().cpu() / float(denom)),
                max_abs_suffix=float(suffix.abs().max().detach().cpu()),
                mean_raw_log_ratio=float(((raw * mask).sum() / denom).detach().cpu()),
                clip_frac=float(
                    ((raw.abs() >= rmax).float() * mask).sum().detach().cpu()
                    / float(denom)
                ),
                pos_frac=pos_frac,
                pos_only=pos_only,
                rmax=rmax,
                alpha=alpha,
                len_norm=len_norm,
                adv_clip=adv_clip,
                t_shape=list(t.shape),
                s_shape=list(s.shape),
                pid=os.getpid(),
            )
    except Exception as exc:  # noqa: BLE001
        if do_audit:
            _audit("skip", n=_STEP_N, reason="exception", err=str(exc), pid=os.getpid())
        print(f"[mopd_boot] OPD failed: {exc}\n{traceback.format_exc()}", flush=True)
    return data


def apply() -> None:
    global _PATCHED
    if _PATCHED or not enabled():
        return

    from areal.trainer.ppo.actor import PPOActor

    orig = PPOActor._compute_advantages

    def _compute_advantages(self, data, meta=None):
        t_in = data.get("teacher_logp") if isinstance(data, dict) else None
        out = orig(self, data, meta)
        if isinstance(out, dict) and t_in is not None and out.get("teacher_logp") is None:
            out["teacher_logp"] = t_in
        return inject_opd(out)

    PPOActor._compute_advantages = _compute_advantages
    _PATCHED = True
    path = os.environ.get("K3_MOPD_AUDIT_PATH", "").strip()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    _audit(
        "boot",
        rmax=_env_float("K3_MOPD_RMAX", 2.0),
        alpha=_env_float("K3_MOPD_ALPHA", 0.5),
        len_norm=_env_bool("K3_MOPD_LEN_NORM", True),
        adv_clip=_env_float("K3_MOPD_ADV_CLIP", 2.0),
        pos_only=_env_bool("K3_MOPD_POS_ONLY", False),
        every=max(1, _env_int("K3_MOPD_AUDIT_EVERY", 8)),
        pid=os.getpid(),
    )
    print(
        f"[mopd_boot] Eq.15 dense OPD α={_env_float('K3_MOPD_ALPHA', 0.5)} "
        f"Rmax={_env_float('K3_MOPD_RMAX', 2.0)} "
        f"len_norm={_env_bool('K3_MOPD_LEN_NORM', True)} "
        f"adv_clip={_env_float('K3_MOPD_ADV_CLIP', 2.0)} "
        f"pos_only={_env_bool('K3_MOPD_POS_ONLY', False)} "
        f"audit={path or 'off'}",
        flush=True,
    )


apply()
