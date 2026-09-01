"""P4 scheme-B λ soft barrier (approx K3 λ-partial).

Env (all optional except enable):
  K3_LAMBDA_ENABLED=1
  K3_LAMBDA=0.5
  K3_LAMBDA_N=160          # prompts per train batch (train_dataset.batch_size)
  K3_LAMBDA_K=8            # n_samples / group size
  K3_LAMBDA_AUDIT_PATH=... # JSONL audit file

Semantics:
  target_traj = ceil(λ * N * K)
  Each accepted GroupedRollout counts as K trajectories.
  When window_completed >= target → capacity/pending=0 (stop new submits).
  Wait loop exits when barrier active and running==0 and enough accepts,
  so prepare_batch does not hang (requires dynamic_bs or early-exit patch).
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any

_PATCHED = False
_AUDIT_LOCK = Lock()


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


def enabled() -> bool:
    return os.environ.get("K3_LAMBDA_ENABLED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _audit_path() -> Path | None:
    p = os.environ.get("K3_LAMBDA_AUDIT_PATH", "").strip()
    return Path(p) if p else None


def audit(event: str, **fields: Any) -> None:
    path = _audit_path()
    if path is None:
        return
    rec = {
        "ts": time.time(),
        "event": event,
        **fields,
    }
    line = json.dumps(rec, ensure_ascii=False)
    with _AUDIT_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def apply() -> None:
    """Monkeypatch AReaL StalenessManager + BatchTaskDispatcher wait loop."""
    global _PATCHED
    if _PATCHED or not enabled():
        return

    from areal.infra.staleness_manager import StalenessManager
    from areal.infra.workflow_executor import BatchTaskDispatcher

    lam = _env_float("K3_LAMBDA", 0.5)
    n_prompts = _env_int("K3_LAMBDA_N", 168)
    k_samples = _env_int("K3_LAMBDA_K", 8)
    dp_size = max(1, _env_int("K3_LAMBDA_DP", 12))
    raw_target_traj = max(1, math.ceil(lam * n_prompts * k_samples))
    # accepted tasks are GroupedRolloutWorkflow → one accept == K trajectories
    raw_target_groups = max(1, math.ceil(raw_target_traj / max(1, k_samples)))
    # FSDP DP requires #groups divisible by dp_size
    target_groups = max(dp_size, (raw_target_groups // dp_size) * dp_size)
    target_traj = target_groups * k_samples

    orig_init = StalenessManager.__init__
    orig_get_capacity = StalenessManager.get_capacity
    orig_get_pending = StalenessManager.get_pending_limit
    orig_accepted = StalenessManager.on_rollout_accepted
    orig_submit_wait = BatchTaskDispatcher.active_submit_and_wait

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self._k3_lambda = lam
        self._k3_n = n_prompts
        self._k3_k = k_samples
        self._k3_target_traj = target_traj
        self._k3_target_groups = target_groups
        self._k3_window_traj = 0
        self._k3_window_groups = 0
        self._k3_barrier = False
        self._k3_last_version = None
        self._k3_window_t0 = time.time()
        audit(
            "init",
            lambda_=lam,
            n=n_prompts,
            k=k_samples,
            target_nk=target_traj,
            target_groups=target_groups,
        )

    def _maybe_reset_on_version(self) -> None:
        try:
            ver = int(self.version_provider.get_version())
        except Exception:
            return
        if self._k3_last_version is None:
            self._k3_last_version = ver
            return
        if ver != self._k3_last_version:
            wall_ms = (time.time() - self._k3_window_t0) * 1000.0
            audit(
                "reset",
                step=ver,
                version=ver,
                prev_version=self._k3_last_version,
                lambda_=self._k3_lambda,
                target_nk=self._k3_target_traj,
                completed=self._k3_window_traj,
                groups=self._k3_window_groups,
                inflight=self.rollout_stat.running,
                capacity=0,
                wall_ms=round(wall_ms, 2),
            )
            self._k3_last_version = ver
            self._k3_window_traj = 0
            self._k3_window_groups = 0
            self._k3_barrier = False
            self._k3_window_t0 = time.time()
            audit(
                "resume",
                step=ver,
                version=ver,
                lambda_=self._k3_lambda,
                target_nk=self._k3_target_traj,
                completed=0,
                inflight=self.rollout_stat.running,
                capacity=-1,
            )

    def patched_get_capacity(self) -> int:
        with self.lock:
            _maybe_reset_on_version(self)
            if self._k3_barrier:
                return 0
        return orig_get_capacity(self)

    def patched_get_pending(self) -> int:
        with self.lock:
            _maybe_reset_on_version(self)
            if self._k3_barrier:
                return 0
        return orig_get_pending(self)

    def patched_accepted(self) -> None:
        orig_accepted(self)
        with self.lock:
            _maybe_reset_on_version(self)
            self._k3_window_groups += 1
            self._k3_window_traj += self._k3_k
            if (not self._k3_barrier) and (
                self._k3_window_traj >= self._k3_target_traj
                or self._k3_window_groups >= self._k3_target_groups
            ):
                self._k3_barrier = True
                wall_ms = (time.time() - self._k3_window_t0) * 1000.0
                ver = self._k3_last_version
                audit(
                    "pause",
                    step=ver,
                    version=ver,
                    lambda_=self._k3_lambda,
                    target_nk=self._k3_target_traj,
                    completed=self._k3_window_traj,
                    groups=self._k3_window_groups,
                    inflight=self.rollout_stat.running,
                    capacity=0,
                    wall_ms=round(wall_ms, 2),
                )

    def patched_submit_and_wait(
        self,
        input_generator,
        batch_size: int,
        dynamic_bs: bool = False,
    ):
        """Same as upstream, plus λ early-exit when barrier reached."""
        accepted_cnt = 0
        total_attempts = 0
        results = []
        min_groups = target_groups
        idle_iters = 0

        while True:
            with self._input_cv:
                pending_inputs = len(self._pending_inputs)
            cap_staleness = self.staleness_manager.get_pending_limit() - pending_inputs
            if self.runner.max_queue_size < batch_size:
                raise ValueError(
                    f"Inference engine config's queue size is too small: "
                    f"{self.runner.max_queue_size} < batch size {batch_size}."
                )
            cap_queue = self.runner.max_queue_size - (
                self.runner.get_input_queue_size() + batch_size
            )
            capacity = min(cap_staleness, cap_queue)
            if capacity > 0:
                for _ in range(min(batch_size, capacity)):
                    try:
                        self.submit_task_input(next(input_generator))
                    except StopIteration as e:
                        raise RuntimeError(
                            "Input generator exhausted before batch completion. "
                            "Use cycle_dataloader() or provide an infinite generator."
                        ) from e
            # IMPORTANT: wait for 1 at a time. Upstream waits for
            # (batch_size - accepted) which deadlocks under λ barrier when
            # fewer than a full batch will ever arrive (partial results sit
            # in _pending_results forever while each wait times out).
            try:
                arrived = self.wait_results(
                    count=1, timeout=1.0, raise_timeout=False
                )
            except TimeoutError:
                arrived = []
            if not arrived:
                arrived = []

            if arrived:
                idle_iters = 0
            else:
                idle_iters += 1

            for res in arrived:
                is_accepted = res is not None
                if not is_accepted:
                    if dynamic_bs:
                        total_attempts += 1
                        if total_attempts >= batch_size:
                            break
                    continue
                accepted_cnt += 1
                total_attempts += 1
                results.append(res)
                if dynamic_bs:
                    if total_attempts >= batch_size:
                        break
                elif accepted_cnt >= batch_size:
                    break
            else:
                sm = self.staleness_manager
                barrier = bool(getattr(sm, "_k3_barrier", False))
                running = 0
                try:
                    running = int(sm.get_stats().running)
                except Exception:
                    running = int(getattr(sm.rollout_stat, "running", 0))
                # Exit once λ threshold met. Prefer drained; if idle too long with
                # enough accepts, force train to avoid deadlock on stuck counters.
                enough = accepted_cnt >= min_groups and accepted_cnt > 0
                drained = running <= 0
                stalled = barrier and enough and idle_iters >= 15
                if barrier and enough and (drained or stalled):
                    # Truncate to DP-aligned group count for FSDP dispatch
                    n_keep = (accepted_cnt // dp_size) * dp_size
                    if n_keep < dp_size:
                        # not enough yet — keep waiting
                        continue
                    if n_keep < accepted_cnt:
                        results = results[:n_keep]
                        accepted_cnt = n_keep
                    audit(
                        "train",
                        lambda_=lam,
                        target_nk=target_traj,
                        completed=accepted_cnt * k_samples,
                        groups=accepted_cnt,
                        inflight=running,
                        capacity=0,
                        batch_size=batch_size,
                        dp_size=dp_size,
                        stalled=bool(stalled),
                    )
                    break
                continue
            break

        # Final safety: never return a non-DP-aligned batch
        if results:
            n_keep = (len(results) // dp_size) * dp_size
            if n_keep == 0:
                raise RuntimeError(
                    f"λ-barrier collected {len(results)} groups but need ≥{dp_size} "
                    f"for FSDP dp={dp_size}"
                )
            results = results[:n_keep]
        return results

    StalenessManager.__init__ = patched_init  # type: ignore[method-assign]
    StalenessManager.get_capacity = patched_get_capacity  # type: ignore[method-assign]
    StalenessManager.get_pending_limit = patched_get_pending  # type: ignore[method-assign]
    StalenessManager.on_rollout_accepted = patched_accepted  # type: ignore[method-assign]
    BatchTaskDispatcher.active_submit_and_wait = patched_submit_and_wait  # type: ignore[method-assign]

    _PATCHED = True
    print(
        f"[lambda_barrier] enabled λ={lam} N={n_prompts} K={k_samples} "
        f"target_traj={target_traj} target_groups={target_groups} "
        f"audit={_audit_path()}",
        flush=True,
    )
