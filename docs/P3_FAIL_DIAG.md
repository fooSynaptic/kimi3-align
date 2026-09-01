# P3 tau005 Failure Diagnosis

## Symptoms
- Arm `p3-h4-tau005` fails after **training step 37**.
- Log: `Task was killed due to the node running low on memory` (`pause` / `destroy` on `rollout/2|3`).
- The queue stops on failure, so `mask-tight` does not start.

## Root Cause (Not Algorithmic)
- Orphaned `multiprocessing.spawn` processes with `ppid=1` exhaust **host RAM** (hundreds of GiB RSS per process).
- Ray kills the rollout worker at `memory_usage_threshold≈0.98`, causing training to fail.
- This is unrelated to τ=0.05 itself; `tau0` already completed all 200 steps.

## Fix / Resume
1. Send **TERM** to orphaned spawn processes. Broad `kill -9` on GPU-attached workers often orphans CUDA contexts and leaves dirty VRAM; TERM lets vLLM/Ray tear down more cleanly.
2. Confirm `MemAvailable` ≥ 200GiB and clean VRAM.
3. Skip completed `tau0` and rerun `tau005` + `mask-tight` (`SKIP_DONE=1`).
4. `saver.freq_steps=200` matches the full 200-step arm so resume does not accumulate extra intermediate checkpoints.
