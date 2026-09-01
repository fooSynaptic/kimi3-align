# K3-align E1 — (Obsolete)

> **The claim in this document is invalid.** See [`EXPERIMENT_DESIGN.md`](EXPERIMENT_DESIGN.md) for the strictly aligned design.  
> The legacy configuration incorrectly labeled SAO `GRPO+DIS` + `max_head_offpolicyness=4` as K3 §4.1.2 partial rollout.

Historical record for the retired E1/partial-grpo line — the K3-aligned mainline is [`EXPERIMENT_DESIGN.md`](EXPERIMENT_DESIGN.md) P0→P6:

- Cold start: E0 r2 epoch2 (boxed `warn_drop`)
- Algorithm: GRPO+DIS G=8 mask `[0.3,5]` (SAO hard recipe)
- Trial: `partial-grpo-dis-e0warm`
- Smoke ~89 s/step; 200 steps were estimated at ≈ 5 h

Next mainline step: stop or relabel this trial → fix recipe R → run a single-factor **`H=0` sync vs `H=4` async-head`** comparison.
