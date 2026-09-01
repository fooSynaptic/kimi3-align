#!/usr/bin/env python3
"""K3-align MATH RLVR entry (AReaL PPOTrainer).

K2.5 τ(log-ratio)^2: set K25_TAU_LOG_RATIO_SQ and ensure k25_boot is on PYTHONPATH
(for driver here; RPC workers import k25_boot via scheduling_spec cmd).

P4 λ soft barrier: set K3_LAMBDA_ENABLED=1 (+ K3_LAMBDA_*); p4_boot patches
StalenessManager / BatchTaskDispatcher (driver). RPC may import p4_boot too.

P5 Effort: set K3_EFFORT_ENABLED=1 (+ K3_EFFORT_*); swaps reward_fn to
effort_reward.effort_gsm8k_reward_fn (hard -1 when T > τ·b0).

P6 approx MOPD: set K3_MOPD_ENABLED=1 (+ K3_MOPD_*); p6_boot patches
PPO advantages with Eq.15 clipped teacher/student log-ratio.
"""
from __future__ import annotations

import os
import sys


def main(args: list[str]) -> None:
    try:
        import k25_boot

        k25_boot.apply()
    except ImportError:
        pass
    try:
        import p4_boot

        p4_boot.apply()
    except ImportError:
        pass
    try:
        import effort_boot

        effort_boot.apply()
    except ImportError:
        pass
    try:
        import p6_boot

        p6_boot.apply()
    except ImportError:
        pass

    from areal import PPOTrainer
    from areal.api.cli_args import PPOConfig, load_expr_config
    from areal.dataset import get_custom_dataset
    from areal.utils.hf_utils import load_hf_tokenizer

    config, _ = load_expr_config(args, PPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    effort_on = os.environ.get("K3_EFFORT_ENABLED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    reward_fn = (
        "effort_reward.effort_gsm8k_reward_fn"
        if effort_on
        else "areal.reward.gsm8k.gsm8k_reward_fn"
    )
    workflow_kwargs = {
        "reward_fn": reward_fn,
        "gconfig": config.gconfig,
        "tokenizer": config.tokenizer_path,
        "enable_thinking": False,
        "reward_config": config.reward,
    }
    with PPOTrainer(config, train_dataset=train_dataset, valid_dataset=None) as trainer:
        trainer.train(
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
