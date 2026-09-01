#!/usr/bin/env python3
"""Prepare MATH-lighteval as chat-templated SFT (input_ids + loss_mask)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

DATASET_ID = "DigitalLearningGmbH/MATH-lighteval"
DEFAULT_MODEL = __import__("os").environ.get("MODEL_PATH", "Qwen/Qwen3-4B-Instruct-2507")


def convert_example(example: dict, tokenizer, max_length: int) -> dict | None:
    user = (
        example["problem"].rstrip()
        + "\nPlease reason step by step, and put your final answer within \\boxed{}."
    )
    assistant = example["solution"].rstrip()
    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    def _tok_ids(msgs, *, add_generation_prompt: bool) -> list[int]:
        # Newer transformers may return BatchEncoding when tokenize=True;
        # len(BatchEncoding)==2 (keys), so always go text→encode.
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
        return tokenizer.encode(text, add_special_tokens=False)

    prompt_ids = _tok_ids(messages[:1], add_generation_prompt=True)
    full_ids = _tok_ids(messages, add_generation_prompt=False)
    if len(full_ids) > max_length or len(full_ids) <= len(prompt_ids):
        return None
    loss_mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
    return {"input_ids": full_ids, "loss_mask": loss_mask}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument(
        "--config",
        default="default",
        help="MATH-lighteval builder config (default=all subjects).",
    )
    p.add_argument(
        "--hf-cache",
        default=__import__("os").environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")),
    )
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_cache)
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(args.hf_cache) / "datasets"))

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    raw = load_dataset(DATASET_ID, args.config)
    out = DatasetDict()
    for split, ds in raw.items():
        rows = []
        for ex in ds:
            row = convert_example(ex, tok, args.max_length)
            if row is not None:
                rows.append(row)
        out[split] = Dataset.from_list(rows)
        print(f"{split}: kept {len(out[split])}/{len(ds)} (max_length={args.max_length})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(str(args.output))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
