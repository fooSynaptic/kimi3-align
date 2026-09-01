#!/usr/bin/env python3
"""MATH boxed-acc probe: base Instruct vs E0 SFT ckpt (vLLM + math_verify)."""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset
from math_verify import parse, verify
from vllm import LLM, SamplingParams


def last_boxed(text: str) -> str | None:
    starts = [m.end() for m in re.finditer(r"\\boxed\s*\{", text)]
    if not starts:
        return None
    start = starts[-1]
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
    return None


def grade(pred: str, gold: str) -> float:
    try:
        gold_boxed = gold if "\\boxed" in gold else f"\\boxed{{{gold}}}"
        pred_boxed = pred if "\\boxed" in pred else (
            f"\\boxed{{{last_boxed(pred)}}}" if last_boxed(pred) else pred
        )
        g = parse(gold_boxed)
        p = parse(pred_boxed)
        return 1.0 if verify(g, p) else 0.0
    except Exception:
        gb = last_boxed(gold) or gold.strip()
        pb = last_boxed(pred)
        return 1.0 if pb is not None and pb.strip() == gb.strip() else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    args = ap.parse_args()

    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="test")
    # In-memory shuffle/select: datasets cache defaults to /tmp or $HOME, which fills small system disks on eval nodes.
    n = min(args.n, len(ds))
    idxs = list(range(len(ds)))
    random.Random(args.seed).shuffle(idxs)
    idxs = idxs[:n]

    prompts = []
    golds = []
    for i in idxs:
        ex = ds[i]
        user = (
            ex["problem"].rstrip()
            + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        )
        # chat template applied by tokenizer in vLLM when using messages via
        # apply_chat_template externally — build string here.
        prompts.append(
            "<|im_start|>user\n"
            + user
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        golds.append(ex["solution"])

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_tokens + 512,
        dtype="bfloat16",
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>"],
    )
    outs = llm.generate(prompts, params)

    correct = 0
    rows = []
    gen_lens: list[int] = []
    for i, (out, gold) in enumerate(zip(outs, golds)):
        text = out.outputs[0].text
        tok_ids = getattr(out.outputs[0], "token_ids", None) or []
        gen_len = int(len(tok_ids))
        gen_lens.append(gen_len)
        ok = grade(text, gold)
        correct += ok
        rows.append(
            {
                "i": i,
                "ok": ok,
                "gen_len": gen_len,
                "pred_boxed": last_boxed(text),
                "gold_boxed": last_boxed(gold),
            }
        )

    n = len(rows)
    acc = correct / n if n else 0.0
    mean_gen_len = (sum(gen_lens) / n) if n else 0.0
    summary = {
        "tag": args.tag,
        "model": args.model,
        "n": n,
        "correct": correct,
        "acc": acc,
        "mean_gen_len": mean_gen_len,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
