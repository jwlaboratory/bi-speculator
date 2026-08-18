"""Viability sweep for mode-specialized speculators.

For each prompt and each target mode (thinking / non-thinking), runs spec-dec
with the draft either unconditioned or given a draft-only system prefix that
tells it which mode it is in (the zero-training stand-in for a mode input).
Aggregates per-region acceptance and top-N headroom across prompts.

    uv run python experiment.py --max-new-tokens 300
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from specdec import RegionStats, speculative_generate

PROMPTS = [
    "A train leaves at 3:40pm traveling 80 km/h. A second train leaves the same "
    "station at 4:10pm at 100 km/h on a parallel track. At what time does the "
    "second train catch the first?",
    "Write a Python function that checks whether a string is a valid IPv6 address, "
    "without using the ipaddress module.",
    "Explain in two paragraphs why the sky is blue during the day but red at sunset.",
    "If all bloops are razzies and some razzies are lazzies, can we conclude that "
    "some bloops are lazzies? Explain your reasoning.",
    "Summarize the main causes of World War I in a few sentences.",
]

# Draft-only system prefixes: the target never sees these.
THINK_PREFIX = (
    "<|im_start|>system\nYou think in long, careful step-by-step chains of "
    "reasoning inside <think> tags before answering, frequently double-checking "
    "yourself.<|im_end|>\n"
)
DIRECT_PREFIX = (
    "<|im_start|>system\nYou answer immediately and concisely, with no "
    "deliberation.<|im_end|>\n"
)

CONDITIONS = [
    # (label, target thinking mode, draft prefix)
    ("think / draft plain", True, None),
    ("think / draft mode-hint", True, THINK_PREFIX),
    ("think / draft wrong-hint", True, DIRECT_PREFIX),
    ("direct / draft plain", False, None),
    ("direct / draft mode-hint", False, DIRECT_PREFIX),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--draft", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16).to(device).eval()
    draft = AutoModelForCausalLM.from_pretrained(args.draft, dtype=torch.bfloat16).to(device).eval()

    results = {}
    for label, thinking, prefix in CONDITIONS:
        prefix_ids = None
        if prefix is not None:
            prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

        agg = {"think": RegionStats(), "answer": RegionStats()}
        start = time.perf_counter()
        for prompt in PROMPTS:
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                enable_thinking=thinking,
                return_tensors="pt",
            )
            input_ids = (encoded if torch.is_tensor(encoded) else encoded["input_ids"]).to(device)
            r = speculative_generate(
                target, draft, tokenizer, input_ids,
                k=args.k, max_new_tokens=args.max_new_tokens,
                draft_prefix_ids=prefix_ids,
            )
            agg["think"].add(r.think)
            agg["answer"].add(r.answer)
        secs = time.perf_counter() - start

        results[label] = {
            region: dict(
                proposed=s.proposed, accepted=s.accepted, rate=round(s.rate, 4),
                verified=s.verified,
                top3=round(s.top_rate(3), 4), top5=round(s.top_rate(5), 4),
                top10=round(s.top_rate(10), 4),
            )
            for region, s in agg.items()
        }
        print(f"[{label}]  ({secs:.0f}s)")
        for region, s in agg.items():
            if not s.proposed:
                continue
            print(
                f"  {region:<8} accept {s.rate:6.1%}  (n={s.proposed})   "
                f"target-token-in-draft top3 {s.top_rate(3):6.1%}  "
                f"top5 {s.top_rate(5):6.1%}  top10 {s.top_rate(10):6.1%}"
            )

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
