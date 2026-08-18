"""End-to-end eval: spec-dec acceptance with and without the trained head,
on the held-out prompts (never seen during head training).

    uv run python eval_head.py --max-new-tokens 300
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiment import PROMPTS
from head import load_head
from specdec import RegionStats, speculative_generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--draft", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--head", default="data/head.pt")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--out", default="head_results.json")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16).to(device).eval()
    draft = AutoModelForCausalLM.from_pretrained(args.draft, dtype=torch.bfloat16).to(device).eval()
    head = load_head(args.head, draft, device)

    conditions = [
        ("draft alone", dict(head=None)),
        ("draft + head", dict(head=head)),
        ("draft + head, wrong region bit", dict(head=head, head_flip_region=True)),
    ]

    results = {}
    for thinking in (True, False):
        mode = "think" if thinking else "direct"
        for label, kwargs in conditions:
            agg = {"think": RegionStats(), "answer": RegionStats()}
            cycles = new_tokens = 0
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
                    k=args.k, max_new_tokens=args.max_new_tokens, **kwargs,
                )
                agg["think"].add(r.think)
                agg["answer"].add(r.answer)
                cycles += r.cycles
                new_tokens += r.new_tokens
            secs = time.perf_counter() - start

            tpf = new_tokens / cycles if cycles else 0.0
            print(f"[{mode} | {label}]  tokens/target-forward {tpf:.2f}  ({secs:.0f}s)")
            row = {"tokens_per_forward": round(tpf, 3)}
            for region, s in agg.items():
                if not s.proposed:
                    continue
                print(f"  {region:<8} accept {s.rate:6.1%}  (n={s.proposed})   "
                      f"top3 {s.top_rate(3):6.1%}  top5 {s.top_rate(5):6.1%}")
                row[region] = dict(proposed=s.proposed, rate=round(s.rate, 4),
                                   top3=round(s.top_rate(3), 4), top5=round(s.top_rate(5), 4))
            results[f"{mode} | {label}"] = row

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
