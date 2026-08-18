"""Run speculative decoding (Qwen3-0.6B draft -> Qwen3-1.7B target) and report
acceptance rates inside vs. outside the <think> region.

Usage:
    uv run python run.py --prompt "..." --thinking --max-new-tokens 400
    uv run python run.py --no-thinking --baseline
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from specdec import greedy_generate, speculative_generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--draft", default="Qwen/Qwen3-0.6B")
    ap.add_argument(
        "--prompt",
        default="A train leaves at 3:40pm traveling 80 km/h. A second train leaves "
        "the same station at 4:10pm at 100 km/h on a parallel track. At what time "
        "does the second train catch the first?",
    )
    thinking = ap.add_mutually_exclusive_group()
    thinking.add_argument("--thinking", dest="thinking", action="store_true", default=True)
    thinking.add_argument("--no-thinking", dest="thinking", action="store_false")
    ap.add_argument("--k", type=int, default=4, help="draft tokens per cycle")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--baseline", action="store_true", help="also time plain greedy decoding")
    ap.add_argument("--show-text", action="store_true", help="print the generated text")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  target={args.target}  draft={args.draft}  k={args.k}  thinking={args.thinking}")

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16).to(device).eval()
    draft = AutoModelForCausalLM.from_pretrained(args.draft, dtype=torch.bfloat16).to(device).eval()

    messages = [{"role": "user", "content": args.prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=args.thinking,
        return_tensors="pt",
    )
    input_ids = (encoded if torch.is_tensor(encoded) else encoded["input_ids"]).to(device)

    start = time.perf_counter()
    result = speculative_generate(
        target, draft, tokenizer, input_ids, k=args.k, max_new_tokens=args.max_new_tokens
    )
    _sync(device)
    spec_secs = time.perf_counter() - start

    if args.show_text:
        print("\n--- output ---")
        print(result.text)
        print("--- end output ---\n")

    print(f"\n{'region':<10}{'proposed':>10}{'accepted':>10}{'rate':>8}")
    for name, stats in (("<think>", result.think), ("answer", result.answer), ("overall", result.overall)):
        print(f"{name:<10}{stats.proposed:>10}{stats.accepted:>10}{stats.rate:>8.1%}")

    print(f"\nnew tokens:               {result.new_tokens}")
    print(f"target forwards (cycles): {result.cycles}")
    print(f"mean accepted per cycle:  {result.mean_accepted_per_cycle:.2f} (of k={args.k})")
    print(f"tokens per target forward:{result.tokens_per_target_forward:>5.2f}")
    print(f"spec-dec throughput:      {result.new_tokens / spec_secs:.1f} tok/s")

    if args.baseline:
        start = time.perf_counter()
        text = greedy_generate(target, tokenizer, input_ids, max_new_tokens=args.max_new_tokens)
        _sync(device)
        base_secs = time.perf_counter() - start
        n = len(tokenizer.encode(text, add_special_tokens=False))
        print(f"baseline throughput:      {n / base_secs:.1f} tok/s")
        print(f"wall-clock speedup:       {(result.new_tokens / spec_secs) / (n / base_secs):.2f}x")


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
