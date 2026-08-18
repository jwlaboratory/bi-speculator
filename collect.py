"""Collect training data for the reranking head.

Generates greedy target traces on 40 training prompts (both thinking and
non-thinking mode), teacher-forces the draft over each trace, and stores
(draft hidden state, target next token, region bit) tuples. These positions
match the spec-dec "verified slot" distribution exactly.

    uv run python collect.py --max-new-tokens 300
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TRAIN_PROMPTS = [
    "If a rectangle's length is twice its width and its perimeter is 36 cm, what is its area?",
    "A store discounts an $80 jacket by 25%, then adds 8% sales tax. What is the final price?",
    "What is the sum of all integers from 1 to 200 that are divisible by 3?",
    "Two dice are rolled. What is the probability that the sum is 8?",
    "A car uses 6 liters of fuel per 100 km. How much fuel does it need for a 450 km trip?",
    "Solve for x: 3(x - 4) + 5 = 2x + 9",
    "Pump A fills a tank in 6 hours and pump B in 4 hours. How long do both together take?",
    "Compute 17 * 23 mentally and explain the trick you used.",
    "Write a Python function that reverses the order of words in a sentence.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Explain the difference between a list and a tuple in Python.",
    "Write a SQL query to find the second-highest salary in an employees table.",
    "Write a JavaScript function that debounces another function.",
    "What does this Python expression produce: [x*x for x in range(10) if x % 2 == 0]?",
    "Write a Python generator that yields all primes up to n.",
    "Explain what a race condition is, with a simple example.",
    "Explain how vaccines train the immune system.",
    "Why do metals feel colder than wood at the same room temperature?",
    "Explain photosynthesis in simple terms.",
    "What causes tides, and why are there two high tides per day?",
    "How does GPS determine your position?",
    "Why does ice float on water?",
    "Explain the greenhouse effect.",
    "How do noise-cancelling headphones work?",
    "A farmer has chickens and cows: 30 heads and 74 legs in total. How many of each?",
    "Explain the 'missing dollar' hotel riddle and where the reasoning goes wrong.",
    "If 5 machines take 5 minutes to make 5 widgets, how long do 100 machines take to make 100 widgets?",
    "You have a 3-liter jug and a 5-liter jug. How do you measure exactly 4 liters?",
    "Write a haiku about autumn rain.",
    "Write a short product description for a solar-powered camping lantern.",
    "Draft a polite email declining a meeting invitation.",
    "Write a two-sentence horror story.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "What were the main achievements of the Apollo program?",
    "Explain the difference between weather and climate.",
    "Who was Ada Lovelace and why is she significant?",
    "Summarize how the DNS system of the internet works.",
    "What is inflation and what causes it?",
    "Describe the water cycle.",
    "What is the difference between a virus and a bacterium?",
]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--draft", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--out", default="data/head_data.pt")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16).to(device).eval()
    draft = AutoModelForCausalLM.from_pretrained(args.draft, dtype=torch.bfloat16).to(device).eval()

    think_open = tokenizer.convert_tokens_to_ids("<think>")
    think_close = tokenizer.convert_tokens_to_ids("</think>")

    hs, labels, regions, modes, draft_top1 = [], [], [], [], []
    recon_match = recon_total = 0

    for thinking in (True, False):
        for p_idx, prompt in enumerate(TRAIN_PROMPTS):
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                enable_thinking=thinking,
                return_tensors="pt",
            )
            input_ids = (encoded if torch.is_tensor(encoded) else encoded["input_ids"]).to(device)
            prompt_len = input_ids.shape[1]
            full = target.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )

            out = draft(full, output_hidden_states=True)
            h = out.hidden_states[-1][0]  # (L, hidden)
            logits = out.logits[0]

            # Sanity check: lm_head(h) should reproduce the draft's own logits.
            rec = (h[:50].float() @ draft.lm_head.weight.float().T).argmax(-1)
            recon_match += int((rec == logits[:50].argmax(-1)).sum())
            recon_total += 50

            L = full.shape[1]
            state = False
            for i in range(prompt_len - 1, L - 1):
                tok = int(full[0, i])
                if tok == think_open:
                    state = True
                elif tok == think_close:
                    state = False
                hs.append(h[i].to(torch.float16).cpu())
                labels.append(int(full[0, i + 1]))
                regions.append(int(state))
                modes.append(int(thinking))
                draft_top1.append(int(logits[i].argmax()))
            print(f"mode={'think' if thinking else 'direct'} prompt {p_idx + 1}/{len(TRAIN_PROMPTS)}: "
                  f"{L - prompt_len} tokens ({len(labels)} total examples)", flush=True)

    apply_norm = recon_match / recon_total < 0.99
    print(f"\nlm_head reconstruction match: {recon_match / recon_total:.1%} -> apply_norm={apply_norm}")
    if apply_norm:
        raise SystemExit("hidden_states[-1] is pre-norm on this transformers version; "
                         "re-collect applying draft.model.norm first.")

    Path(args.out).parent.mkdir(exist_ok=True)
    torch.save(
        {
            "h": torch.stack(hs),
            "labels": torch.tensor(labels),
            "regions": torch.tensor(regions, dtype=torch.uint8),
            "modes": torch.tensor(modes, dtype=torch.uint8),
            "draft_top1": torch.tensor(draft_top1),
            "apply_norm": apply_norm,
            "draft_model": args.draft,
            "target_model": args.target,
        },
        args.out,
    )
    n = len(labels)
    in_think = sum(regions)
    print(f"saved {n} examples to {args.out} ({in_think} think-region, {n - in_think} answer-region)")


if __name__ == "__main__":
    main()
