"""DFlash speculative decoding for Qwen3-4B on Modal GPUs.

Runs two engines in parallel (DFlash vs. no-spec baseline), each through the
same phases: thinking @ 512 tokens, thinking @ 4096 (the long-length case),
and non-thinking @ 1024. vLLM's cumulative spec-decode counters are diffed
between phases, giving per-phase acceptance plus the per-position acceptance
profile across the 15-token draft block.

    modal run gpu/modal_dflash.py
"""

import json
import time

import modal

TARGET = "Qwen/Qwen3-4B"
DRAFTER = "z-lab/Qwen3-4B-DFlash-b16"
NUM_SPEC_TOKENS = 15

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "hf_transfer")
    # No CUDA toolkit in the slim image, so flashinfer can't JIT its sampling
    # kernels — use vLLM's native torch sampler instead (greedy anyway).
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
hf_cache = modal.Volume.from_name("bi-spec-hf-cache", create_if_missing=True)
app = modal.App("bi-speculator-dflash")

PROMPTS = [
    # Hard problems that produce long thinking traces.
    "Find all three-digit numbers that equal the sum of the cubes of their digits. "
    "Show your reasoning.",
    "How many trailing zeros does 2026! have? Explain carefully, then also find the "
    "last nonzero digit of 25!.",
    "Write a Python function that solves a 9x9 Sudoku using backtracking with "
    "constraint propagation, and analyze its worst-case complexity.",
    "A snail climbs 3 meters up a 30-meter well each day and slips back 2 meters "
    "each night. On which day does it escape? Then generalize to climbing a meters, "
    "slipping b, depth d: derive and prove the formula.",
    # Held-out prompts from the local experiments, for continuity.
    "A train leaves at 3:40pm traveling 80 km/h. A second train leaves the same "
    "station at 4:10pm at 100 km/h on a parallel track. At what time does the "
    "second train catch the first?",
    "Write a Python function that checks whether a string is a valid IPv6 address, "
    "without using the ipaddress module.",
    "If all bloops are razzies and some razzies are lazzies, can we conclude that "
    "some bloops are lazzies? Explain your reasoning.",
    "Explain in two paragraphs why the sky is blue during the day but red at sunset.",
]

PHASES = [
    {"thinking": True, "max_tokens": 512},
    {"thinking": True, "max_tokens": 4096},
    {"thinking": False, "max_tokens": 1024},
]


def spec_metric_snapshot(llm) -> dict:
    out = {}
    try:
        for m in llm.get_metrics():
            if "spec_decode" not in m.name:
                continue
            if hasattr(m, "value"):
                out[m.name] = m.value
            elif hasattr(m, "values"):
                out[m.name] = list(m.values)
    except Exception as e:  # noqa: BLE001 - metrics are best-effort
        out["metrics_error"] = str(e)
    return out


def diff(cur: dict, prev: dict) -> dict:
    out = {}
    for k, v in cur.items():
        if isinstance(v, (int, float)):
            out[k] = v - prev.get(k, 0)
        elif isinstance(v, list):
            p = prev.get(k, [0] * len(v))
            out[k] = [a - b for a, b in zip(v, p)]
    return out


@app.function(image=image, gpu="L40S", timeout=3600, volumes={"/root/.cache/huggingface": hf_cache})
def run_engine(use_dflash: bool) -> dict:
    from vllm import LLM, SamplingParams

    kwargs = {}
    if use_dflash:
        kwargs["speculative_config"] = {
            "method": "dflash",
            "model": DRAFTER,
            "num_speculative_tokens": NUM_SPEC_TOKENS,
        }
    llm = LLM(
        model=TARGET,
        max_model_len=8192,
        max_num_batched_tokens=32768,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        disable_log_stats=False,  # keep spec-decode acceptance counters
        **kwargs,
    )

    results = []
    prev = spec_metric_snapshot(llm)
    for phase in PHASES:
        sp = SamplingParams(temperature=0.0, max_tokens=phase["max_tokens"])
        messages = [[{"role": "user", "content": p}] for p in PROMPTS]
        start = time.time()
        outs = llm.chat(messages, sp, chat_template_kwargs={"enable_thinking": phase["thinking"]})
        secs = time.time() - start

        lengths = [len(o.outputs[0].token_ids) for o in outs]
        cur = spec_metric_snapshot(llm)
        results.append(
            {
                "phase": phase,
                "seconds": round(secs, 1),
                "gen_tokens": sum(lengths),
                "tok_per_s": round(sum(lengths) / secs, 1),
                "lengths": lengths,
                "spec_metrics": diff(cur, prev),
            }
        )
        prev = cur

    return {"dflash": use_dflash, "results": results}


@app.local_entrypoint()
def main(only: str = "both") -> None:
    configs = {"both": [True, False], "dflash": [True], "baseline": [False]}[only]
    handles = [run_engine.spawn(c) for c in configs]
    out = [h.get() for h in handles]
    print("===RESULTS===")
    print(json.dumps(out, indent=2))
