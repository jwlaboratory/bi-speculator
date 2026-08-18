"""EAGLE-3 speculative decoding for Qwen3-1.7B via vLLM.

Requires an NVIDIA GPU (will not run on macOS). Run on a GPU box with:
    pip install vllm
    python gpu/run_eagle3_vllm.py

vLLM logs per-step draft acceptance stats; pass --disable-log-stats=False if
you don't see them.
"""

from vllm import LLM, SamplingParams

TARGET = "Qwen/Qwen3-1.7B"
SPECULATOR = "AngelSlim/Qwen3-1.7B_eagle3"

llm = LLM(
    model=TARGET,
    speculative_config={
        "method": "eagle3",
        "model": SPECULATOR,
        "num_speculative_tokens": 4,
    },
    max_model_len=8192,
    gpu_memory_utilization=0.8,
)

prompt = (
    "A train leaves at 3:40pm traveling 80 km/h. A second train leaves the same "
    "station at 4:10pm at 100 km/h on a parallel track. At what time does the "
    "second train catch the first?"
)
messages = [{"role": "user", "content": prompt}]

for enable_thinking in (True, False):
    outputs = llm.chat(
        messages,
        SamplingParams(temperature=0.0, max_tokens=1024),
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )
    print(f"\n=== enable_thinking={enable_thinking} ===")
    print(outputs[0].outputs[0].text)
