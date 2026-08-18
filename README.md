# bi-speculator

Experiments on mode-aware speculative decoding: do draft models behave
differently inside a target's `<think>...</think>` reasoning region vs. its
final answer, and is it worth specializing the speculator (separate drafts,
mode-conditioned heads, or per-mode logit adjustment) for each regime?

## Local harness (runs on Apple Silicon / any torch device)

Plain greedy speculative decoding with Qwen3-0.6B drafting for Qwen3-1.7B,
implemented from scratch in `specdec.py` so every proposed token can be tagged
with the region it lands in. Reports acceptance rate per region, mean accepted
tokens per verification cycle, and throughput vs. a plain greedy baseline.

```sh
uv sync
uv run python run.py --thinking --max-new-tokens 400 --baseline
uv run python run.py --no-thinking --max-new-tokens 400 --baseline
```

Useful flags: `--k` (draft tokens per cycle), `--prompt`, `--show-text`,
`--target` / `--draft` (any same-tokenizer HF pair).

Note: on MPS the wall-clock speedup is modest-to-negative — a 0.6B draft is a
third of the target's cost and runs k sequential forwards per cycle. The
harness is for measuring *acceptance behavior* per region; real speedups need
a tiny trained head (EAGLE-3/DFlash style) and a GPU.

## Findings so far (Qwen3-0.6B draft -> Qwen3-1.7B target, greedy)

- Teacher-forced on the target's own trajectory, the draft already predicts
  the target's greedy token **78.8%** of the time; the target's token is in
  the draft's top-5 ~91% (`collect.py` / `experiment.py`). Headroom for a
  reranking head is therefore ~+12pt, not the ~50pt a naive read of
  spec-dec acceptance-per-proposed suggests.
- A tiny reranking head (residual MLP + frozen draft lm_head, `head.py`,
  tried at 4.2M and 0.5M params) trained on 22k collected tokens **does not
  beat the draft baseline** (78.0% vs 78.1% val top-1) — it memorizes the
  training set instead. EAGLE-style heads need orders of magnitude more
  data; laptop-scale collection is the bottleneck, not the architecture.
- Mode/region conditioning shows **no signal in three separate tests**:
  draft-only system-prompt hints (±1pt, within noise, `experiment.py`),
  a learned region bit, and a wrong-region-bit ablation (78.5% vs 78.4%,
  `train_head.py`). Thinking-region and answer-region stats are nearly
  identical throughout — specializing the speculator by mode looks
  low-value for this pair.

Pipeline: `collect.py` -> `train_head.py` -> `eval_head.py` (end-to-end
acceptance with the head plugged into the spec-dec loop).

## Verdict: mode specialization (gpu/modal_specialize.py)

Decisive experiment on Modal GPUs: 3.7M tokens of Qwen3-1.7B traces (2500
GSM8K+Alpaca prompts x both modes), then matched-token-budget training of
(a) four reranking heads and (b) three full fine-tunes of the 0.6B draft,
evaluated on held-out prompts (n=230k think-region / 138k answer-region).

Held-out next-token agreement with the target, per region:

| variant                  | think  | answer |
|--------------------------|--------|--------|
| draft baseline           | 82.9%  | 81.3%  |
| head: mixed              | 83.2%  | 80.6%  |
| head: region-conditioned | 83.2%  | 80.5%  |
| head: think-specialist   | 83.6%  | 78.2%  |
| head: answer-specialist  | 79.2%  | 81.1%  |
| FT: mixed                | 84.7%  | 82.4%  |
| FT: think-only           | 84.8%  | 81.3%  |
| FT: answer-only          | 80.9%  | 82.1%  |

1. **Mode-conditioning is disproven** (4 independent tests): a region-bit
   input changes nothing, at 22k or 900k training examples.
2. **Separate specialists: real but useless.** At head capacity specialists
   beat the generalist by +0.4-0.5pt on home turf (5-6 sigma); at full-model
   capacity the gap vanishes (+0.16 think / -0.29 answer vs. mixed FT) —
   interference, not distribution mismatch, was the binding constraint.
   Cross-region penalty is 2.4-3.9pt, so misrouting is costly. Two models
   also cost 2x memory + routing complexity for <1% wall-clock.
3. **What works: one mixed on-policy-distilled speculator** (+1.8pt think /
   +1.1pt answer over baseline, tokens/forward 3.45 -> 3.57 at k=4), and
   mode-aware draft *length* (see DFlash per-position decay above).

### Does target scale change the verdict? No.

Same experiment rerun with target = Qwen3-8B (same 0.6B draft, so 5x less
relative draft capacity — the case where specialization should help most if
capacity interference were mode-specific). Specialist-minus-generalist gap
on the home region:

| capacity        | 1.7B target      | 8B target        |
|-----------------|------------------|------------------|
| head (~4M)      | +0.46 / +0.53 pt | +0.38 / +0.55 pt |
| full model FT   | +0.16 / -0.29 pt | +0.04 / -0.07 pt |
| region-bit head | +0.0 pt          | +0.0 pt          |

The gap is scale-invariant at head capacity and zero at full capacity for
both targets; mode-conditioning failed a 5th and 6th time. The verdict
holds across target scale.

## DFlash on Modal GPUs (Qwen3-4B + z-lab/Qwen3-4B-DFlash-b16)

`gpu/modal_dflash.py` (`modal run gpu/modal_dflash.py [--only dflash|baseline]`)
runs the trained DFlash block-diffusion drafter vs. a no-spec baseline on
L40S GPUs, batch of 8 prompts, greedy. Results (vLLM 0.27.1):

| phase              | accept len/step | tok/s (dflash) | tok/s (base) | speedup |
|--------------------|-----------------|----------------|--------------|---------|
| thinking @ 512     | 2.94 of 16      | 765            | 616          | 1.24x   |
| thinking @ 4096    | 3.66 of 16      | 1124           | 445          | 2.53x   |
| non-thinking @1024 | 5.33 of 16      | 1468           | 442          | 3.32x   |

- Acceptance *improves* deeper into thinking traces: tokens 512-4096 accept at
  ~3.8/step vs 2.9/step for the first 512 — longer generations speculate
  better, confirming the length hypothesis.
- Non-thinking answer text drafts best of all (5.3/step, 3.3x) — again,
  thinking text is the *harder* regime for the speculator, not the easier one.
- Per-position acceptance decays steeply in think mode (77% at draft pos 0,
  ~14% by pos 5) and much more gently in answer mode (83% -> 27% at pos 5),
  so long 15-token draft blocks pay off mainly outside the think region —
  an argument for mode-aware *draft length* rather than mode-aware drafts.

## GPU: trained EAGLE-3 speculator

`gpu/run_eagle3_vllm.py` runs Qwen3-1.7B with the trained
[AngelSlim/Qwen3-1.7B_eagle3](https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3)
head under vLLM — the smallest known-good pretrained speculator. Requires an
NVIDIA GPU (Modal / RunPod / Colab); it will not run on macOS.
