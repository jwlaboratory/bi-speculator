"""Prove/disprove: can speculators be specialized to thinking vs. non-thinking?

Three stages on Modal GPUs (Qwen3-0.6B draft, Qwen3-1.7B target, greedy):

  generate  - 2500 prompts (GSM8K + Alpaca) x both modes -> ~3M token traces.
  heads     - train 4 reranking heads on draft hidden states at MATCHED token
              budgets: mixed / region-conditioned / think-only / answer-only.
  finetune  - full-FT the 0.6B draft 3 ways at matched budgets:
              mixed / think-only / answer-only.

All variants evaluate on held-out prompts: per-region next-token agreement
with the target, plus simulated spec-dec accepted-run length (k=4).
Specialization is proven iff a specialist beats the matched-budget generalist
on its home region beyond noise (~0.3pt at n~100k).

    modal run gpu/modal_specialize.py --stage generate
    modal run gpu/modal_specialize.py --stage heads
    modal run gpu/modal_specialize.py --stage finetune
"""

import json

import modal

TARGET = "Qwen/Qwen3-1.7B"
DRAFT = "Qwen/Qwen3-0.6B"
TRACES = "/data/traces.jsonl"
POOL_CAP = 900_000  # per-region cap on training examples

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "datasets", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)
vol = modal.Volume.from_name("bi-spec-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("bi-spec-hf-cache", create_if_missing=True)
app = modal.App("bi-speculator-specialize")
volumes = {"/data": vol, "/root/.cache/huggingface": hf_cache}


def is_heldout(pid: int) -> bool:
    return pid % 10 == 0


def regions_for_output(out_ids, open_id, close_id):
    """Region in which each output token is *predicted* (state before it)."""
    regs, state = [], False
    for tok in out_ids:
        regs.append(state)
        if tok == open_id:
            state = True
        elif tok == close_id:
            state = False
    return regs


def sim_accept(match, k=4):
    """Simulate greedy spec-dec over a per-token match array; returns
    (accepted_draft_tokens, cycles, tokens_advanced)."""
    i = cycles = accepted = 0
    n = len(match)
    while i < n:
        c = 0
        while c < k and i + c < n and match[i + c]:
            c += 1
        accepted += c
        cycles += 1
        i += c + 1
    return accepted, cycles, n


def load_traces(min_out_tokens=3):
    out = []
    with open(TRACES) as f:
        for line in f:
            t = json.loads(line)
            if len(t["output_ids"]) >= min_out_tokens:
                out.append(t)
    return out


# ---------------------------------------------------------------- stage 1


@app.function(image=image, gpu="L40S", timeout=3600, volumes=volumes)
def generate() -> dict:
    import random

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    random.seed(0)
    alp = [r["instruction"] for r in load_dataset("tatsu-lab/alpaca", split="train") if not r["input"]]
    gsm = [r["question"] for r in load_dataset("openai/gsm8k", "main", split="train")]
    random.shuffle(alp)
    random.shuffle(gsm)
    prompts = alp[:1250] + gsm[:1250]
    random.Random(1).shuffle(prompts)

    llm = LLM(model=TARGET, max_model_len=4096, gpu_memory_utilization=0.85)
    counts = {}
    with open(TRACES, "w") as f:
        for thinking, max_tok in ((True, 1536), (False, 768)):
            msgs = [[{"role": "user", "content": p}] for p in prompts]
            outs = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=max_tok),
                            chat_template_kwargs={"enable_thinking": thinking})
            total = 0
            for pid, o in enumerate(outs):
                ids = list(o.outputs[0].token_ids)
                total += len(ids)
                f.write(json.dumps({"pid": pid, "thinking": thinking,
                                    "prompt_ids": list(o.prompt_token_ids),
                                    "output_ids": ids}) + "\n")
            counts[f"thinking={thinking}"] = total
    vol.commit()
    return counts


# ------------------------------------------------------- shared eval logic


def batched_forward(model, traces, device, batch_size=8, want_hidden=False):
    """Teacher-forced forward over traces (sorted by length), yielding
    (trace, preds_for_output_positions, hidden_or_None) per trace."""
    import torch

    order = sorted(range(len(traces)), key=lambda i: len(traces[i]["prompt_ids"]) + len(traces[i]["output_ids"]))
    pad = 0
    for s in range(0, len(order), batch_size):
        chunk = [traces[i] for i in order[s : s + batch_size]]
        seqs = [t["prompt_ids"] + t["output_ids"] for t in chunk]
        maxlen = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for r, x in enumerate(seqs):
            ids[r, : len(x)] = torch.tensor(x)
            mask[r, : len(x)] = 1
        with torch.no_grad():
            out = model(ids.to(device), attention_mask=mask.to(device),
                        output_hidden_states=want_hidden)
        for r, t in enumerate(chunk):
            plen, olen = len(t["prompt_ids"]), len(t["output_ids"])
            pos = torch.arange(plen - 1, plen + olen - 1)
            logits = out.logits[r, pos]
            hidden = out.hidden_states[-1][r, pos] if want_hidden else None
            yield t, logits, hidden


def eval_agreement(preds_by_trace, open_id, close_id):
    """preds_by_trace: list of (trace, pred_ids list). Returns per-region and
    per-mode aggregate metrics."""
    cells = {}  # (mode, region) -> [match, total]
    sims = {}  # mode -> [accepted, cycles, advanced]
    for t, preds in preds_by_trace:
        regs = regions_for_output(t["output_ids"], open_id, close_id)
        match = [int(p == y) for p, y in zip(preds, t["output_ids"])]
        mode = "think_mode" if t["thinking"] else "direct_mode"
        for mt, rg in zip(match, regs):
            key = (mode, "think" if rg else "answer")
            cells.setdefault(key, [0, 0])
            cells[key][0] += mt
            cells[key][1] += 1
        a, c, n = sim_accept(match)
        sims.setdefault(mode, [0, 0, 0])
        for j, v in enumerate((a, c, n)):
            sims[mode][j] += v
    out = {}
    region_tot = {}
    for (mode, region), (m, tot) in sorted(cells.items()):
        out[f"agree {mode}/{region}"] = {"n": tot, "rate": round(m / tot, 4)}
        rt = region_tot.setdefault(region, [0, 0])
        rt[0] += m
        rt[1] += tot
    for region, (m, tot) in region_tot.items():
        out[f"agree region={region} (all traces)"] = {"n": tot, "rate": round(m / tot, 4)}
    for mode, (a, c, n) in sims.items():
        out[f"specdec {mode}"] = {"mean_accepted_per_cycle_k4": round(a / c, 3),
                                  "tokens_per_target_forward": round(n / c, 3)}
    return out


# ---------------------------------------------------------------- stage 2


@app.function(image=image, gpu="L40S", timeout=5400, memory=65536, volumes=volumes)
def run_head(variant: str) -> dict:
    import random

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(TARGET)
    open_id = tok.convert_tokens_to_ids("<think>")
    close_id = tok.convert_tokens_to_ids("</think>")
    draft = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.bfloat16).to(device).eval()
    hidden_size = draft.config.hidden_size

    traces = load_traces()
    train_traces = [t for t in traces if not is_heldout(t["pid"])]
    held = [t for t in traces if is_heldout(t["pid"])]
    random.Random(0).shuffle(train_traces)

    # Build per-region training pools of (hidden, label).
    pools = {0: [], 1: []}
    base_match = {0: [0, 0], 1: [0, 0]}
    for t, logits, hidden in batched_forward(draft, train_traces, device, want_hidden=True):
        if len(pools[0]) >= POOL_CAP and len(pools[1]) >= POOL_CAP:
            break
        regs = regions_for_output(t["output_ids"], open_id, close_id)
        preds = logits.argmax(-1).tolist()
        for j, (rg, y) in enumerate(zip(regs, t["output_ids"])):
            r = int(rg)
            base_match[r][0] += int(preds[j] == y)
            base_match[r][1] += 1
            if len(pools[r]) < POOL_CAP:
                pools[r].append((hidden[j].to(torch.float16).cpu(), y))
    budget = min(len(pools[0]), len(pools[1]))

    if variant == "think":
        examples = [(h, y, 1) for h, y in pools[1][:budget]]
    elif variant == "answer":
        examples = [(h, y, 0) for h, y in pools[0][:budget]]
    else:  # mixed / region: balanced halves, same total budget
        examples = [(h, y, 1) for h, y in pools[1][: budget // 2]]
        examples += [(h, y, 0) for h, y in pools[0][: budget // 2]]
    random.Random(1).shuffle(examples)

    h_all = torch.stack([e[0] for e in examples])
    y_all = torch.tensor([e[1] for e in examples])
    r_all = torch.tensor([e[2] for e in examples])
    if variant != "region":
        r_all = torch.zeros_like(r_all)  # unconditioned variants ignore region

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(2, 16)
            self.net = nn.Sequential(nn.Linear(hidden_size + 16, 2048), nn.GELU(),
                                     nn.Linear(2048, hidden_size))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
            self.register_buffer("lm_w", draft.lm_head.weight.detach().float(), persistent=False)

        def forward(self, h, r):
            h = h.float()
            h = h + self.net(torch.cat([h, self.emb(r)], dim=-1))
            return h @ self.lm_w.T

    head = Head().to(device)
    opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.01)
    epochs, batch = 4, 2048
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for epoch in range(epochs):
        order = torch.randperm(len(y_all))
        tot = 0.0
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            logits = head(h_all[idx].to(device), r_all[idx].to(device))
            loss = F.cross_entropy(logits, y_all[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        print(f"[{variant}] epoch {epoch + 1}: loss {tot / len(order):.4f}", flush=True)

    # Held-out eval: head predictions per trace.
    head.eval()
    preds_by_trace = []
    with torch.no_grad():
        for t, _, hidden in batched_forward(draft, held, device, want_hidden=True):
            regs = regions_for_output(t["output_ids"], open_id, close_id)
            r = torch.tensor([int(x) for x in regs], device=device)
            if variant != "region":
                r = torch.zeros_like(r)
            preds = head(hidden.to(device), r).argmax(-1).tolist()
            preds_by_trace.append((t, preds))

    result = {"variant": variant, "budget": budget,
              "train_pool_base_agreement": {
                  "answer": round(base_match[0][0] / max(base_match[0][1], 1), 4),
                  "think": round(base_match[1][0] / max(base_match[1][1], 1), 4)},
              "heldout": eval_agreement(preds_by_trace, open_id, close_id)}
    print(json.dumps(result, indent=2), flush=True)
    return result


# ---------------------------------------------------------------- stage 3


@app.function(image=image, gpu="L40S", timeout=7200, memory=65536, volumes=volumes)
def run_finetune(variant: str) -> dict:
    import random

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(TARGET)
    open_id = tok.convert_tokens_to_ids("<think>")
    close_id = tok.convert_tokens_to_ids("</think>")

    traces = load_traces()
    train_traces = [t for t in traces if not is_heldout(t["pid"])]
    held = [t for t in traces if is_heldout(t["pid"])]
    random.Random(0).shuffle(train_traces)

    # Matched budget: min(total think, total answer) loss tokens.
    totals = {0: 0, 1: 0}
    for t in train_traces:
        for rg in regions_for_output(t["output_ids"], open_id, close_id):
            totals[int(rg)] += 1
    budget = min(totals.values())
    keep = {"mixed": (0, 1), "think": (1,), "answer": (0,)}[variant]

    train_set, masked = [], 0
    for t in train_traces:
        if masked >= budget:
            break
        regs = regions_for_output(t["output_ids"], open_id, close_id)
        sel = [j for j, rg in enumerate(regs) if int(rg) in keep]
        if not sel:
            continue
        masked += len(sel)
        train_set.append((t, set(sel)))

    model = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.float32).to(device)

    def evaluate(m) -> dict:
        m.eval()
        preds_by_trace = []
        with torch.autocast("cuda", torch.bfloat16):
            for t, logits, _ in batched_forward(m, held, device):
                preds_by_trace.append((t, logits.argmax(-1).tolist()))
        return eval_agreement(preds_by_trace, open_id, close_id)

    baseline = evaluate(model) if variant == "mixed" else None

    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)
    epochs, batch_tokens, accum = 2, 2048, 4

    # Sort by length so padded batches stay tight; shuffle batch order instead.
    train_set.sort(key=lambda it: len(it[0]["prompt_ids"]) + len(it[0]["output_ids"]))
    batches, cur, ctok = [], [], 0
    for item in train_set:
        cur.append(item)
        ctok += len(item[0]["prompt_ids"]) + len(item[0]["output_ids"])
        if ctok >= batch_tokens:
            batches.append(cur)
            cur, ctok = [], 0
    if cur:
        batches.append(cur)

    model.train()
    step = micro = 0
    for epoch in range(epochs):
        random.Random(epoch).shuffle(batches)
        for batch in batches:
            seqs = [t["prompt_ids"] + t["output_ids"] for t, _ in batch]
            maxlen = max(len(x) for x in seqs)
            ids = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            mask = torch.zeros_like(ids)
            labels = torch.full_like(ids, -100)
            for r, ((t, sel), x) in enumerate(zip(batch, seqs)):
                ids[r, : len(x)] = torch.tensor(x)
                mask[r, : len(x)] = 1
                plen = len(t["prompt_ids"])
                for j in sel:
                    labels[r, plen + j] = x[plen + j]
            with torch.autocast("cuda", torch.bfloat16):
                out = model(ids.to(device), attention_mask=mask.to(device))
                flat_labels = labels[:, 1:].reshape(-1).to(device)
                keep_rows = flat_labels != -100
                logits = out.logits[:, :-1].reshape(-1, out.logits.shape[-1])[keep_rows]
                loss = F.cross_entropy(logits.float(), flat_labels[keep_rows])
            del out, logits
            (loss / accum).backward()
            micro += 1
            if micro == accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                micro = 0
                step += 1
                if step % 25 == 0:
                    print(f"[{variant}] epoch {epoch + 1} step {step}: loss {float(loss):.4f}", flush=True)

    result = {"variant": variant, "budget_tokens": budget, "trained_tokens": masked,
              "heldout": evaluate(model)}
    if baseline is not None:
        result["baseline_heldout"] = baseline
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(stage: str = "all") -> None:
    results = {}
    if stage in ("generate", "all"):
        results["generate"] = generate.remote()
    if stage in ("heads", "all"):
        handles = {v: run_head.spawn(v) for v in ("mixed", "region", "think", "answer")}
        results["heads"] = {v: h.get() for v, h in handles.items()}
    if stage in ("finetune", "all"):
        handles = {v: run_finetune.spawn(v) for v in ("mixed", "think", "answer")}
        results["finetune"] = {v: h.get() for v, h in handles.items()}
    print("===RESULTS===")
    print(json.dumps(results, indent=2))
