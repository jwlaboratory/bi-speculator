"""Fine-tune z-lab/Qwen3-4B-DFlash-b16 on mode-split traces (mixed / think /
answer, matched budgets) and measure per-region block acceptance.

The DFlash training recipe is unreleased, so this implements the paper's
objective directly against z-lab's released architecture (their inference code
is the reference): teacher-force Qwen3-4B over our greedy traces, extract
hidden features from layers [1,9,17,25,33], and train the drafter to fill
16-token blocks [anchor, mask x15] non-causally, CE against the trace tokens
through the frozen target lm_head. Hyperparameters follow the proven DSpark
block-16 warm-start recipe from the sparklingtree repo (lr 1e-4, grad clip 1,
exp(-p/4) position weighting, ~32 anchors per step).

Traces are greedy, so teacher-forced block eval on held-out traces IS the
inference-time acceptance behavior (identical context), split cleanly by
region.

    modal run gpu/modal_dflash_train.py            # all three variants
"""

import json

import modal

TARGET = "Qwen/Qwen3-4B"
DRAFTER = "z-lab/Qwen3-4B-DFlash-b16"
TRACES = "/data/traces-Qwen3-4B.jsonl"
BLOCK = 16
MASK_ID = 151669
TARGET_LAYERS = [1, 9, 17, 25, 33]
GAMMA = 4.0  # exp(-p/gamma) position weighting from the proven recipe
ANCHORS_PER_STEP = 32

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "transformers==4.57.3", "accelerate", "safetensors",
                 "huggingface_hub", "hf_transfer")
    .run_commands("pip install --no-deps git+https://github.com/z-lab/dflash")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)
vol = modal.Volume.from_name("bi-spec-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("bi-spec-hf-cache", create_if_missing=True)
app = modal.App("bi-speculator-dflash-train")
volumes = {"/data": vol, "/root/.cache/huggingface": hf_cache}


def is_heldout(pid: int) -> bool:
    return pid % 10 == 0


def regions_for_output(out_ids, open_id, close_id):
    regs, state = [], False
    for tok in out_ids:
        regs.append(state)
        if tok == open_id:
            state = True
        elif tok == close_id:
            state = False
    return regs


def load_traces(min_out_tokens=BLOCK + 1):
    out = []
    with open(TRACES) as f:
        for line in f:
            t = json.loads(line)
            if len(t["output_ids"]) >= min_out_tokens:
                out.append(t)
    return out


def seq_blocks(t, open_id, close_id):
    """Non-overlapping 16-token blocks over the output: (anchor_pos, labels,
    label_regions). anchor_pos indexes the full sequence; labels are the 15
    tokens after it."""
    plen = len(t["prompt_ids"])
    full = t["prompt_ids"] + t["output_ids"]
    regs = regions_for_output(t["output_ids"], open_id, close_id)
    blocks = []
    a = plen - 1
    while a + BLOCK <= len(full):
        labels = full[a + 1 : a + BLOCK]
        label_regs = regs[a + 1 - plen : a + BLOCK - plen]
        blocks.append((a, labels, label_regs))
        a += BLOCK
    return full, blocks


@app.function(image=image, gpu="L40S", timeout=10800, memory=32768, volumes=volumes)
def train_eval(variant: str) -> dict:
    import math
    import random

    import torch
    import torch.nn.functional as F
    from dflash.model import DFlashDraftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    keep = {"mixed": (0, 1), "think": (1,), "answer": (0,)}[variant]

    tok = AutoTokenizer.from_pretrained(TARGET)
    open_id = tok.convert_tokens_to_ids("<think>")
    close_id = tok.convert_tokens_to_ids("</think>")

    target = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.bfloat16).to(device).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    draft = DFlashDraftModel.from_pretrained(
        DRAFTER, dtype=torch.float32, attn_implementation="sdpa"
    ).to(device)

    traces = load_traces()
    train_traces = [t for t in traces if not is_heldout(t["pid"])]
    held = [t for t in traces if is_heldout(t["pid"])]
    random.Random(0).shuffle(train_traces)

    embed = target.model.embed_tokens
    pos_weights = torch.exp(-torch.arange(BLOCK - 1, dtype=torch.float32, device=device) / GAMMA)

    def target_features(full_ids):
        ids = torch.tensor([full_ids], device=device)
        with torch.no_grad():
            out = target(ids, output_hidden_states=True)
        return torch.cat([out.hidden_states[l + 1] for l in TARGET_LAYERS], dim=-1)

    def draft_block_logits(feats, full_ids, anchors, grad=False):
        """One drafter forward for K anchors of the same sequence. Returns
        (K, 15, vocab) logits."""
        K, L = len(anchors), feats.shape[1]
        blocks = torch.tensor(
            [[full_ids[a]] + [MASK_ID] * (BLOCK - 1) for a in anchors], device=device
        )
        noise_emb = embed(blocks)
        anch = torch.tensor(anchors, device=device)
        ctx_pos = torch.arange(L, device=device).expand(K, L)
        blk_pos = anch.unsqueeze(1) + torch.arange(BLOCK, device=device)
        position_ids = torch.cat([ctx_pos, blk_pos], dim=1)
        # context visible up to and including the anchor; block fully visible
        allowed = torch.cat(
            [ctx_pos <= anch.unsqueeze(1), torch.ones_like(blk_pos, dtype=torch.bool)], dim=1
        )
        attn_mask = torch.zeros((K, 1, BLOCK, L + BLOCK), device=device)
        attn_mask.masked_fill_(~allowed.unsqueeze(1).unsqueeze(2), float("-inf"))
        with torch.autocast("cuda", torch.bfloat16):
            out = draft(
                position_ids=position_ids,
                attention_mask=attn_mask,
                noise_embedding=noise_emb,
                target_hidden=feats.expand(K, -1, -1),
            )
            logits = target.lm_head(out[:, 1 - BLOCK :, :])
        return logits

    @torch.no_grad()
    def evaluate() -> dict:
        draft.eval()
        cells = {}  # (mode, region) -> [match, total]
        acc = {}  # mode -> [accept_sum, blocks]
        for t in held:
            full, blocks = seq_blocks(t, open_id, close_id)
            if not blocks:
                continue
            feats = target_features(full)
            mode = "think_mode" if t["thinking"] else "direct_mode"
            for s in range(0, len(blocks), 64):
                chunk = blocks[s : s + 64]
                logits = draft_block_logits(feats, full, [b[0] for b in chunk])
                preds = logits.argmax(-1)
                for (a, labels, label_regs), pred in zip(chunk, preds):
                    match = [int(p == y) for p, y in zip(pred.tolist(), labels)]
                    for m, rg in zip(match, label_regs):
                        key = (mode, "think" if rg else "answer")
                        cells.setdefault(key, [0, 0])
                        cells[key][0] += m
                        cells[key][1] += 1
                    run = 0
                    while run < len(match) and match[run]:
                        run += 1
                    am = acc.setdefault(mode, [0, 0])
                    am[0] += run
                    am[1] += 1
        out = {}
        for (mode, region), (m, tot) in sorted(cells.items()):
            out[f"match {mode}/{region}"] = {"n": tot, "rate": round(m / tot, 4)}
        for mode, (s, b) in acc.items():
            out[f"accept_len {mode}"] = round(s / b, 3)
        return out

    # Matched budget: min over regions of total in-region label tokens.
    totals = {0: 0, 1: 0}
    for t in train_traces:
        _, blocks = seq_blocks(t, open_id, close_id)
        for _, _, label_regs in blocks:
            for rg in label_regs:
                totals[int(rg)] += 1
    budget = min(totals.values())

    baseline = evaluate() if variant == "mixed" else None

    # Collect training blocks up to budget.
    work, consumed = [], 0
    for t in train_traces:
        if consumed >= budget:
            break
        full, blocks = seq_blocks(t, open_id, close_id)
        sel = []
        for a, labels, label_regs in blocks:
            n_in = sum(1 for rg in label_regs if int(rg) in keep)
            if n_in:
                sel.append((a, labels, label_regs))
                consumed += n_in
        if sel:
            work.append((full, sel))

    n_steps = sum(math.ceil(len(sel) / ANCHORS_PER_STEP) for _, sel in work)
    opt = torch.optim.AdamW(draft.parameters(), lr=1e-4, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-4, total_steps=n_steps, pct_start=0.03, anneal_strategy="cos"
    )
    print(f"[{variant}] budget={budget} tokens, {len(work)} seqs, {n_steps} steps", flush=True)

    draft.train()
    step = 0
    for full, sel in work:
        feats = target_features(full)
        for s in range(0, len(sel), ANCHORS_PER_STEP):
            chunk = sel[s : s + ANCHORS_PER_STEP]
            logits = draft_block_logits(feats, full, [b[0] for b in chunk], grad=True)
            labels = torch.full((len(chunk), BLOCK - 1), -100, dtype=torch.long, device=device)
            for r, (_, labs, label_regs) in enumerate(chunk):
                for j, (y, rg) in enumerate(zip(labs, label_regs)):
                    if int(rg) in keep:
                        labels[r, j] = y
            ce = F.cross_entropy(
                logits.float().transpose(1, 2), labels, ignore_index=-100, reduction="none"
            )
            mask = (labels != -100).float()
            w = pos_weights.unsqueeze(0) * mask
            loss = (ce * w).sum() / w.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"[{variant}] step {step}/{n_steps}: loss {float(loss):.4f}", flush=True)

    result = {"variant": variant, "budget": budget, "steps": n_steps, "heldout": evaluate()}
    if baseline is not None:
        result["baseline_heldout"] = baseline
    print(json.dumps(result, indent=2), flush=True)

    from safetensors.torch import save_file
    import os

    os.makedirs(f"/data/dflash_ft_{variant}", exist_ok=True)
    save_file(
        {k: v.to(torch.bfloat16).contiguous() for k, v in draft.state_dict().items()},
        f"/data/dflash_ft_{variant}/model.safetensors",
    )
    vol.commit()
    return result


@app.local_entrypoint()
def main(only: str = "") -> None:
    variants = [only] if only else ["mixed", "think", "answer"]
    handles = {v: train_eval.spawn(v) for v in variants}
    out = {v: h.get() for v, h in handles.items()}
    print("===RESULTS===")
    print(json.dumps(out, indent=2))
