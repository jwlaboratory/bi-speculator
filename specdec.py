"""Greedy speculative decoding with per-region (thinking vs. answer) acceptance metrics.

Draft model proposes k tokens autoregressively; target verifies them in one
forward pass and accepts the longest prefix that matches its own greedy choice
(lossless for greedy decoding). Every proposed token is tagged with the region
it lands in — inside the <think>...</think> block or after it — so acceptance
rates can be compared across the two regimes.
"""

from dataclasses import dataclass, field

import torch
from transformers import DynamicCache


def crop_cache(cache: DynamicCache, length: int) -> None:
    """Roll a KV cache back to `length` tokens (rejected drafts are discarded)."""
    excess = cache.get_seq_length() - length
    if excess > 0:
        cache.crop(-excess)


@dataclass
class RegionStats:
    proposed: int = 0
    accepted: int = 0
    # Headroom: among verified slots, how often the target's greedy token was
    # in the draft's top-N. top-1 ~= acceptance; the top-1 -> top-N gap is the
    # ceiling a small reranking/logit-adjustment head could recover.
    verified: int = 0
    top3: int = 0
    top5: int = 0
    top10: int = 0

    @property
    def rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    def top_rate(self, n: int) -> float:
        return getattr(self, f"top{n}") / self.verified if self.verified else 0.0

    def add(self, other: "RegionStats") -> None:
        for f in ("proposed", "accepted", "verified", "top3", "top5", "top10"):
            setattr(self, f, getattr(self, f) + getattr(other, f))


@dataclass
class SpecDecResult:
    text: str
    new_tokens: int
    think: RegionStats
    answer: RegionStats
    cycles: int
    accepted_lengths: list[int] = field(default_factory=list)

    @property
    def overall(self) -> RegionStats:
        return RegionStats(
            self.think.proposed + self.answer.proposed,
            self.think.accepted + self.answer.accepted,
        )

    @property
    def mean_accepted_per_cycle(self) -> float:
        return sum(self.accepted_lengths) / len(self.accepted_lengths) if self.cycles else 0.0

    @property
    def tokens_per_target_forward(self) -> float:
        # Each cycle is exactly one target forward and yields accepted + 1 bonus token.
        return self.new_tokens / self.cycles if self.cycles else 0.0


@torch.no_grad()
def speculative_generate(
    target,
    draft,
    tokenizer,
    input_ids: torch.Tensor,
    *,
    k: int = 4,
    max_new_tokens: int = 512,
    draft_prefix_ids: torch.Tensor | None = None,
    head=None,
    head_flip_region: bool = False,
) -> SpecDecResult:
    """`draft_prefix_ids` is prepended to the draft's context only — the target
    never sees it. This is the zero-training stand-in for a mode input to the
    speculator (e.g. a system line saying it is in thinking mode).

    `head` is an optional trained DraftHead: proposals then come from the
    head's adjusted logits instead of the draft's own. `head_flip_region`
    feeds the head the wrong region bit (ablation for mode conditioning)."""
    think_open = tokenizer.convert_tokens_to_ids("<think>")
    think_close = tokenizer.convert_tokens_to_ids("</think>")
    eos = _eos_ids(target, tokenizer)

    d_offset = draft_prefix_ids.shape[1] if draft_prefix_ids is not None else 0
    d_input = (
        torch.cat([draft_prefix_ids, input_ids], dim=1) if d_offset else input_ids
    )

    t_cache, d_cache = DynamicCache(), DynamicCache()
    # Both caches cover the context minus its last token; that token is fed as
    # the first "pending" input so each model's next forward yields the logits
    # needed to continue the sequence.
    target(input_ids[:, :-1], past_key_values=t_cache, use_cache=True)
    draft(d_input[:, :-1], past_key_values=d_cache, use_cache=True)
    pending = input_ids[:, -1:]

    generated: list[int] = []
    in_think = False
    think, answer = RegionStats(), RegionStats()
    accepted_lengths: list[int] = []
    cycles = 0
    done = False

    while not done and len(generated) < max_new_tokens:
        cycles += 1

        # Draft proposes k tokens greedily, one forward each. Region state is
        # walked over the proposals so the head sees the right region bit.
        proposals: list[int] = []
        draft_top10: list[list[int]] = []
        regions: list[bool] = []
        state = in_think
        d_in = pending
        for _ in range(k):
            d_out = draft(
                d_in, past_key_values=d_cache, use_cache=True,
                output_hidden_states=head is not None,
            )
            if head is not None:
                region = int(state) ^ int(head_flip_region)
                d_logits = head(
                    d_out.hidden_states[-1][:, -1],
                    torch.tensor([region], device=input_ids.device),
                )
            else:
                d_logits = d_out.logits[:, -1]
            top10 = d_logits.topk(10).indices[0].tolist()
            draft_top10.append(top10)
            proposals.append(top10[0])
            regions.append(state)
            if top10[0] == think_open:
                state = True
            elif top10[0] == think_close:
                state = False
            d_in = torch.tensor([[top10[0]]], device=input_ids.device)

        # Target verifies all k proposals in a single forward pass.
        t_in = torch.cat(
            [pending, torch.tensor([proposals], device=input_ids.device)], dim=1
        )
        t_logits = target(t_in, past_key_values=t_cache, use_cache=True).logits
        t_pred = t_logits.argmax(-1)[0].tolist()  # k+1 greedy choices

        m = 0
        while m < k and proposals[m] == t_pred[m]:
            m += 1
        bonus = t_pred[m]
        accepted_lengths.append(m)

        # Tag each proposed slot with the region recorded during proposing.
        for i in range(len(proposals)):
            stats = think if regions[i] else answer
            stats.proposed += 1
            if i < m:
                stats.accepted += 1
            # Slots 0..m are "verified": the draft context there matches what
            # the target actually conditioned on, so top-N membership of the
            # target's token is a clean headroom measure.
            if i <= m:
                stats.verified += 1
                if t_pred[i] in draft_top10[i][:3]:
                    stats.top3 += 1
                if t_pred[i] in draft_top10[i][:5]:
                    stats.top5 += 1
                if t_pred[i] in draft_top10[i]:
                    stats.top10 += 1

        # Commit accepted tokens plus the target's bonus/correction token.
        for tok in proposals[:m] + [bonus]:
            generated.append(tok)
            if tok == think_open:
                in_think = True
            elif tok == think_close:
                in_think = False
            if tok in eos:
                done = True
                break

        # Roll caches back to the committed sequence; bonus becomes pending.
        prompt_len = input_ids.shape[1]
        committed = prompt_len + len(generated) - 1  # everything before bonus
        crop_cache(t_cache, committed)
        crop_cache(d_cache, committed + d_offset)
        pending = torch.tensor([[generated[-1]]], device=input_ids.device)

    generated = generated[:max_new_tokens]
    return SpecDecResult(
        text=tokenizer.decode(generated, skip_special_tokens=False),
        new_tokens=len(generated),
        think=think,
        answer=answer,
        cycles=cycles,
        accepted_lengths=accepted_lengths,
    )


@torch.no_grad()
def greedy_generate(target, tokenizer, input_ids: torch.Tensor, *, max_new_tokens: int = 512) -> str:
    """Plain greedy baseline: one target forward per token."""
    eos = _eos_ids(target, tokenizer)
    cache = DynamicCache()
    target(input_ids[:, :-1], past_key_values=cache, use_cache=True)
    pending = input_ids[:, -1:]
    generated: list[int] = []
    for _ in range(max_new_tokens):
        logits = target(pending, past_key_values=cache, use_cache=True).logits[:, -1]
        tok = int(logits.argmax(-1))
        generated.append(tok)
        if tok in eos:
            break
        pending = torch.tensor([[tok]], device=input_ids.device)
    return tokenizer.decode(generated, skip_special_tokens=False)


def _eos_ids(model, tokenizer) -> set[int]:
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    return set(eos) if isinstance(eos, (list, tuple)) else {eos}
