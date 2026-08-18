"""Tiny reranking head: nudges the draft's output distribution toward the target.

The head takes the draft's final hidden state plus a region bit (inside
<think> vs. answer), applies a small residual MLP, and re-projects through the
draft's own frozen lm_head. ~4M trainable params.
"""

import torch
import torch.nn as nn


class DraftHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        lm_head_weight: torch.Tensor,
        *,
        region_dim: int = 16,
        inner: int = 2048,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.region_emb = nn.Embedding(2, region_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_size + region_dim, inner),
            nn.GELU(),
            nn.Linear(inner, hidden_size),
        )
        nn.init.zeros_(self.net[-1].weight)  # start as identity: head output = draft output
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("lm_w", lm_head_weight.detach().float(), persistent=False)
        norm_w = norm_weight.detach().float() if norm_weight is not None else None
        self.register_buffer("norm_w", norm_w, persistent=False)
        self.norm_eps = norm_eps

    def forward(self, h: torch.Tensor, region_id: torch.Tensor) -> torch.Tensor:
        h = h.float()
        if self.norm_w is not None:
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.norm_w
        x = torch.cat([h, self.region_emb(region_id)], dim=-1)
        h = h + self.net(x)
        return h @ self.lm_w.T


def save_head(head: DraftHead, config: dict, path: str) -> None:
    torch.save({"state": head.state_dict(), "config": config}, path)


def load_head(path: str, draft, device: str) -> DraftHead:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    head = DraftHead(
        cfg["hidden_size"],
        draft.lm_head.weight,
        region_dim=cfg["region_dim"],
        inner=cfg["inner"],
        norm_weight=draft.model.norm.weight if cfg["apply_norm"] else None,
        norm_eps=cfg["norm_eps"],
    )
    head.load_state_dict(ckpt["state"], strict=False)  # frozen buffers rebuilt above
    return head.to(device).eval()
