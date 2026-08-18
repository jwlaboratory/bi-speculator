"""Train the reranking head on collected (draft hidden, target token) pairs.

    uv run python train_head.py --epochs 30
"""

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from head import DraftHead, save_head


def accuracy(head, h, labels, regions, device, batch=2048) -> float:
    correct = 0
    with torch.no_grad():
        for i in range(0, len(labels), batch):
            logits = head(h[i : i + batch].to(device), regions[i : i + batch].to(device))
            correct += int((logits.argmax(-1).cpu() == labels[i : i + batch]).sum())
    return correct / len(labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/head_data.pt")
    ap.add_argument("--out", default="data/head.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--inner", type=int, default=2048)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load(args.data)
    h, labels = data["h"], data["labels"]
    regions = data["regions"].long()
    n = len(labels)

    draft = AutoModelForCausalLM.from_pretrained(data["draft_model"], dtype=torch.bfloat16)
    hidden_size = h.shape[1]
    head = DraftHead(
        hidden_size,
        draft.lm_head.weight,
        inner=args.inner,
        norm_weight=draft.model.norm.weight if data["apply_norm"] else None,
    ).to(device)
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    n_val = max(n // 20, 1)
    val, train = perm[:n_val], perm[n_val:]

    def split(idx):
        return h[idx], labels[idx], regions[idx]

    h_tr, y_tr, r_tr = split(train)
    h_va, y_va, r_va = split(val)

    # Baseline: the draft's own top-1 vs the target token.
    base_all = (data["draft_top1"] == labels).float().mean()
    base_val = (data["draft_top1"][val] == labels[val]).float().mean()
    print(f"examples={n} (val={n_val})  head params={n_params / 1e6:.1f}M  device={device}")
    print(f"draft top-1 = target: {base_all:.1%} overall, {base_val:.1%} on val split")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_val, best_state = 0.0, None

    for epoch in range(args.epochs):
        head.train()
        order = torch.randperm(len(y_tr), generator=g)
        total_loss = 0.0
        for i in range(0, len(order), args.batch):
            idx = order[i : i + args.batch]
            logits = head(h_tr[idx].to(device), r_tr[idx].to(device))
            loss = F.cross_entropy(logits, y_tr[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss) * len(idx)
        sched.step()
        head.eval()
        val_acc = accuracy(head, h_va, y_va, r_va, device)
        marker = ""
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            marker = " *"
        print(f"epoch {epoch + 1:>3}: loss {total_loss / len(order):.3f}  val top-1 {val_acc:.1%}{marker}", flush=True)

    head.load_state_dict(best_state)
    head.eval()

    # Final report incl. region breakdown and a flipped-region-bit ablation.
    for name, mask in (("think", r_va == 1), ("answer", r_va == 0)):
        if int(mask.sum()) == 0:
            continue
        acc = accuracy(head, h_va[mask], y_va[mask], r_va[mask], device)
        base = (data["draft_top1"][val][mask] == y_va[mask]).float().mean()
        flipped = accuracy(head, h_va[mask], y_va[mask], 1 - r_va[mask], device)
        print(f"val {name:<7}: draft {base:.1%} -> head {acc:.1%}  (wrong region bit: {flipped:.1%})")

    save_head(
        head,
        {
            "hidden_size": hidden_size,
            "region_dim": 16,
            "inner": args.inner,
            "apply_norm": bool(data["apply_norm"]),
            "norm_eps": 1e-6,
        },
        args.out,
    )
    print(f"saved best head (val {best_val:.1%}) to {args.out}")


if __name__ == "__main__":
    main()
