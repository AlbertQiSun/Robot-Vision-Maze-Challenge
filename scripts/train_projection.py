#!/usr/bin/env python3
"""
Train DINOv2 projection head for place recognition.

Two-phase training:
  Phase A — frozen backbone, train GeM + Projection   (default 30 epochs)
  Phase B — unfreeze last 2 ViT blocks + heads         (default 20 epochs)

Usage
-----
  # Single-maze quick-start:
  python scripts/train_projection.py --data-dir data/ --single-maze --device cuda

  # Multi-maze full training:
  python scripts/train_projection.py --data-dir training_data/ --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from vis_nav.config import paths, feat_cfg, train_cfg
from vis_nav.utils import get_device, ensure_dirs
from vis_nav.models import PlaceFeatureExtractor
from vis_nav.data import (
    MazeExplorationDataset,
    PKSampler,
    get_train_transform,
    get_inference_transform,
)


# ── Multi-Similarity Loss ───────────────────────────────────────────────
class MultiSimilarityLoss(nn.Module):
    def __init__(
        self,
        alpha: float = train_cfg.ms_alpha,
        beta: float = train_cfg.ms_beta,
        base: float = train_cfg.ms_base,
    ):
        super().__init__()
        self.alpha, self.beta, self.base = alpha, beta, base

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sim = embeddings @ embeddings.T
        # Use a graph-connected zero so .backward() works even if no valid pairs
        loss = (embeddings * 0).sum()
        valid = 0

        for i in range(len(embeddings)):
            pos_mask = (labels == labels[i]).clone()
            pos_mask[i] = False
            neg_mask = labels != labels[i]

            pos_sim = sim[i][pos_mask]
            neg_sim = sim[i][neg_mask]
            if len(pos_sim) == 0 or len(neg_sim) == 0:
                continue

            neg_sim = neg_sim[neg_sim > pos_sim.min() - 0.1]
            pos_sim = pos_sim[pos_sim < neg_sim.max() + 0.1] if len(neg_sim) > 0 else pos_sim
            if len(pos_sim) == 0 or len(neg_sim) == 0:
                continue

            loss += (
                (1 / self.alpha)
                * torch.log(1 + torch.sum(torch.exp(-self.alpha * (pos_sim - self.base))))
                + (1 / self.beta)
                * torch.log(1 + torch.sum(torch.exp(self.beta * (neg_sim - self.base))))
            )
            valid += 1

        return loss / max(valid, 1)


# ── Training helpers ────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, n = 0.0, 0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        if labels.unique().numel() < 2:
            continue
        loss = criterion(model(images), labels)
        if torch.isnan(loss) or torch.isinf(loss) or not loss.requires_grad:
            continue
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}")
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    feats, lbls = [], []
    for imgs, lab in loader:
        feats.append(model(imgs.to(device)).cpu())
        lbls.append(lab)
    feats, lbls = torch.cat(feats), torch.cat(lbls)
    sim = feats @ feats.T
    sim.fill_diagonal_(-2)
    recalls = {}
    for k in (1, 5, 10):
        _, topk = sim.topk(k, dim=1)
        recalls[k] = sum(lbls[i] in lbls[topk[i]] for i in range(len(lbls))) / len(lbls)
    return recalls


# ── Discover mazes ──────────────────────────────────────────────────────
def find_maze_dirs(data_dir: str, single: bool) -> list[str]:
    if single:
        return [data_dir]
    dirs = sorted(
        os.path.join(data_dir, d) for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and os.path.exists(os.path.join(data_dir, d, "data_info.json"))
    )
    if not dirs and os.path.exists(os.path.join(data_dir, "data_info.json")):
        dirs = [data_dir]
    return dirs


# ── Main ────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="training_data/")
    ap.add_argument("--single-maze", action="store_true")
    ap.add_argument("--texture-dir", default=paths.texture_dir)
    ap.add_argument("--output", default=paths.projection_head)
    ap.add_argument("--backbone", default=feat_cfg.backbone_name)
    ap.add_argument("--output-dim", type=int, default=feat_cfg.projection_dim)
    ap.add_argument("--epochs-phase-a", type=int, default=train_cfg.epochs_phase_a)
    ap.add_argument("--epochs-phase-b", type=int, default=train_cfg.epochs_phase_b)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=train_cfg.lr_heads)
    ap.add_argument("--lr-backbone", type=float, default=train_cfg.lr_backbone)
    ap.add_argument("--subsample", type=int, default=train_cfg.subsample_rate)
    ap.add_argument("--synthetic-ratio", type=float, default=train_cfg.synthetic_ratio)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = get_device() if args.device == "auto" else torch.device(args.device)
    ensure_dirs()
    print(f"Device: {device}")

    data_dirs = find_maze_dirs(args.data_dir, args.single_maze)
    if not data_dirs:
        print(f"ERROR: no maze data in {args.data_dir}")
        return
    print(f"Found {len(data_dirs)} maze(s)")

    # Datasets
    train_ds = MazeExplorationDataset(
        data_dirs, get_train_transform(), args.subsample,
        args.texture_dir, args.synthetic_ratio,
    )
    val_ds = MazeExplorationDataset(
        data_dirs, get_inference_transform(), args.subsample,
    )
    sampler = PKSampler(train_ds.labels, P=train_cfg.pk_P,
                        K=max(2, args.batch_size // train_cfg.pk_P))
    train_dl = DataLoader(train_ds, args.batch_size, sampler=sampler,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # Model
    model = PlaceFeatureExtractor(
        args.backbone, args.output_dim, freeze_backbone=True,
    ).to(device)
    criterion = MultiSimilarityLoss()
    best_loss = float("inf")

    # ── Phase A ──────────────────────────────────────────────────────
    print("\n" + "=" * 60 + "\nPHASE A: frozen backbone\n" + "=" * 60)
    head_params = list(model.gem.parameters()) + list(model.projection.parameters())
    opt_a = AdamW(head_params, lr=args.lr, weight_decay=train_cfg.weight_decay)
    sched_a = CosineAnnealingWarmRestarts(opt_a, T_0=train_cfg.cosine_T0, T_mult=2)

    for ep in range(1, args.epochs_phase_a + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_dl, criterion, opt_a, device, ep)
        sched_a.step()
        print(f"Epoch {ep}/{args.epochs_phase_a} | Loss {loss:.4f} | "
              f"LR {opt_a.param_groups[0]['lr']:.6f} | {time.time()-t0:.1f}s")
        if ep % 5 == 0:
            r = validate(model, val_dl, device)
            print(f"  R@1={r[1]:.4f}  R@5={r[5]:.4f}  R@10={r[10]:.4f}")
        if loss < best_loss:
            best_loss = loss
            model.save_heads(args.output)

    # ── Phase B ──────────────────────────────────────────────────────
    if args.epochs_phase_b > 0:
        print("\n" + "=" * 60 + "\nPHASE B: unfreeze last 2 blocks\n" + "=" * 60)
        model.unfreeze_last_n_blocks(train_cfg.unfreeze_blocks)
        bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
        opt_b = AdamW([
            {"params": bb_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr * 0.3},
        ], weight_decay=train_cfg.weight_decay)
        sched_b = CosineAnnealingWarmRestarts(opt_b, T_0=5, T_mult=2)

        for ep in range(1, args.epochs_phase_b + 1):
            t0 = time.time()
            loss = train_one_epoch(model, train_dl, criterion, opt_b, device, ep)
            sched_b.step()
            print(f"Epoch {ep}/{args.epochs_phase_b} | Loss {loss:.4f} | {time.time()-t0:.1f}s")
            if ep % 5 == 0:
                r = validate(model, val_dl, device)
                print(f"  R@1={r[1]:.4f}  R@5={r[5]:.4f}  R@10={r[10]:.4f}")
            if loss < best_loss:
                best_loss = loss
                model.save_full(args.output.replace(".pth", "_full.pth"))
                model.save_heads(args.output)

    model.save_heads(args.output)
    print(f"\nDone — saved to {args.output}")

    json.dump(
        {"backbone": args.backbone, "dim": args.output_dim,
         "best_loss": best_loss, "device": str(device)},
        open(paths.training_log, "w"), indent=2,
    )


if __name__ == "__main__":
    main()
