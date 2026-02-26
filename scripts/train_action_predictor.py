#!/usr/bin/env python3
"""
Train the attention-based action predictor.

Usage
-----
  python scripts/train_action_predictor.py \
      --data-dir data/ --single-maze --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, random_split

from vis_nav.config import (
    paths, feat_cfg, train_cfg,
    ACTION_TO_IDX, PURE_ACTIONS, REVERSE_ACTION,
)
from vis_nav.utils import get_device, ensure_dirs
from vis_nav.models import PlaceFeatureExtractor, ActionPredictor
from vis_nav.data import extract_features_batch


# ── Dataset ─────────────────────────────────────────────────────────────
class ActionPairDataset(Dataset):
    """
    Create (current_feat, goal_feat, action_history, label) pairs.

    Uses MULTIPLE goal ranges to teach the model both short-range
    and long-range navigation.
    """

    def __init__(self, features_by_maze, frames_by_maze,
                 n_goals=train_cfg.action_n_goals):
        self.pairs: list[tuple] = []

        # Short-range only: long-range labels are noise because the local
        # action at frame i has little to do with direction to a goal 50+ away.
        ranges = [(1, 8), (8, 20)]

        for mid, feats in features_by_maze.items():
            frames = frames_by_maze[mid]
            n = len(feats)
            for i in range(n):
                act = frames[i]["action"]
                if act not in ACTION_TO_IDX:
                    continue

                for min_gap, max_gap in ranges:
                    lo = min(i + min_gap, n)
                    hi = min(i + max_gap, n)
                    if lo >= hi:
                        continue
                    k = min(n_goals, hi - lo)
                    for j in random.sample(range(lo, hi), k):
                        hist = self._hist(frames, i)
                        self.pairs.append(
                            (feats[i], feats[j], hist, ACTION_TO_IDX[act])
                        )
                        # Reverse pair
                        if j > 0 and frames[j - 1]["action"] in REVERSE_ACTION:
                            rev = REVERSE_ACTION[frames[j - 1]["action"]]
                            self.pairs.append(
                                (feats[j], feats[i], self._hist(frames, j),
                                 ACTION_TO_IDX[rev])
                            )
        random.shuffle(self.pairs)
        print(f"  Dataset: {len(self.pairs)} pairs")

    @staticmethod
    def _hist(frames, idx, hlen=3):
        h = np.zeros(4 * hlen, dtype=np.float32)
        for k in range(hlen):
            hi = idx - hlen + k
            if 0 <= hi < len(frames) and frames[hi]["action"] in ACTION_TO_IDX:
                h[k * 4 + ACTION_TO_IDX[frames[hi]["action"]]] = 1.0
        return h

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        cf, gf, ah, lbl = self.pairs[i]
        return (torch.tensor(cf, dtype=torch.float32),
                torch.tensor(gf, dtype=torch.float32),
                torch.tensor(ah, dtype=torch.float32),
                torch.tensor(lbl, dtype=torch.long))


# ── Main ────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="training_data/")
    ap.add_argument("--single-maze", action="store_true")
    ap.add_argument("--projection-model", default=paths.projection_head)
    ap.add_argument("--output", default=paths.action_predictor)
    ap.add_argument("--epochs", type=int, default=train_cfg.action_epochs)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=train_cfg.action_lr)
    ap.add_argument("--subsample", type=int, default=train_cfg.subsample_rate)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = get_device() if args.device == "auto" else torch.device(args.device)
    ensure_dirs()

    # Discover mazes
    if args.single_maze:
        data_dirs = [args.data_dir]
    else:
        data_dirs = sorted(
            os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d))
            and os.path.exists(os.path.join(args.data_dir, d, "data_info.json"))
        )
        if not data_dirs:
            if os.path.exists(os.path.join(args.data_dir, "data_info.json")):
                data_dirs = [args.data_dir]

    # Feature extractor
    feat_model = PlaceFeatureExtractor(freeze_backbone=True)
    if os.path.exists(args.projection_model):
        feat_model.load_heads(args.projection_model, device)
    feat_model.to(device).eval()

    # Extract features per maze
    features_by_maze, frames_by_maze = {}, {}
    for mid, ddir in enumerate(data_dirs):
        with open(os.path.join(ddir, "data_info.json")) as f:
            raw = json.load(f)
        motion = [
            {"step": d["step"], "image": d["image"], "action": d["action"][0]}
            for d in raw
            if len(d["action"]) == 1 and d["action"][0] in PURE_ACTIONS
        ][:: args.subsample]

        imgs, valid = [], []
        for fr in motion:
            img = cv2.imread(os.path.join(ddir, "images", fr["image"]))
            if img is not None:
                imgs.append(img)
                valid.append(fr)
        if not imgs:
            continue
        features_by_maze[mid] = extract_features_batch(feat_model, imgs, 64, device)
        frames_by_maze[mid] = valid
        print(f"  Maze {mid}: {len(valid)} frames, {len(features_by_maze[mid])} features")

    # Dataset
    ds = ActionPairDataset(features_by_maze, frames_by_maze)
    n_val = int(len(ds) * 0.1)
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # Model
    model = ActionPredictor().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.05,  # helps generalization
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    best_acc, best_state = 0.0, None

    for ep in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        tot, cor, cnt = 0.0, 0, 0
        for cf, gf, ah, lbl in train_dl:
            cf = cf.to(device); gf = gf.to(device)
            ah = ah.to(device); lbl = lbl.to(device)

            logits = model(cf, gf, ah)
            loss = criterion(logits, lbl)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tot += loss.item() * len(lbl)
            cor += (logits.argmax(1) == lbl).sum().item()
            cnt += len(lbl)

        scheduler.step()

        # ── Validate ──
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for cf, gf, ah, lbl in val_dl:
                cf = cf.to(device); gf = gf.to(device)
                ah = ah.to(device); lbl = lbl.to(device)
                vc += (model(cf, gf, ah).argmax(1) == lbl).sum().item()
                vt += len(lbl)
        val_acc = vc / max(vt, 1)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Ep {ep}/{args.epochs} | loss {tot/cnt:.4f} | "
              f"train {cor/cnt:.4f} | val {val_acc:.4f} | lr {lr_now:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"\nSaved → {args.output}  (best val acc {best_acc:.4f})")


if __name__ == "__main__":
    main()
