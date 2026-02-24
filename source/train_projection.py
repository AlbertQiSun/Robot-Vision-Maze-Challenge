"""
Training Script: DINOv2 + GeM + Projection MLP for Place Recognition.

Two-phase training:
  Phase A: Frozen backbone, train GeM + Projection (30 epochs)
  Phase B: Unfreeze last 2 ViT blocks + heads (20 epochs)

Loss: Multi-Similarity Loss with cross-maze hard negative mining.

Usage:
  python source/train_projection.py \
    --data-dir training_data/ \
    --output models/projection_head.pth \
    --device cuda \
    --epochs-phase-a 30 \
    --epochs-phase-b 20 \
    --batch-size 64

  # Single-maze training (practice maze only):
  python source/train_projection.py \
    --data-dir data/ \
    --single-maze \
    --output models/projection_head.pth \
    --device cuda
"""

import argparse
import json
import os
import sys
import random
import hashlib
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Add source to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import (
    PlaceFeatureExtractor, get_train_transform, get_inference_transform,
    get_device, PROJECTION_DIM,
)

# ---------------------------------------------------------------------------
# Multi-Similarity Loss
# ---------------------------------------------------------------------------
class MultiSimilarityLoss(nn.Module):
    """
    Multi-Similarity Loss (Wang et al., CVPR 2019).

    Considers all positive and negative pairs simultaneously,
    automatically weighting harder pairs more.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 50.0,
                 base: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base

    def forward(self, embeddings: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) L2-normalized embeddings
            labels: (B,) integer place-group labels
        """
        sim_mat = embeddings @ embeddings.T  # (B, B)
        batch_size = len(embeddings)

        loss = torch.tensor(0.0, device=embeddings.device)
        valid_count = 0

        for i in range(batch_size):
            pos_mask = (labels == labels[i])
            pos_mask[i] = False  # exclude self
            neg_mask = (labels != labels[i])

            pos_sim = sim_mat[i][pos_mask]
            neg_sim = sim_mat[i][neg_mask]

            if len(pos_sim) == 0 or len(neg_sim) == 0:
                continue

            # Hard mining: filter informative pairs
            # Keep negatives harder than easiest positive
            neg_sim = neg_sim[neg_sim > pos_sim.min() - 0.1]
            # Keep positives harder than hardest negative
            pos_sim = pos_sim[pos_sim < neg_sim.max() + 0.1] if len(neg_sim) > 0 else pos_sim

            if len(pos_sim) == 0 or len(neg_sim) == 0:
                continue

            pos_loss = (1.0 / self.alpha) * torch.log(
                1 + torch.sum(torch.exp(-self.alpha * (pos_sim - self.base)))
            )
            neg_loss = (1.0 / self.beta) * torch.log(
                1 + torch.sum(torch.exp(self.beta * (neg_sim - self.base)))
            )
            loss = loss + pos_loss + neg_loss
            valid_count += 1

        if valid_count > 0:
            loss = loss / valid_count
        return loss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MazeExplorationDataset(Dataset):
    """
    Dataset from one or more maze exploration trajectories.

    Each sample: (image_tensor, place_label)
    Place labels encode: maze_id * 100000 + position_group
    """

    POSITIVE_RANGE = 5      # frames within ±5 steps = same place
    PURE_ACTIONS = {'FORWARD', 'LEFT', 'RIGHT', 'BACKWARD'}

    def __init__(self, data_dirs: list[str], transform=None,
                 subsample_rate: int = 2,
                 texture_dir: str = None,
                 synthetic_ratio: float = 0.0):
        """
        Args:
            data_dirs: list of maze data directories (each with data_info.json + images/)
            transform: image transform
            subsample_rate: take every Nth motion frame
            texture_dir: path to texture images for synthetic generation
            synthetic_ratio: fraction of synthetic images per epoch
        """
        self.transform = transform
        self.subsample_rate = subsample_rate
        self.synthetic_ratio = synthetic_ratio

        # Load textures for synthetic generation
        self.textures = []
        if texture_dir and os.path.isdir(texture_dir):
            for f in sorted(os.listdir(texture_dir)):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = cv2.imread(os.path.join(texture_dir, f))
                    if img is not None:
                        self.textures.append(img)
            print(f"  Loaded {len(self.textures)} textures for synthesis")

        # Load exploration data from all mazes
        self.samples = []       # (image_path, place_label, maze_id)
        self.maze_ids = []
        self.labels = []

        for maze_id, data_dir in enumerate(data_dirs):
            info_path = os.path.join(data_dir, "data_info.json")
            img_dir = os.path.join(data_dir, "images")

            if not os.path.exists(info_path):
                print(f"  [WARN] Skipping {data_dir}: no data_info.json")
                continue

            with open(info_path) as f:
                raw = json.load(f)

            # Filter to pure single-action motion frames
            motion = [
                d for d in raw
                if len(d['action']) == 1 and d['action'][0] in self.PURE_ACTIONS
            ]

            # Subsample
            motion = motion[::subsample_rate]

            for idx, frame in enumerate(motion):
                img_path = os.path.join(img_dir, frame['image'])
                if not os.path.exists(img_path):
                    continue

                # Place label: encodes maze + position group
                position_group = idx // self.POSITIVE_RANGE
                label = maze_id * 100000 + position_group

                self.samples.append((img_path, label, maze_id))
                self.labels.append(label)
                self.maze_ids.append(maze_id)

        self.labels = np.array(self.labels)
        self.maze_ids = np.array(self.maze_ids)
        self.n_mazes = len(data_dirs)

        print(f"  Dataset: {len(self.samples)} samples from {self.n_mazes} maze(s)")
        print(f"  Unique places: {len(np.unique(self.labels))}")

    def __len__(self):
        n_real = len(self.samples)
        n_synthetic = int(n_real * self.synthetic_ratio)
        return n_real + n_synthetic

    def __getitem__(self, idx):
        if idx < len(self.samples):
            # Real image
            img_path, label, maze_id = self.samples[idx]
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((240, 320, 3), dtype=np.uint8)
        else:
            # Synthetic image — always a unique negative
            img = self._generate_synthetic()
            label = -1  # unique negative (won't match anything)

        # BGR → RGB for transforms
        rgb = img[:, :, ::-1].copy()

        if self.transform:
            tensor = self.transform(rgb)
        else:
            tensor = get_inference_transform()(rgb)

        return tensor, label

    def _generate_synthetic(self) -> np.ndarray:
        """Generate a synthetic corridor view from random textures."""
        h, w = 240, 320
        img = np.zeros((h, w, 3), dtype=np.uint8)

        if len(self.textures) < 2:
            # Random noise if no textures
            return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        left_tex = random.choice(self.textures)
        right_tex = random.choice(self.textures)
        floor_color = np.random.randint(50, 200, 3).astype(np.uint8)

        # Left wall
        left_crop = cv2.resize(left_tex, (w // 3, h))
        pts_src = np.float32([[0, 0], [w // 3, 0], [w // 3, h], [0, h]])
        pts_dst = np.float32([[0, h // 5], [w // 4, 0], [w // 4, h], [0, h * 4 // 5]])
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        left_wall = cv2.warpPerspective(left_crop, M, (w, h))

        # Right wall
        right_crop = cv2.resize(right_tex, (w // 3, h))
        pts_dst_r = np.float32([[w, h // 5], [w * 3 // 4, 0],
                                 [w * 3 // 4, h], [w, h * 4 // 5]])
        M_r = cv2.getPerspectiveTransform(pts_src, pts_dst_r)
        right_wall = cv2.warpPerspective(right_crop, M_r, (w, h))

        # Compose
        img[h * 3 // 5:] = floor_color
        img[:h // 5] = (floor_color * 0.8).astype(np.uint8)
        mask_l = left_wall.sum(axis=2) > 0
        mask_r = right_wall.sum(axis=2) > 0
        img[mask_l] = left_wall[mask_l]
        img[mask_r] = right_wall[mask_r]

        return img


# ---------------------------------------------------------------------------
# PK Sampler (for metric learning)
# ---------------------------------------------------------------------------
class PKSampler(Sampler):
    """
    Sample P classes with K instances each.

    For multi-maze training: samples places from different mazes
    to create cross-maze negative pairs.
    """

    def __init__(self, labels: np.ndarray, P: int = 16, K: int = 4):
        self.labels = labels
        self.P = P
        self.K = K

        # Build label → indices map
        self.label_to_indices = {}
        for idx, label in enumerate(labels):
            if label < 0:
                continue  # skip synthetic
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

        # Only use labels with at least K instances
        self.valid_labels = [
            l for l, idxs in self.label_to_indices.items()
            if len(idxs) >= K
        ]
        if len(self.valid_labels) < P:
            # Relax: allow labels with fewer instances (sample with replacement)
            self.valid_labels = list(self.label_to_indices.keys())

    def __iter__(self):
        # Shuffle and yield batches of P*K
        random.shuffle(self.valid_labels)
        batch = []
        for label in self.valid_labels:
            indices = self.label_to_indices[label]
            if len(indices) >= self.K:
                chosen = random.sample(indices, self.K)
            else:
                chosen = random.choices(indices, k=self.K)
            batch.extend(chosen)

            if len(batch) >= self.P * self.K:
                yield from batch[:self.P * self.K]
                batch = batch[self.P * self.K:]

        if batch:
            yield from batch

    def __len__(self):
        return (len(self.valid_labels) // self.P) * self.P * self.K


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Skip batches with all same labels (no negatives)
        unique_labels = labels.unique()
        if len(unique_labels) < 2:
            continue

        # Forward
        embeddings = model(images)

        # Loss
        loss = criterion(embeddings, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx} | "
                  f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, device):
    """Compute recall@1 and recall@5 on validation set."""
    model.eval()
    all_features = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        features = model(images)
        all_features.append(features.cpu())
        all_labels.append(labels)

    features = torch.cat(all_features)
    labels = torch.cat(all_labels)

    # Compute similarity matrix
    sim = features @ features.T
    sim.fill_diagonal_(-2)

    # Recall@K
    recalls = {}
    for k in [1, 5, 10]:
        correct = 0
        total = 0
        _, top_k_indices = sim.topk(k, dim=1)
        for i in range(len(labels)):
            retrieved_labels = labels[top_k_indices[i]]
            if labels[i] in retrieved_labels:
                correct += 1
            total += 1
        recalls[k] = correct / max(total, 1)

    return recalls


# ---------------------------------------------------------------------------
# Main Training
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train DINOv2 projection head")
    parser.add_argument("--data-dir", type=str, default="training_data/",
                        help="Directory containing maze subdirectories")
    parser.add_argument("--single-maze", action="store_true",
                        help="Treat data-dir as a single maze (not multi-maze)")
    parser.add_argument("--texture-dir", type=str, default="data/textures/",
                        help="Directory with texture images for synthesis")
    parser.add_argument("--output", type=str, default="models/projection_head.pth",
                        help="Output path for trained model")
    parser.add_argument("--backbone", type=str, default="dinov2_vitb14",
                        help="DINOv2 backbone name")
    parser.add_argument("--output-dim", type=int, default=PROJECTION_DIM,
                        help="Projection output dimension")
    parser.add_argument("--epochs-phase-a", type=int, default=30,
                        help="Epochs for Phase A (frozen backbone)")
    parser.add_argument("--epochs-phase-b", type=int, default=20,
                        help="Epochs for Phase B (unfrozen last 2 blocks)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (P*K for PK sampling)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate for projection head")
    parser.add_argument("--lr-backbone", type=float, default=1e-5,
                        help="Learning rate for backbone (Phase B)")
    parser.add_argument("--subsample", type=int, default=2,
                        help="Subsample rate for motion frames")
    parser.add_argument("--synthetic-ratio", type=float, default=0.2,
                        help="Fraction of synthetic images per epoch")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cuda, mps, cpu, or auto")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of data for validation")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Discover maze directories
    if args.single_maze:
        data_dirs = [args.data_dir]
    else:
        data_dirs = sorted([
            os.path.join(args.data_dir, d)
            for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d))
            and os.path.exists(os.path.join(args.data_dir, d, "data_info.json"))
        ])
        if not data_dirs:
            # Fallback: check if data_dir itself is a maze
            if os.path.exists(os.path.join(args.data_dir, "data_info.json")):
                data_dirs = [args.data_dir]
            else:
                print(f"ERROR: No maze data found in {args.data_dir}")
                return

    print(f"Found {len(data_dirs)} maze(s):")
    for d in data_dirs[:5]:
        print(f"  {d}")
    if len(data_dirs) > 5:
        print(f"  ... and {len(data_dirs) - 5} more")

    # Dataset
    train_transform = get_train_transform()
    val_transform = get_inference_transform()

    full_dataset = MazeExplorationDataset(
        data_dirs=data_dirs,
        transform=train_transform,
        subsample_rate=args.subsample,
        texture_dir=args.texture_dir,
        synthetic_ratio=args.synthetic_ratio,
    )

    # Train/val split
    n_val = int(len(full_dataset.samples) * args.val_split)
    n_train = len(full_dataset.samples) - n_val

    # Simple split: last N% for validation
    val_dataset = MazeExplorationDataset(
        data_dirs=data_dirs,
        transform=val_transform,
        subsample_rate=args.subsample,
        synthetic_ratio=0.0,
    )

    # PK Sampler for training
    P = 16
    K = max(2, args.batch_size // P)
    sampler = PKSampler(full_dataset.labels, P=P, K=K)

    train_loader = DataLoader(
        full_dataset, batch_size=args.batch_size,
        sampler=sampler, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model
    print(f"\nInitializing {args.backbone}...")
    model = PlaceFeatureExtractor(
        backbone_name=args.backbone,
        projection_dim=args.output_dim,
        freeze_backbone=True,
    ).to(device)

    criterion = MultiSimilarityLoss(alpha=2.0, beta=50.0, base=0.5)

    # ===========================================================
    # Phase A: Frozen backbone
    # ===========================================================
    print("\n" + "=" * 60)
    print("PHASE A: Training projection head (backbone frozen)")
    print("=" * 60)

    # Only optimize GeM + Projection
    trainable_params = list(model.gem.parameters()) + \
                       list(model.projection.parameters())
    optimizer_a = AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler_a = CosineAnnealingWarmRestarts(optimizer_a, T_0=10, T_mult=2)

    best_loss = float('inf')
    for epoch in range(1, args.epochs_phase_a + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, criterion,
                               optimizer_a, device, epoch)
        scheduler_a.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs_phase_a} | "
              f"Loss: {loss:.4f} | "
              f"LR: {optimizer_a.param_groups[0]['lr']:.6f} | "
              f"Time: {elapsed:.1f}s")

        # Validate every 5 epochs
        if epoch % 5 == 0:
            recalls = validate(model, val_loader, device)
            print(f"  Val R@1={recalls[1]:.4f}  R@5={recalls[5]:.4f}  "
                  f"R@10={recalls[10]:.4f}")

        # Save best
        if loss < best_loss:
            best_loss = loss
            model.save_heads(args.output)
            print(f"  Saved best model (loss={best_loss:.4f})")

    # ===========================================================
    # Phase B: Unfreeze last 2 blocks
    # ===========================================================
    if args.epochs_phase_b > 0:
        print("\n" + "=" * 60)
        print("PHASE B: Fine-tuning backbone (last 2 blocks)")
        print("=" * 60)

        model.unfreeze_last_n_blocks(n=2)

        # Separate learning rates
        backbone_params = [p for p in model.backbone.parameters()
                          if p.requires_grad]
        head_params = list(model.gem.parameters()) + \
                      list(model.projection.parameters())

        optimizer_b = AdamW([
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr * 0.3},
        ], weight_decay=1e-4)
        scheduler_b = CosineAnnealingWarmRestarts(optimizer_b, T_0=5, T_mult=2)

        for epoch in range(1, args.epochs_phase_b + 1):
            t0 = time.time()
            loss = train_one_epoch(model, train_loader, criterion,
                                   optimizer_b, device, epoch)
            scheduler_b.step()
            elapsed = time.time() - t0

            print(f"Epoch {epoch}/{args.epochs_phase_b} | "
                  f"Loss: {loss:.4f} | "
                  f"Time: {elapsed:.1f}s")

            if epoch % 5 == 0:
                recalls = validate(model, val_loader, device)
                print(f"  Val R@1={recalls[1]:.4f}  R@5={recalls[5]:.4f}  "
                      f"R@10={recalls[10]:.4f}")

            if loss < best_loss:
                best_loss = loss
                # Save full model (including fine-tuned backbone)
                output_full = args.output.replace('.pth', '_full.pth')
                model.save_full(output_full)
                model.save_heads(args.output)
                print(f"  Saved best full model (loss={best_loss:.4f})")

    # Final save
    model.save_heads(args.output)
    print(f"\nTraining complete. Model saved to {args.output}")

    # Save training log
    log_dir = os.path.dirname(args.output)
    log_path = os.path.join(log_dir, "training_log.json")
    log = {
        "backbone": args.backbone,
        "output_dim": args.output_dim,
        "epochs_a": args.epochs_phase_a,
        "epochs_b": args.epochs_phase_b,
        "n_mazes": len(data_dirs),
        "n_samples": len(full_dataset.samples),
        "best_loss": float(best_loss),
        "device": str(device),
    }
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"Training log saved to {log_path}")


if __name__ == "__main__":
    main()
