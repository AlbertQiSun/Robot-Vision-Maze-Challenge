"""
Training Script: Action Predictor MLP.

Trains a small network that predicts navigation actions from
(current_feature, goal_feature, action_history) tuples.

Used as a fallback controller when graph-based planning fails.

Usage:
  python source/train_action_predictor.py \
    --data-dir training_data/ \
    --projection-model models/projection_head.pth \
    --output models/action_predictor.pth \
    --device cuda

  # Single maze:
  python source/train_action_predictor.py \
    --data-dir data/ \
    --single-maze \
    --projection-model models/projection_head.pth \
    --output models/action_predictor.pth
"""

import argparse
import json
import os
import sys
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import (
    PlaceFeatureExtractor, ActionPredictor,
    get_inference_transform, get_device,
    extract_features_batch, PROJECTION_DIM,
)

# Action encoding
ACTION_TO_IDX = {'FORWARD': 0, 'LEFT': 1, 'RIGHT': 2, 'BACKWARD': 3}
IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}
PURE_ACTIONS = set(ACTION_TO_IDX.keys())


# ---------------------------------------------------------------------------
# Dataset: Action Prediction Pairs
# ---------------------------------------------------------------------------
class ActionPairDataset(Dataset):
    """
    Dataset of (current_feature, goal_feature, action_history, action_label) tuples.

    For each frame i in a trajectory:
      - Sample several future frames j (50-200 steps ahead) as "goals"
      - Current feature = feature_i
      - Goal feature = feature_j
      - Label = action at frame i (what action moves toward the goal)
      - Also generate reverse: (feature_j, feature_i, reverse_action)
    """

    def __init__(self, features_by_maze: dict,
                 frames_by_maze: dict,
                 n_goals_per_frame: int = 10,
                 min_goal_gap: int = 50,
                 max_goal_gap: int = 200):
        """
        Args:
            features_by_maze: dict[maze_id] → (N, D) feature array
            frames_by_maze: dict[maze_id] → list of frame dicts with 'action'
            n_goals_per_frame: number of goal samples per anchor frame
            min_goal_gap: minimum step gap for goal sampling
            max_goal_gap: maximum step gap for goal sampling
        """
        self.pairs = []     # (current_feat, goal_feat, action_history, action_label)

        reverse_map = {'FORWARD': 'BACKWARD', 'BACKWARD': 'FORWARD',
                       'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}

        for maze_id, features in features_by_maze.items():
            frames = frames_by_maze[maze_id]
            n = len(features)

            for i in range(n):
                action_i = frames[i]['action']
                if action_i not in ACTION_TO_IDX:
                    continue

                # Sample goal frames
                lo = min(i + min_goal_gap, n)
                hi = min(i + max_goal_gap, n)
                if lo >= hi:
                    continue

                n_samples = min(n_goals_per_frame, hi - lo)
                goals = random.sample(range(lo, hi), n_samples)

                for j in goals:
                    # Forward pair: (i, j) → action at i
                    action_hist = self._get_action_history(frames, i)
                    self.pairs.append((
                        features[i],
                        features[j],
                        action_hist,
                        ACTION_TO_IDX[action_i],
                    ))

                    # Reverse pair: (j, i) → reverse action at j-1
                    if j > 0 and frames[j - 1]['action'] in reverse_map:
                        rev_action = reverse_map[frames[j - 1]['action']]
                        rev_hist = self._get_action_history(frames, j)
                        self.pairs.append((
                            features[j],
                            features[i],
                            rev_hist,
                            ACTION_TO_IDX[rev_action],
                        ))

        random.shuffle(self.pairs)
        print(f"  Action pairs: {len(self.pairs)} total")

    def _get_action_history(self, frames, idx, history_len=3):
        """Get one-hot encoded action history before idx."""
        hist = np.zeros(4 * history_len, dtype=np.float32)
        for k in range(history_len):
            h_idx = idx - history_len + k
            if 0 <= h_idx < len(frames):
                action = frames[h_idx]['action']
                if action in ACTION_TO_IDX:
                    hist[k * 4 + ACTION_TO_IDX[action]] = 1.0
        return hist

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        current_feat, goal_feat, action_hist, label = self.pairs[idx]
        return (
            torch.tensor(current_feat, dtype=torch.float32),
            torch.tensor(goal_feat, dtype=torch.float32),
            torch.tensor(action_hist, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_action_predictor(model, train_loader, val_loader,
                            epochs, device, lr=1e-3):
    """Train the action predictor."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for current_feat, goal_feat, action_hist, labels in train_loader:
            current_feat = current_feat.to(device)
            goal_feat = goal_feat.to(device)
            action_hist = action_hist.to(device)
            labels = labels.to(device)

            logits = model(current_feat, goal_feat, action_hist)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for current_feat, goal_feat, action_hist, labels in val_loader:
                current_feat = current_feat.to(device)
                goal_feat = goal_feat.to(device)
                action_hist = action_hist.to(device)
                labels = labels.to(device)

                logits = model(current_feat, goal_feat, action_hist)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / max(val_total, 1)

        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

    if best_state:
        model.load_state_dict(best_state)

    return best_val_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train action predictor")
    parser.add_argument("--data-dir", type=str, default="training_data/")
    parser.add_argument("--single-maze", action="store_true")
    parser.add_argument("--projection-model", type=str,
                        default="models/projection_head.pth")
    parser.add_argument("--output", type=str,
                        default="models/action_predictor.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-goals", type=int, default=10,
                        help="Goal samples per frame")
    parser.add_argument("--subsample", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Discover mazes
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
            if os.path.exists(os.path.join(args.data_dir, "data_info.json")):
                data_dirs = [args.data_dir]
            else:
                print(f"ERROR: No maze data found in {args.data_dir}")
                return

    print(f"Found {len(data_dirs)} maze(s)")

    # Load feature extractor
    print("Loading feature extractor...")
    feat_model = PlaceFeatureExtractor(freeze_backbone=True)
    if os.path.exists(args.projection_model):
        feat_model.load_heads(args.projection_model, device)
    feat_model.to(device)
    feat_model.eval()

    # Extract features for all mazes
    features_by_maze = {}
    frames_by_maze = {}

    for maze_id, data_dir in enumerate(data_dirs):
        print(f"\nProcessing maze {maze_id}: {data_dir}")
        info_path = os.path.join(data_dir, "data_info.json")
        img_dir = os.path.join(data_dir, "images")

        with open(info_path) as f:
            raw = json.load(f)

        # Filter pure motion frames
        motion = [
            {'step': d['step'], 'image': d['image'], 'action': d['action'][0]}
            for d in raw
            if len(d['action']) == 1 and d['action'][0] in PURE_ACTIONS
        ]
        motion = motion[::args.subsample]

        # Load images
        images = []
        valid_frames = []
        for frame in motion:
            img_path = os.path.join(img_dir, frame['image'])
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                valid_frames.append(frame)

        if not images:
            continue

        # Extract features
        feats = extract_features_batch(feat_model, images, batch_size=64,
                                        device=device)
        features_by_maze[maze_id] = feats
        frames_by_maze[maze_id] = valid_frames
        print(f"  {len(images)} frames → {feats.shape}")

    # Build dataset
    print("\nBuilding action pair dataset...")
    dataset = ActionPairDataset(
        features_by_maze=features_by_maze,
        frames_by_maze=frames_by_maze,
        n_goals_per_frame=args.n_goals,
    )

    # Train/val split (90/10)
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers,
                               pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    # Train
    print(f"\nTraining action predictor ({len(train_dataset)} train, "
          f"{len(val_dataset)} val)...")
    action_model = ActionPredictor(descriptor_dim=PROJECTION_DIM)
    best_acc = train_action_predictor(
        action_model, train_loader, val_loader,
        epochs=args.epochs, device=device, lr=args.lr,
    )

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(action_model.state_dict(), args.output)
    print(f"\nSaved action predictor to {args.output}")
    print(f"Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
