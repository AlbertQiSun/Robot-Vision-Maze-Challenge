"""
Maze exploration dataset + PK sampler for metric learning.
"""

from __future__ import annotations

import json
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from vis_nav.config import train_cfg, PURE_ACTIONS
from vis_nav.data.transforms import get_inference_transform


class MazeExplorationDataset(Dataset):
    """
    Dataset from one or more maze exploration trajectories.

    Each sample: ``(image_tensor, place_label)``
    Place labels encode  ``maze_id * 100_000 + position_group``.
    """

    def __init__(
        self,
        data_dirs: list[str],
        transform=None,
        subsample_rate: int = train_cfg.subsample_rate,
        texture_dir: str | None = None,
        synthetic_ratio: float = 0.0,
    ):
        self.transform = transform
        self.synthetic_ratio = synthetic_ratio

        # Load textures for synthetic corridor generation
        self.textures: list[np.ndarray] = []
        if texture_dir and os.path.isdir(texture_dir):
            for f in sorted(os.listdir(texture_dir)):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    img = cv2.imread(os.path.join(texture_dir, f))
                    if img is not None:
                        self.textures.append(img)

        # Collect samples from every maze directory
        self.samples: list[tuple[str, int, int]] = []
        self.labels: np.ndarray
        self.maze_ids: np.ndarray
        self.n_mazes = 0

        _labels: list[int] = []
        _mids: list[int] = []

        for maze_id, data_dir in enumerate(data_dirs):
            info_path = os.path.join(data_dir, "data_info.json")
            img_dir = os.path.join(data_dir, "images")
            if not os.path.exists(info_path):
                continue

            with open(info_path) as f:
                raw = json.load(f)

            motion = [
                d for d in raw
                if len(d["action"]) == 1 and d["action"][0] in PURE_ACTIONS
            ][::subsample_rate]

            for idx, frame in enumerate(motion):
                img_path = os.path.join(img_dir, frame["image"])
                if not os.path.exists(img_path):
                    continue
                label = maze_id * 100_000 + idx // train_cfg.positive_range
                self.samples.append((img_path, label, maze_id))
                _labels.append(label)
                _mids.append(maze_id)

        self.labels = np.array(_labels)
        self.maze_ids = np.array(_mids)
        self.n_mazes = len(data_dirs)

    # ── length includes synthetic images ─────────────────────────────
    def __len__(self) -> int:
        n_real = len(self.samples)
        return n_real + int(n_real * self.synthetic_ratio)

    def __getitem__(self, idx: int):
        if idx < len(self.samples):
            img_path, label, _ = self.samples[idx]
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((240, 320, 3), dtype=np.uint8)
        else:
            img = self._generate_synthetic()
            label = -1

        rgb = img[:, :, ::-1].copy()
        tensor = (self.transform or get_inference_transform())(rgb)
        return tensor, label

    # ── synthetic corridor view from random textures ─────────────────
    def _generate_synthetic(self) -> np.ndarray:
        h, w = 240, 320
        if len(self.textures) < 2:
            return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        left_tex = random.choice(self.textures)
        right_tex = random.choice(self.textures)
        floor = np.random.randint(50, 200, 3).astype(np.uint8)

        img = np.zeros((h, w, 3), dtype=np.uint8)
        pts_src = np.float32([[0, 0], [w // 3, 0], [w // 3, h], [0, h]])

        left_crop = cv2.resize(left_tex, (w // 3, h))
        pts_l = np.float32([[0, h // 5], [w // 4, 0], [w // 4, h], [0, h * 4 // 5]])
        left_wall = cv2.warpPerspective(
            left_crop, cv2.getPerspectiveTransform(pts_src, pts_l), (w, h)
        )

        right_crop = cv2.resize(right_tex, (w // 3, h))
        pts_r = np.float32(
            [[w, h // 5], [w * 3 // 4, 0], [w * 3 // 4, h], [w, h * 4 // 5]]
        )
        right_wall = cv2.warpPerspective(
            right_crop, cv2.getPerspectiveTransform(pts_src, pts_r), (w, h)
        )

        img[h * 3 // 5 :] = floor
        img[: h // 5] = (floor * 0.8).astype(np.uint8)
        img[left_wall.sum(2) > 0] = left_wall[left_wall.sum(2) > 0]
        img[right_wall.sum(2) > 0] = right_wall[right_wall.sum(2) > 0]
        return img


# ── PK Sampler ──────────────────────────────────────────────────────────
class PKSampler(Sampler):
    """Sample P classes × K instances per batch for metric learning."""

    def __init__(self, labels: np.ndarray, P: int = 16, K: int = 4):
        self.P, self.K = P, K
        self.label_to_idx: dict[int, list[int]] = {}
        for i, lbl in enumerate(labels):
            if lbl < 0:
                continue
            self.label_to_idx.setdefault(int(lbl), []).append(i)
        self.valid = [l for l, v in self.label_to_idx.items() if len(v) >= K]
        if len(self.valid) < P:
            self.valid = list(self.label_to_idx.keys())

    def __iter__(self):
        random.shuffle(self.valid)
        batch: list[int] = []
        for lbl in self.valid:
            idxs = self.label_to_idx[lbl]
            chosen = random.sample(idxs, self.K) if len(idxs) >= self.K \
                else random.choices(idxs, k=self.K)
            batch.extend(chosen)
            if len(batch) >= self.P * self.K:
                yield from batch[: self.P * self.K]
                batch = batch[self.P * self.K :]
        if batch:
            yield from batch

    def __len__(self) -> int:
        return (len(self.valid) // self.P) * self.P * self.K
