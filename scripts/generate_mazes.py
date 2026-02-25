#!/usr/bin/env python3
"""
Generate random maze layouts for multi-maze training.

Usage
-----
  python scripts/generate_mazes.py --n-mazes 30 --output-dir training_data/
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np


def generate_maze(size: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    random.seed(seed)
    maze = np.ones((size, size), dtype=int)
    maze[1, 1] = 0
    stack = [(1, 1)]
    while stack:
        r, c = stack.pop()
        nbrs = [
            (r + dr, c + dc)
            for dr, dc in [(0, -2), (0, 2), (-2, 0), (2, 0)]
            if 0 <= r + dr < size and 0 <= c + dc < size and maze[r + dr, c + dc] == 1
        ]
        if nbrs:
            nr, nc = nbrs[np.random.randint(len(nbrs))]
            maze[r + (nr - r) // 2, c + (nc - c) // 2] = 0
            maze[nr, nc] = 0
            stack += [(r, c), (nr, nc)]
    for i in range(2, size - 2):
        for j in range(2, size - 2):
            if maze[i, j] == 0 and np.random.rand() < 0.1:
                rh = np.random.randint(1, 4)
                rw = np.random.randint(1, 4)
                if i + rh < size and j + rw < size:
                    maze[i : i + rh, j : j + rw] = 0
    return maze


def simulate_exploration(maze: np.ndarray, seed: int) -> list[dict]:
    random.seed(seed)
    size = maze.shape[0]
    dr, dc = [0, 1, 0, -1], [1, 0, -1, 0]
    pos, facing, step = (1, 1), 0, 0
    visited: set = {pos}
    traj: list[dict] = []
    for _ in range(20_000):
        moved = False
        for name, delta in [("RIGHT", 1), ("FORWARD", 0), ("LEFT", -1), ("BACKWARD", 2)]:
            nf = (facing + delta) % 4
            nr, nc = pos[0] + dr[nf], pos[1] + dc[nf]
            if 0 <= nr < size and 0 <= nc < size and maze[nr, nc] == 0:
                if delta == 1:
                    traj.append({"step": step, "action": "RIGHT", "position": pos}); step += 1
                elif delta == -1:
                    traj.append({"step": step, "action": "LEFT", "position": pos}); step += 1
                elif delta == 2:
                    for _ in range(2):
                        traj.append({"step": step, "action": "LEFT", "position": pos}); step += 1
                traj.append({"step": step, "action": "FORWARD", "position": pos}); step += 1
                pos, facing = (nr, nc), nf
                visited.add(pos)
                moved = True
                break
        if not moved:
            traj.append({"step": step, "action": "RIGHT", "position": pos}); step += 1
            facing = (facing + 1) % 4
        if len(visited) >= (maze == 0).sum() * 0.95:
            break
    return traj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mazes", type=int, default=30)
    ap.add_argument("--size", type=int, default=31)
    ap.add_argument("--output-dir", default="training_data/")
    ap.add_argument("--base-seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(args.n_mazes):
        seed = args.base_seed + i
        mdir = os.path.join(args.output_dir, f"maze_{i:03d}")
        os.makedirs(os.path.join(mdir, "images"), exist_ok=True)

        maze = generate_maze(args.size, seed)
        traj = simulate_exploration(maze, seed)
        info = [{"step": e["step"], "image": f"{e['step']}.jpg", "action": [e["action"]]} for e in traj]
        with open(os.path.join(mdir, "data_info.json"), "w") as f:
            json.dump(info, f)
        np.save(os.path.join(mdir, "maze_layout.npy"), maze)
        print(f"  maze_{i:03d}: {(maze==0).sum()} open, {len(traj)} steps")

    print(f"\nGenerated {args.n_mazes} mazes → {args.output_dir}")


if __name__ == "__main__":
    main()
