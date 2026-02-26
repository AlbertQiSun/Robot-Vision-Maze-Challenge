#!/usr/bin/env python3
"""
Generate random maze layouts with rendered first-person-view images for
multi-maze training.

Usage
-----
  python scripts/generate_mazes.py --n-mazes 30 --output-dir training_data/

  # Use real game textures (much more realistic):
  python scripts/generate_mazes.py --n-mazes 30 --output-dir training_data/ \
      --texture-dir data/textures/

  # Quick test with 1 maze:
  python scripts/generate_mazes.py --n-mazes 1 --output-dir training_data/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random

import cv2
import numpy as np


# ── Maze generation ──────────────────────────────────────────────────────
def generate_maze(size: int, seed: int) -> np.ndarray:
    """Generate a random maze using randomised DFS with optional room carving."""
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
    # Carve some rooms
    for i in range(2, size - 2):
        for j in range(2, size - 2):
            if maze[i, j] == 0 and np.random.rand() < 0.1:
                rh = np.random.randint(1, 4)
                rw = np.random.randint(1, 4)
                if i + rh < size and j + rw < size:
                    maze[i : i + rh, j : j + rw] = 0
    return maze


# ── Exploration simulation ───────────────────────────────────────────────
def simulate_exploration(maze: np.ndarray, seed: int) -> list[dict]:
    """Simulate a robot exploring the maze, returning trajectory with positions
    and facing directions."""
    random.seed(seed)
    size = maze.shape[0]
    # facing: 0=East(+col), 1=South(+row), 2=West(-col), 3=North(-row)
    dr = [0, 1, 0, -1]
    dc = [1, 0, -1, 0]
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
                    traj.append({"step": step, "action": "RIGHT",
                                 "position": pos, "facing": facing})
                    step += 1
                    facing = nf
                elif delta == -1:
                    traj.append({"step": step, "action": "LEFT",
                                 "position": pos, "facing": facing})
                    step += 1
                    facing = nf
                elif delta == 2:
                    for _ in range(2):
                        traj.append({"step": step, "action": "LEFT",
                                     "position": pos, "facing": facing})
                        step += 1
                        facing = (facing - 1) % 4
                traj.append({"step": step, "action": "FORWARD",
                             "position": pos, "facing": facing})
                step += 1
                pos = (nr, nc)
                facing = nf
                visited.add(pos)
                moved = True
                break
        if not moved:
            traj.append({"step": step, "action": "RIGHT",
                         "position": pos, "facing": facing})
            step += 1
            facing = (facing + 1) % 4
        if len(visited) >= (maze == 0).sum() * 0.95:
            break
    return traj


# ── Raycasting Renderer ─────────────────────────────────────────────────
class MazeRenderer:
    """Simple raycasting renderer producing 320×240 first-person-view images
    from a 2D grid maze — similar to classic Wolfenstein 3D.

    Each wall cell is assigned one of the provided textures (or procedural
    colours). The floor and ceiling are flat-shaded with distance fog.
    """

    def __init__(
        self,
        maze: np.ndarray,
        textures: list[np.ndarray] | None = None,
        img_w: int = 320,
        img_h: int = 240,
        fov_deg: float = 60.0,
        seed: int = 0,
    ):
        self.maze = maze
        self.h, self.w = maze.shape
        self.img_w = img_w
        self.img_h = img_h
        self.fov = math.radians(fov_deg)
        self.max_depth = float(max(self.h, self.w))

        # Prepare textures: resize all to 64×64 for fast sampling
        self.tex_size = 64
        rng = np.random.RandomState(seed)

        if textures and len(textures) > 0:
            self.wall_textures = [
                cv2.resize(t, (self.tex_size, self.tex_size))
                for t in textures
            ]
        else:
            # Generate procedural colour textures
            self.wall_textures = []
            for _ in range(20):
                base = rng.randint(40, 220, 3).astype(np.uint8)
                tex = np.full((self.tex_size, self.tex_size, 3), base, dtype=np.uint8)
                # Add some noise/pattern
                noise = rng.randint(-30, 30, (self.tex_size, self.tex_size, 3))
                tex = np.clip(tex.astype(int) + noise, 0, 255).astype(np.uint8)
                # Horizontal brick lines
                for y in range(0, self.tex_size, self.tex_size // 4):
                    tex[y, :] = np.clip(base.astype(int) - 40, 0, 255).astype(np.uint8)
                self.wall_textures.append(tex)

        # Assign a texture index to each wall cell
        self.wall_tex_map = np.zeros((self.h, self.w), dtype=int)
        n_tex = len(self.wall_textures)
        for r in range(self.h):
            for c in range(self.w):
                if maze[r, c] == 1:
                    self.wall_tex_map[r, c] = rng.randint(0, n_tex)

        # Floor/ceiling colours
        self.floor_color = rng.randint(60, 140, 3).astype(np.uint8)
        self.ceiling_color = (self.floor_color * 0.6).astype(np.uint8)

    def render(self, pos: tuple[int, int], facing: int) -> np.ndarray:
        """Render a first-person view from grid position `pos` facing direction
        `facing` (0=East, 1=South, 2=West, 3=North).

        The camera is placed at the centre of the grid cell.
        """
        # Camera position in continuous coords (row=y, col=x)
        cam_x = pos[1] + 0.5  # column → x
        cam_y = pos[0] + 0.5  # row → y

        # Facing → angle in radians (East=0, South=π/2, West=π, North=3π/2)
        dir_angle = facing * (math.pi / 2.0)

        img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)

        # Draw floor and ceiling
        img[: self.img_h // 2] = self.ceiling_color
        img[self.img_h // 2 :] = self.floor_color

        half_fov = self.fov / 2.0

        for col in range(self.img_w):
            # Ray angle
            ray_frac = (col / self.img_w) - 0.5  # -0.5 to +0.5
            ray_angle = dir_angle + ray_frac * self.fov

            ray_dx = math.cos(ray_angle)
            ray_dy = math.sin(ray_angle)

            # DDA raycasting
            dist, hit_x, hit_y, side = self._cast_ray(
                cam_x, cam_y, ray_dx, ray_dy
            )

            if dist <= 0:
                continue

            # Fix fisheye
            perp_dist = dist * math.cos(ray_frac * self.fov)
            if perp_dist < 0.01:
                perp_dist = 0.01

            # Wall height
            line_height = int(self.img_h / perp_dist)
            draw_start = max(0, self.img_h // 2 - line_height // 2)
            draw_end = min(self.img_h, self.img_h // 2 + line_height // 2)

            if draw_end <= draw_start:
                continue

            # Texture coordinate (which column of the texture to use)
            if side == 0:  # vertical grid line hit
                wall_x = hit_y - math.floor(hit_y)
            else:  # horizontal grid line hit
                wall_x = hit_x - math.floor(hit_x)

            # Get the wall cell that was hit
            map_x = int(math.floor(hit_x - ray_dx * 0.001))
            map_y = int(math.floor(hit_y - ray_dy * 0.001))
            if side == 0:
                map_x = int(math.floor(hit_x)) - (1 if ray_dx < 0 else 0)
                map_y = int(math.floor(hit_y))
            else:
                map_x = int(math.floor(hit_x))
                map_y = int(math.floor(hit_y)) - (1 if ray_dy < 0 else 0)

            map_x = max(0, min(map_x, self.w - 1))
            map_y = max(0, min(map_y, self.h - 1))

            tex_idx = self.wall_tex_map[map_y, map_x]
            tex = self.wall_textures[tex_idx]

            tex_col = int(wall_x * self.tex_size) % self.tex_size

            # Draw the vertical stripe
            wall_stripe = tex[:, tex_col]  # (tex_size, 3)
            n_pixels = draw_end - draw_start

            # Map texture rows to screen rows
            tex_rows = np.linspace(0, self.tex_size - 1, n_pixels).astype(int)
            stripe = wall_stripe[tex_rows]  # (n_pixels, 3)

            # Distance-based shading (fog)
            shade = max(0.25, min(1.0, 1.5 / perp_dist))
            # Side shading (gives 3D depth cue)
            if side == 1:
                shade *= 0.7

            stripe = (stripe * shade).astype(np.uint8)
            img[draw_start:draw_end, col] = stripe

            # Floor shading with distance gradient
            if draw_end < self.img_h:
                n_floor = self.img_h - draw_end
                floor_shade = np.linspace(0.6, 0.2, n_floor)[:, None]
                img[draw_end:, col] = (self.floor_color * floor_shade).astype(np.uint8)

            if draw_start > 0:
                n_ceil = draw_start
                ceil_shade = np.linspace(0.2, 0.6, n_ceil)[:, None]
                img[:draw_start, col] = (self.ceiling_color * ceil_shade).astype(np.uint8)

        return img

    def _cast_ray(
        self, ox: float, oy: float, dx: float, dy: float
    ) -> tuple[float, float, float, int]:
        """DDA ray casting. Returns (distance, hit_x, hit_y, side)."""
        map_x, map_y = int(math.floor(ox)), int(math.floor(oy))

        # Avoid division by zero
        abs_dx = abs(dx) if abs(dx) > 1e-10 else 1e-10
        abs_dy = abs(dy) if abs(dy) > 1e-10 else 1e-10

        delta_dist_x = abs(1.0 / abs_dx)
        delta_dist_y = abs(1.0 / abs_dy)

        if dx < 0:
            step_x = -1
            side_dist_x = (ox - map_x) * delta_dist_x
        else:
            step_x = 1
            side_dist_x = (map_x + 1.0 - ox) * delta_dist_x

        if dy < 0:
            step_y = -1
            side_dist_y = (oy - map_y) * delta_dist_y
        else:
            step_y = 1
            side_dist_y = (map_y + 1.0 - oy) * delta_dist_y

        side = 0
        for _ in range(int(self.max_depth * 2) + 1):
            if side_dist_x < side_dist_y:
                side_dist_x += delta_dist_x
                map_x += step_x
                side = 0
            else:
                side_dist_y += delta_dist_y
                map_y += step_y
                side = 1

            # Bounds check
            if map_x < 0 or map_x >= self.w or map_y < 0 or map_y >= self.h:
                return self.max_depth, ox, oy, side

            # Wall hit
            if self.maze[map_y, map_x] == 1:
                if side == 0:
                    dist = side_dist_x - delta_dist_x
                else:
                    dist = side_dist_y - delta_dist_y
                hit_x = ox + dist * dx
                hit_y = oy + dist * dy
                return dist, hit_x, hit_y, side

        return self.max_depth, ox, oy, side


# ── Main ─────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate random mazes with rendered FPV images for training."
    )
    ap.add_argument("--n-mazes", type=int, default=30,
                    help="Number of mazes to generate")
    ap.add_argument("--size", type=int, default=31,
                    help="Maze grid size (odd number)")
    ap.add_argument("--output-dir", default="training_data/",
                    help="Root output directory")
    ap.add_argument("--texture-dir", default="data/textures/",
                    help="Directory with wall texture images (PNG/JPG)")
    ap.add_argument("--base-seed", type=int, default=42,
                    help="Base random seed")
    ap.add_argument("--img-w", type=int, default=320,
                    help="Rendered image width")
    ap.add_argument("--img-h", type=int, default=240,
                    help="Rendered image height")
    ap.add_argument("--jpeg-quality", type=int, default=90,
                    help="JPEG compression quality (0-100)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load textures once
    textures: list[np.ndarray] = []
    if os.path.isdir(args.texture_dir):
        for f in sorted(os.listdir(args.texture_dir)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                img = cv2.imread(os.path.join(args.texture_dir, f))
                if img is not None:
                    textures.append(img)
        print(f"Loaded {len(textures)} textures from {args.texture_dir}")
    else:
        print(f"No texture dir found at {args.texture_dir}, using procedural textures")

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]

    for i in range(args.n_mazes):
        seed = args.base_seed + i
        mdir = os.path.join(args.output_dir, f"maze_{i:03d}")
        img_dir = os.path.join(mdir, "images")
        os.makedirs(img_dir, exist_ok=True)

        # Generate maze
        maze = generate_maze(args.size, seed)
        open_cells = int((maze == 0).sum())

        # Simulate exploration
        traj = simulate_exploration(maze, seed)

        # Create renderer for this maze (each maze gets different texture assignments)
        renderer = MazeRenderer(
            maze,
            textures=textures if textures else None,
            img_w=args.img_w,
            img_h=args.img_h,
            seed=seed,
        )

        # Render images for every step in the trajectory
        info = []
        for si, entry in enumerate(traj):
            img = renderer.render(entry["position"], entry["facing"])
            fname = f"{entry['step']}.jpg"
            cv2.imwrite(os.path.join(img_dir, fname), img, encode_params)
            info.append({
                "step": entry["step"],
                "image": fname,
                "action": [entry["action"]],
            })

            if (si + 1) % 5000 == 0:
                print(f"    maze_{i:03d}: rendered {si + 1}/{len(traj)} images...")

        # Save metadata
        with open(os.path.join(mdir, "data_info.json"), "w") as f:
            json.dump(info, f)
        np.save(os.path.join(mdir, "maze_layout.npy"), maze)

        print(f"  maze_{i:03d}: {open_cells} open cells, "
              f"{len(traj)} steps → {len(info)} images saved")

    print(f"\nGenerated {args.n_mazes} mazes → {args.output_dir}")


if __name__ == "__main__":
    main()
