"""
Multi-Maze Data Generator.

Generates multiple random mazes using the game engine and runs
exploration to collect training images.

This script uses the Maze class from vis_nav_game_public to create
random mazes, then simulates exploration to collect FPV images.

Usage:
  python source/generate_mazes.py \
    --n-mazes 30 \
    --size 31 \
    --output-dir training_data/ \
    --game-engine-path vis_nav_game_public/

NOTE: This script generates maze LAYOUTS only (the numpy maze arrays +
      data_info.json stubs). To collect actual exploration images, you
      need to run the game engine on each maze. On a CUDA machine,
      run the game in headless mode for each generated maze.

      Alternatively, if you have the practice maze data already in data/,
      you can train on that single maze first, then add more later.
"""

import argparse
import json
import os
import sys
import random
import numpy as np
from pathlib import Path


def generate_maze(size: int, seed: int) -> np.ndarray:
    """
    Generate a random maze using depth-first search.
    Replicates the logic from vis_nav_game_public/maze.py.

    Args:
        size: maze dimensions (size × size), should be odd
        seed: random seed for reproducibility

    Returns:
        (size, size) numpy array: 0 = path, 1 = wall
    """
    np.random.seed(seed)
    random.seed(seed)

    maze = np.ones((size, size), dtype=int)

    # Outer walls
    maze[0, :] = 1
    maze[-1, :] = 1
    maze[:, 0] = 1
    maze[:, -1] = 1

    # Start from (1,1)
    maze[1, 1] = 0

    # DFS maze generation
    stack = [(1, 1)]
    while stack:
        row, col = stack.pop()
        neighbors = []
        for dr, dc in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < size and 0 <= nc < size and maze[nr, nc] == 1:
                neighbors.append((nr, nc))

        if neighbors:
            nr, nc = neighbors[np.random.randint(len(neighbors))]
            maze[row + (nr - row) // 2, col + (nc - col) // 2] = 0
            maze[nr, nc] = 0
            stack.append((row, col))
            stack.append((nr, nc))

    # Add random rooms (same as game engine)
    room_density = 0.1
    max_room_size = 3
    for i in range(2, size - 2):
        for j in range(2, size - 2):
            if maze[i, j] == 0 and np.random.rand() < room_density:
                room_h = np.random.randint(1, max_room_size + 1)
                room_w = np.random.randint(1, max_room_size + 1)
                if i + room_h < size and j + room_w < size:
                    maze[i:i + room_h, j:j + room_w] = 0

    return maze


def simulate_exploration(maze: np.ndarray, seed: int) -> list[dict]:
    """
    Simulate a simple DFS exploration through the maze.
    Generates a sequence of actions that visits accessible cells.

    Args:
        maze: (size, size) array, 0 = path, 1 = wall
        seed: random seed

    Returns:
        list of dicts: {'step': int, 'action': str, 'position': (r, c)}
    """
    random.seed(seed)
    size = maze.shape[0]

    # Start at (1, 1) facing east
    start = (1, 1)
    facing = 0  # 0=east, 1=south, 2=west, 3=north
    dr = [0, 1, 0, -1]
    dc = [1, 0, -1, 0]

    trajectory = []
    visited = set()
    stack = [(start, facing)]
    pos = start
    step = 0

    # Wall-following exploration
    visited.add(pos)
    max_steps = 20000

    while step < max_steps:
        # Try: right, forward, left, backward
        for turn_name, turn_delta in [('RIGHT', 1), ('FORWARD', 0),
                                       ('LEFT', -1), ('BACKWARD', 2)]:
            new_facing = (facing + turn_delta) % 4
            nr = pos[0] + dr[new_facing]
            nc = pos[1] + dc[new_facing]

            if 0 <= nr < size and 0 <= nc < size and maze[nr, nc] == 0:
                # Turn first if needed
                if turn_delta == 1:
                    trajectory.append({'step': step, 'action': 'RIGHT',
                                       'position': pos})
                    step += 1
                elif turn_delta == -1:
                    trajectory.append({'step': step, 'action': 'LEFT',
                                       'position': pos})
                    step += 1
                elif turn_delta == 2:
                    trajectory.append({'step': step, 'action': 'LEFT',
                                       'position': pos})
                    step += 1
                    trajectory.append({'step': step, 'action': 'LEFT',
                                       'position': pos})
                    step += 1

                # Move forward
                trajectory.append({'step': step, 'action': 'FORWARD',
                                   'position': pos})
                step += 1
                pos = (nr, nc)
                facing = new_facing
                visited.add(pos)
                break
        else:
            # Dead end — just turn right
            trajectory.append({'step': step, 'action': 'RIGHT',
                               'position': pos})
            step += 1
            facing = (facing + 1) % 4

        # Check if we've visited enough
        total_open = (maze == 0).sum()
        if len(visited) >= total_open * 0.95:
            break

    return trajectory


def main():
    parser = argparse.ArgumentParser(description="Generate training mazes")
    parser.add_argument("--n-mazes", type=int, default=30,
                        help="Number of mazes to generate")
    parser.add_argument("--size", type=int, default=31,
                        help="Maze size (should be odd)")
    parser.add_argument("--output-dir", type=str, default="training_data/",
                        help="Output directory for maze data")
    parser.add_argument("--base-seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--game-engine-path", type=str,
                        default="vis_nav_game_public/",
                        help="Path to vis_nav_game_public repo")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating {args.n_mazes} mazes ({args.size}×{args.size})...")
    print(f"Output: {args.output_dir}")
    print()

    for maze_idx in range(args.n_mazes):
        seed = args.base_seed + maze_idx
        maze_dir = os.path.join(args.output_dir, f"maze_{maze_idx:03d}")
        os.makedirs(os.path.join(maze_dir, "images"), exist_ok=True)

        # Generate maze layout
        maze = generate_maze(args.size, seed)
        open_cells = (maze == 0).sum()

        # Simulate exploration
        trajectory = simulate_exploration(maze, seed)

        # Save data_info.json (mimicking game engine format)
        data_info = []
        for entry in trajectory:
            data_info.append({
                'step': entry['step'],
                'image': f"{entry['step']}.jpg",
                'action': [entry['action']],
            })

        with open(os.path.join(maze_dir, "data_info.json"), 'w') as f:
            json.dump(data_info, f)

        # Save maze layout for reference
        np.save(os.path.join(maze_dir, "maze_layout.npy"), maze)

        print(f"  Maze {maze_idx:03d}: {open_cells} open cells, "
              f"{len(trajectory)} exploration steps")

    print(f"\nGenerated {args.n_mazes} maze layouts in {args.output_dir}")
    print()
    print("NEXT STEPS:")
    print("  To collect actual exploration images, you need to run the")
    print("  game engine on each generated maze. On a CUDA machine:")
    print()
    print("  1. Copy the maze layouts to the game engine data directory")
    print("  2. Run: python vis_nav_core.py for each maze")
    print("  3. The engine will generate FPV images during exploration")
    print()
    print("  For now, you can train on the practice maze in data/")
    print("  using: python source/train_projection.py --data-dir data/ --single-maze")


if __name__ == "__main__":
    main()
