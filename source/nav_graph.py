"""
Navigation Graph: Topological graph from exploration trajectory.

Nodes = subsampled exploration frames (with DINOv2 features)
Edges = temporal (consecutive frames) + visual shortcuts (loop closures)
"""

import numpy as np
import networkx as nx
import pickle
import os
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Edge weights
TEMPORAL_FWD_WEIGHT = 1.0       # forward temporal edge cost
TEMPORAL_BWD_WEIGHT = 1.5       # backward temporal edge cost (slight penalty)
VISUAL_EDGE_WEIGHT = 0.5        # visual shortcut weight (near-free teleportation)

# Shortcut construction
MIN_SHORTCUT_GAP = 30           # minimum trajectory gap for shortcuts
GLOBAL_TOP_K = 500              # number of global visual shortcut edges
PER_NODE_TOP_K = 3              # per-node shortcut connections
PER_NODE_SIM_THRESHOLD = 0.80   # minimum similarity for per-node shortcuts

# Heuristic
HEURISTIC_SCALE = 5.0           # scale factor for A* heuristic

# Action reverse map
REVERSE_ACTION = {
    'FORWARD': 'BACKWARD',
    'BACKWARD': 'FORWARD',
    'LEFT': 'RIGHT',
    'RIGHT': 'LEFT',
}


class NavigationGraph:
    """
    Topological navigation graph built from exploration trajectory.

    Supports:
    - Temporal edges (forward + backward)
    - Visual shortcut edges (loop closures via feature similarity)
    - A* path planning with feature-similarity heuristic
    - Per-node shortcut guarantees (no isolated regions)
    """

    def __init__(self, features: np.ndarray, motion_frames: list[dict],
                 global_top_k: int = GLOBAL_TOP_K,
                 per_node_k: int = PER_NODE_TOP_K,
                 min_gap: int = MIN_SHORTCUT_GAP):
        """
        Args:
            features: (N, D) L2-normalized feature matrix
            motion_frames: list of dicts with 'action', 'step', 'image' keys
            global_top_k: number of global visual shortcut edges
            per_node_k: per-node shortcut connections
            min_gap: minimum trajectory index gap for shortcuts
        """
        self.features = features
        self.motion_frames = motion_frames
        self.n = len(features)
        self.global_top_k = global_top_k
        self.per_node_k = per_node_k
        self.min_gap = min_gap

        self.G = nx.DiGraph()  # directed graph for forward/backward distinction
        self._build()

    def _build(self):
        """Build the full navigation graph."""
        print(f"Building navigation graph ({self.n} nodes)...")
        self.G.add_nodes_from(range(self.n))

        self._add_temporal_edges()
        self._add_visual_shortcuts()
        self._add_per_node_shortcuts()

        print(f"  Graph: {self.G.number_of_nodes()} nodes, "
              f"{self.G.number_of_edges()} edges")

    def _add_temporal_edges(self):
        """Add forward and backward temporal edges."""
        for i in range(self.n - 1):
            action = self.motion_frames[i]['action']

            # Forward: i → i+1 with recorded action
            self.G.add_edge(i, i + 1,
                            weight=TEMPORAL_FWD_WEIGHT,
                            action=action,
                            edge_type='temporal_forward')

            # Backward: i+1 → i with reversed action
            rev_action = REVERSE_ACTION.get(action, 'BACKWARD')
            self.G.add_edge(i + 1, i,
                            weight=TEMPORAL_BWD_WEIGHT,
                            action=rev_action,
                            edge_type='temporal_backward')

        print(f"  Added {2 * (self.n - 1)} temporal edges")

    def _add_visual_shortcuts(self):
        """Add global top-K visual shortcut edges."""
        print("  Computing similarity matrix for shortcuts...")
        sim = self.features @ self.features.T

        # Mask diagonal and nearby frames
        np.fill_diagonal(sim, -2)
        for i in range(self.n):
            lo = max(0, i - self.min_gap)
            hi = min(self.n, i + self.min_gap + 1)
            sim[i, lo:hi] = -2

        # Only upper triangle to avoid duplicate edges
        sim_upper = sim.copy()
        sim_upper[np.tril_indices(self.n)] = -2

        # Extract top-K
        flat = sim_upper.ravel()
        k = min(self.global_top_k, (flat > -1).sum())
        if k == 0:
            print("  No valid shortcut candidates found")
            return

        top_idx = np.argpartition(flat, -k)[-k:]
        top_idx = top_idx[np.argsort(-flat[top_idx])]

        count = 0
        for fi in top_idx:
            i, j = divmod(int(fi), self.n)
            s = float(sim_upper[i, j])
            if s <= 0:
                continue

            # Bidirectional visual edges
            self.G.add_edge(i, j,
                            weight=VISUAL_EDGE_WEIGHT,
                            similarity=s,
                            edge_type='visual')
            self.G.add_edge(j, i,
                            weight=VISUAL_EDGE_WEIGHT,
                            similarity=s,
                            edge_type='visual')
            count += 1

        print(f"  Added {count * 2} global visual shortcut edges")

    def _add_per_node_shortcuts(self):
        """Ensure every node has at least per_node_k shortcut connections."""
        count = 0
        for i in range(self.n):
            # Check existing visual edges
            existing_visual = sum(
                1 for _, _, d in self.G.edges(i, data=True)
                if d.get('edge_type') == 'visual'
            )
            if existing_visual >= self.per_node_k:
                continue

            # Find top-K most similar non-nearby nodes
            sims = self.features[i] @ self.features.T
            sims[max(0, i - self.min_gap):min(self.n, i + self.min_gap + 1)] = -1

            needed = self.per_node_k - existing_visual
            top_k = np.argsort(sims)[-needed:]

            for j in top_k:
                j = int(j)
                if sims[j] > PER_NODE_SIM_THRESHOLD:
                    if not self.G.has_edge(i, j) or \
                            self.G[i][j].get('edge_type') != 'visual':
                        self.G.add_edge(i, j,
                                        weight=VISUAL_EDGE_WEIGHT,
                                        similarity=float(sims[j]),
                                        edge_type='visual')
                        self.G.add_edge(j, i,
                                        weight=VISUAL_EDGE_WEIGHT,
                                        similarity=float(sims[j]),
                                        edge_type='visual')
                        count += 1

        print(f"  Added {count * 2} per-node shortcut edges")

    # --- Path Planning ---

    def find_path(self, start: int, goal: int,
                  goal_feature: Optional[np.ndarray] = None) -> list[int]:
        """
        Find shortest path from start to goal using A*.

        Args:
            start: source node index
            goal: target node index
            goal_feature: optional goal feature for A* heuristic

        Returns:
            list of node indices from start to goal
        """
        if start == goal:
            return [start]

        try:
            if goal_feature is not None:
                # A* with feature-similarity heuristic
                def heuristic(node, _goal):
                    sim = float(self.features[node] @ goal_feature)
                    return max(0, 1 - sim) * HEURISTIC_SCALE

                return nx.astar_path(self.G, start, goal,
                                     heuristic=heuristic, weight='weight')
            else:
                return nx.shortest_path(self.G, start, goal, weight='weight')
        except nx.NetworkXNoPath:
            # Fallback: try undirected
            try:
                G_undirected = self.G.to_undirected()
                return nx.shortest_path(G_undirected, start, goal, weight='weight')
            except nx.NetworkXNoPath:
                return [start]

    def get_edge_action(self, a: int, b: int) -> str:
        """Get the action needed to traverse edge a → b."""
        if self.G.has_edge(a, b):
            data = self.G[a][b]
            edge_type = data.get('edge_type', 'temporal_forward')
            if edge_type.startswith('temporal'):
                return data.get('action', '?')
            else:
                # Visual edge — no physical action needed
                return 'VISUAL'
        return '?'

    def get_path_actions(self, path: list[int]) -> list[str]:
        """Get sequence of actions for a path."""
        actions = []
        for a, b in zip(path[:-1], path[1:]):
            actions.append(self.get_edge_action(a, b))
        return actions

    # --- Save / Load ---

    def save(self, path: str):
        """Save graph to pickle."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            'graph': self.G,
            'n': self.n,
            'motion_frames': self.motion_frames,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved graph to {path}")

    @classmethod
    def load(cls, path: str, features: np.ndarray) -> 'NavigationGraph':
        """Load graph from pickle."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.G = data['graph']
        obj.n = data['n']
        obj.motion_frames = data['motion_frames']
        obj.features = features
        obj.global_top_k = GLOBAL_TOP_K
        obj.per_node_k = PER_NODE_TOP_K
        obj.min_gap = MIN_SHORTCUT_GAP
        print(f"Loaded graph from {path} ({obj.n} nodes, "
              f"{obj.G.number_of_edges()} edges)")
        return obj
