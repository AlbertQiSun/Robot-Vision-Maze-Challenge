"""
Topological navigation graph built from an exploration trajectory.

Nodes = sub-sampled exploration frames.
Edges = temporal (consecutive) + visual shortcuts (loop closures).
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

import networkx as nx
import numpy as np

from vis_nav.config import graph_cfg as C, REVERSE_ACTION


class NavigationGraph:
    """Directed topological graph with temporal + visual edges."""

    def __init__(
        self,
        features: np.ndarray,
        motion_frames: list[dict],
        global_top_k: int = C.global_top_k,
        per_node_k: int = C.per_node_top_k,
        min_gap: int = C.min_shortcut_gap,
    ):
        self.features = features
        self.motion_frames = motion_frames
        self.n = len(features)
        self.global_top_k = global_top_k
        self.per_node_k = per_node_k
        self.min_gap = min_gap

        self.G = nx.DiGraph()
        self._build()

    # ── Build ────────────────────────────────────────────────────────
    def _build(self) -> None:
        print(f"Building navigation graph ({self.n} nodes)...")
        self.G.add_nodes_from(range(self.n))
        self._add_temporal_edges()
        self._add_visual_shortcuts()
        self._add_per_node_shortcuts()
        print(
            f"  Graph: {self.G.number_of_nodes()} nodes, "
            f"{self.G.number_of_edges()} edges"
        )

    def _add_temporal_edges(self) -> None:
        for i in range(self.n - 1):
            action = self.motion_frames[i]["action"]
            self.G.add_edge(
                i, i + 1,
                weight=C.temporal_fwd_weight, action=action,
                edge_type="temporal_forward",
            )
            # Add temporal backward edges so the robot can retrace its steps.
            # Walking backwards keeps the camera facing the known exploration view,
            # which is critical since we don't have images of the reverse direction!
            self.G.add_edge(
                i + 1, i,
                weight=C.temporal_bwd_weight,
                action=REVERSE_ACTION.get(action, "BACKWARD"),
                edge_type="temporal_backward",
            )
        print(f"  Added {2 * (self.n - 1)} temporal edges")

    def _add_visual_shortcuts(self) -> None:
        print("  Computing similarity matrix for shortcuts...")
        sim = self.features @ self.features.T
        np.fill_diagonal(sim, -2)
        for i in range(self.n):
            lo, hi = max(0, i - self.min_gap), min(self.n, i + self.min_gap + 1)
            sim[i, lo:hi] = -2

        sim_upper = sim.copy()
        sim_upper[np.tril_indices(self.n)] = -2

        flat = sim_upper.ravel()
        k = min(self.global_top_k, int((flat > -1).sum()))
        if k == 0:
            return

        top_idx = np.argpartition(flat, -k)[-k:]
        top_idx = top_idx[np.argsort(-flat[top_idx])]

        count = 0
        for fi in top_idx:
            i, j = divmod(int(fi), self.n)
            s = float(sim_upper[i, j])
            if s <= 0:
                continue
            for a, b in ((i, j), (j, i)):
                self.G.add_edge(
                    a, b, weight=C.visual_edge_weight,
                    similarity=s, edge_type="visual",
                )
            count += 1
        print(f"  Added {count * 2} global visual shortcut edges")

    def _add_per_node_shortcuts(self) -> None:
        count = 0
        for i in range(self.n):
            existing = sum(
                1 for _, _, d in self.G.edges(i, data=True)
                if d.get("edge_type") == "visual"
            )
            if existing >= self.per_node_k:
                continue

            sims = self.features[i] @ self.features.T
            sims[max(0, i - self.min_gap) : min(self.n, i + self.min_gap + 1)] = -1

            needed = self.per_node_k - existing
            for j in np.argsort(sims)[-needed:]:
                j = int(j)
                if sims[j] > C.per_node_sim_threshold:
                    for a, b in ((i, j), (j, i)):
                        if not self.G.has_edge(a, b) or \
                                self.G[a][b].get("edge_type") != "visual":
                            self.G.add_edge(
                                a, b, weight=C.visual_edge_weight,
                                similarity=float(sims[j]), edge_type="visual",
                            )
                    count += 1
        print(f"  Added {count * 2} per-node shortcut edges")

    # ── Path planning ────────────────────────────────────────────────
    @staticmethod
    def _edge_weight(u: int, v: int, d: dict) -> float:
        """Weight function used by A* / Dijkstra at query time.

        Visual shortcut edges are stored with weight=0.3 (built during
        exploration) but at navigation time we strongly penalise them
        (weight=50) so the planner follows the recorded temporal path
        instead of routing the robot through shortcuts that may cross walls.
        A shortcut is only taken if the temporal detour would cost > 50 hops,
        which essentially never happens in a 31×31 maze.
        """
        if d.get("edge_type", "").startswith("temporal"):
            return d.get("weight", 1.0)
        return 50.0  # visual shortcuts: last-resort only

    def find_path(
        self,
        start: int,
        goal: int,
        goal_feature: Optional[np.ndarray] = None,
    ) -> list[int]:
        if start == goal:
            return [start]
        try:
            if goal_feature is not None:
                def heuristic(node, _goal):
                    sim = float(self.features[node] @ goal_feature)
                    return max(0, 1 - sim) * C.heuristic_scale
                return nx.astar_path(
                    self.G, start, goal,
                    heuristic=heuristic, weight=self._edge_weight,
                )
            return nx.dijkstra_path(self.G, start, goal, weight=self._edge_weight)
        except nx.NetworkXNoPath:
            try:
                # Undirected fallback still prefers temporal edges
                return nx.dijkstra_path(
                    self.G.to_undirected(), start, goal, weight=self._edge_weight,
                )
            except nx.NetworkXNoPath:
                return [start]

    def get_edge_action(self, a: int, b: int) -> str:
        if self.G.has_edge(a, b):
            d = self.G[a][b]
            if d.get("edge_type", "").startswith("temporal"):
                return d.get("action", "?")
            return "VISUAL"
        return "?"

    def get_path_actions(self, path: list[int]) -> list[str]:
        return [self.get_edge_action(a, b) for a, b in zip(path, path[1:])]

    # ── Serialisation ────────────────────────────────────────────────
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"graph": self.G, "n": self.n, "motion_frames": self.motion_frames}, f,
            )

    @classmethod
    def load(cls, path: str, features: np.ndarray) -> NavigationGraph:
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.G = data["graph"]
        obj.n = data["n"]
        obj.motion_frames = data["motion_frames"]
        obj.features = features
        obj.global_top_k = C.global_top_k
        obj.per_node_k = C.per_node_top_k
        obj.min_gap = C.min_shortcut_gap
        return obj
