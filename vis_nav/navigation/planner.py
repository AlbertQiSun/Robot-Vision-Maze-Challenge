"""
Goal planner: multi-view goal matching, check-in logic, stuck recovery.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Optional

import networkx as nx
import numpy as np

from vis_nav.config import nav_cfg as C, ACTION_TO_IDX, IDX_TO_ACTION
from vis_nav.data.transforms import extract_single_feature
from vis_nav.navigation.localizer import Localizer


class GoalPlanner:
    """
    High-level navigation controller.

    Responsibilities:
    - Set up the goal node from 4 target views.
    - Compute and follow a path.
    - Decide when to check in.
    - Detect and recover from stuck situations.
    """

    def __init__(
        self,
        localizer: Localizer,
        features: np.ndarray,
    ):
        self.localizer = localizer
        self.features = features

        # Goal state
        self.goal_node: int | None = None
        self.goal_features: list[np.ndarray] = []
        self.goal_candidates: list[int] = []
        self.goal_feature_fused: np.ndarray | None = None

        # Path-following state
        self.current_node: int | None = None
        self.current_path: list[int] | None = None
        self.path_index: int = 0

        # Stuck detection
        self.stuck_buffer: deque[np.ndarray] = deque(maxlen=C.stuck_window)
        self.stuck_counter: int = 0
        self.recovery_phase: int = 0

        # Check-in
        self.checkin_confidence: int = 0

    # ── Goal setup ───────────────────────────────────────────────────
    def setup_goal(
        self,
        target_images: list[np.ndarray],
        feat_model,
        device,
    ) -> None:
        """Match all 4 target views against the exploration database."""
        view_weights = [1.0, 0.7, 0.5, 0.7]

        self.goal_features = [
            extract_single_feature(feat_model, img, device) for img in target_images
        ]

        candidates: dict[int, float] = {}
        for feat, weight in zip(self.goal_features, view_weights):
            idxs, scores = self.localizer.query_top_k(feat, C.rerank_top_k)
            for rank, (idx, sim) in enumerate(zip(idxs, scores)):
                score = float(sim) * weight / (1 + 0.1 * rank)
                candidates[idx] = candidates.get(idx, 0) + score

        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        self.goal_node = ranked[0][0]
        self.goal_candidates = [c[0] for c in ranked[:5]]

        w = np.array(view_weights[: len(self.goal_features)])
        w /= w.sum()
        fused = sum(wi * fi for wi, fi in zip(w, self.goal_features))
        norm = np.linalg.norm(fused)
        self.goal_feature_fused = fused / norm if norm > 0 else fused

        print(
            f"  Goal node: {self.goal_node} (score: {ranked[0][1]:.4f})\n"
            f"  Backup candidates: {self.goal_candidates}"
        )

    # ── Localise + plan ──────────────────────────────────────────────
    def localize_and_plan(self, query_feat: np.ndarray, nav_graph) -> None:
        self.stuck_buffer.append(query_feat)
        self.current_node = self.localizer.localize(query_feat)

        if self.goal_node is None:
            return

        self.current_path = nav_graph.find_path(
            self.current_node, self.goal_node,
            goal_feature=self.goal_feature_fused,
        )
        self.path_index = 0

    # ── Check-in ─────────────────────────────────────────────────────
    def should_checkin(self, query_feat: np.ndarray, nav_graph=None) -> bool:
        if not self.goal_features:
            return False

        sims = [float(np.dot(query_feat, gf)) for gf in self.goal_features]
        if max(sims) > C.checkin_sim_threshold:
            self.checkin_confidence += 1
        else:
            self.checkin_confidence = max(0, self.checkin_confidence - 1)

        if (
            nav_graph is not None
            and self.current_node is not None
            and self.goal_node is not None
        ):
            try:
                dist = nx.shortest_path_length(
                    nav_graph.G, self.current_node, self.goal_node, weight="weight",
                )
                if dist <= C.checkin_graph_dist:
                    self.checkin_confidence += 2
            except Exception:
                pass

        return self.checkin_confidence >= C.checkin_confidence_needed

    # ── Stuck detection ──────────────────────────────────────────────
    _RECOVERY_ACTIONS = ("LEFT", "FORWARD", "RIGHT", "RIGHT", "FORWARD", "BACKWARD")

    def check_stuck(self) -> str | None:
        """Return an action name if stuck, else ``None``."""
        if len(self.stuck_buffer) < C.stuck_window:
            return None

        recent = np.array(list(self.stuck_buffer))
        pw = recent @ recent.T
        n = len(recent)
        avg = (pw.sum() - np.trace(pw)) / (n * (n - 1))

        if avg > C.stuck_sim_threshold:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            return None

        if self.stuck_counter < C.stuck_patience:
            return None

        self.recovery_phase = (self.recovery_phase + 1) % len(self._RECOVERY_ACTIONS)
        self.current_path = None
        self.stuck_counter = 0
        return self._RECOVERY_ACTIONS[self.recovery_phase]
