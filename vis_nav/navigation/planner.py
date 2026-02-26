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

        # Stuck detection — node-spread based (not feature-similarity)
        # Holds recent current_node integers; spread < stuck_node_spread → stuck
        self.stuck_buffer: deque[int] = deque(maxlen=C.stuck_window)
        self.stuck_counter: int = 0
        self.recovery_phase: int = 0
        self._recovery_queue: list[str] = []
        # After a recovery ends, ignore stuck detection for N steps so the robot
        # has time to actually move before we evaluate again.
        self._post_recovery_cooldown: int = 0

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

        w = np.array(view_weights[: len(self.goal_features)], dtype=np.float32)
        w /= w.sum()
        stacked = np.stack(self.goal_features).astype(np.float32)  # (V, dim)
        fused = (w[:, None] * stacked).sum(axis=0)                 # weighted mean
        norm = np.linalg.norm(fused)
        self.goal_feature_fused = (fused / norm if norm > 0 else fused).astype(np.float32)

        print(
            f"  Goal node: {self.goal_node} (score: {ranked[0][1]:.4f})\n"
            f"  Backup candidates: {self.goal_candidates}"
        )

    # ── Localise + plan ──────────────────────────────────────────────
    def localize_and_plan(self, query_feat: np.ndarray, nav_graph) -> None:
        self.current_node = self.localizer.localize(query_feat)

        if self.goal_node is None:
            return

        self.current_path = nav_graph.find_path(
            self.current_node, self.goal_node,
            goal_feature=self.goal_feature_fused,
        )
        self.path_index = 0

    # ── Check-in ─────────────────────────────────────────────────────
    @staticmethod
    def _checkin_edge_weight(u: int, v: int, d: dict) -> float:
        """Same weighting as NavigationGraph._edge_weight: visual shortcuts cost 50."""
        if d.get("edge_type", "").startswith("temporal"):
            return d.get("weight", 1.0)
        return 50.0

    def should_checkin(self, query_feat: np.ndarray, nav_graph=None) -> bool:
        if not self.goal_features:
            return False

        sims = [float(np.dot(query_feat, gf)) for gf in self.goal_features]
        best_sim = max(sims)

        # Hard gate: if visual similarity is far too low to plausibly be at goal,
        # reset all accumulated confidence and bail out. Prevents the graph-distance
        # path (+2 per step) from overriding clear visual evidence that we're NOT there.
        # This was the root cause of a false CHECKIN at sim=0.419 (step 190).
        if best_sim < C.checkin_min_sim:
            self.checkin_confidence = 0
            return False

        if best_sim > C.checkin_sim_threshold:
            self.checkin_confidence += 1
        else:
            self.checkin_confidence = max(0, self.checkin_confidence - 1)

        if (
            nav_graph is not None
            and self.current_node is not None
            and self.goal_node is not None
        ):
            try:
                # IMPORTANT: use visual-edge-weighted distance (same 50-weight as
                # find_path) so that a single visual shortcut does NOT appear as
                # dist=1 and incorrectly boost confidence.
                dist = nx.shortest_path_length(
                    nav_graph.G, self.current_node, self.goal_node,
                    weight=self._checkin_edge_weight,
                )
                if dist <= C.checkin_graph_dist:
                    self.checkin_confidence += 2
            except Exception:
                pass

        return self.checkin_confidence >= C.checkin_confidence_needed

    # ── Stuck detection ──────────────────────────────────────────────
    # Recovery sequences.
    # OLD design had 40-action sequences ending with FORWARD×10 — the robot
    # walked straight back into the obstacle.  New design:
    #   • Short BACKWARD (3–5 steps) to clear the wall
    #   • A decisive turn (8–14 steps)
    #   • Short FORWARD (3–5 steps) only — enough to commit to new direction
    # The planner replans from the new position after recovery completes.
    _RECOVERY_SEQUENCES = [
        ["BACKWARD"] * 4 + ["LEFT"] * 8  + ["FORWARD"] * 4,   # back-left
        ["BACKWARD"] * 4 + ["RIGHT"] * 8 + ["FORWARD"] * 4,   # back-right
        ["LEFT"]  * 14 + ["FORWARD"] * 4,                      # 180° left + step
        ["RIGHT"] * 14 + ["FORWARD"] * 4,                      # 180° right + step
        ["BACKWARD"] * 6 + ["LEFT"]  * 12 + ["FORWARD"] * 3,  # stronger left
        ["BACKWARD"] * 6 + ["RIGHT"] * 12 + ["FORWARD"] * 3,  # stronger right
    ]

    def check_stuck(self) -> str | None:
        """Return an action name if stuck, else ``None``.

        Uses **node-spread** detection instead of feature-similarity:
        if the range of recently localised nodes (max − min) is smaller
        than ``stuck_node_spread``, the robot hasn't made topological
        progress and is considered stuck.

        This avoids the false-positive triggered during normal navigation
        through a visually-repetitive corridor (feature similarity is
        naturally high even when the robot is moving).
        """
        # If we're in the middle of a recovery burst, keep going
        if self._recovery_queue:
            return self._recovery_queue.pop(0)

        # Post-recovery cooldown: give the robot time to actually move before
        # evaluating stuck again. Without this (combined with the buffer guard in
        # player.py), two normal steps were enough to re-trigger at phase 44+.
        if self._post_recovery_cooldown > 0:
            self._post_recovery_cooldown -= 1
            self.stuck_buffer.clear()   # discard any residual entries
            return None

        if len(self.stuck_buffer) < C.stuck_window:
            return None

        recent_nodes = list(self.stuck_buffer)
        spread = max(recent_nodes) - min(recent_nodes)

        if spread < C.stuck_node_spread:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            return None

        if self.stuck_counter < C.stuck_patience:
            return None

        # Trigger a recovery burst
        seq = self._RECOVERY_SEQUENCES[
            self.recovery_phase % len(self._RECOVERY_SEQUENCES)
        ]
        self.recovery_phase += 1
        self._recovery_queue = list(seq[1:])
        self.current_path = None
        self.stuck_counter = 0
        self.stuck_buffer.clear()
        self._post_recovery_cooldown = C.post_recovery_cooldown
        return seq[0]
