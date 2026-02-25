#!/usr/bin/env python3
"""
Entry point — autonomous visual navigation player.

Run:
    python player.py
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from collections import Counter, deque

log = logging.getLogger(__name__)

import cv2
import numpy as np
import pygame
import torch

from vis_nav_game import Player, Action, Phase

from vis_nav.config import (
    paths, nav_cfg, feat_cfg,
    ACTION_TO_IDX, IDX_TO_ACTION,
)
from vis_nav.utils import (
    get_device, ensure_dirs, compute_cache_hash, load_trajectory,
)
from vis_nav.models import PlaceFeatureExtractor, ActionPredictor
from vis_nav.data import extract_features_batch, extract_single_feature
from vis_nav.navigation import NavigationGraph, Localizer, GoalPlanner

# Action helpers
_ACT_MAP = {
    "FORWARD": Action.FORWARD,
    "BACKWARD": Action.BACKWARD,
    "LEFT": Action.LEFT,
    "RIGHT": Action.RIGHT,
}
_ACT_NAMES = {v: k for k, v in _ACT_MAP.items()}


class KeyboardPlayerPyGame(Player):
    """
    Fully autonomous visual navigation player.

    * Exploration phase — passively receives FPV frames.
    * Pre-navigation   — builds feature database, FAISS index, graph, goal.
    * Navigation phase — autonomous path following with stuck recovery.
    * Falls back to keyboard if the autonomous pipeline fails.
    """

    def __init__(self) -> None:
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap: dict | None = None

        # autonomous controller state
        self.autonomous = False
        self.step_count = 0
        self.device = get_device()

        # modules (populated in pre_navigation)
        self.feat_model: PlaceFeatureExtractor | None = None
        self.action_predictor: ActionPredictor | None = None
        self.features: np.ndarray | None = None
        self.file_list: list[str] = []
        self.motion_frames: list[dict] = []
        self.nav_graph: NavigationGraph | None = None
        self.localizer: Localizer | None = None
        self.planner: GoalPlanner | None = None
        self.action_history: deque = deque(maxlen=20)
        self._last_node: int | None = None
        self._same_node_count: int = 0
        self._path_steps_taken: int = 0
        self._current_feat: np.ndarray | None = None
        self._last_turn: str = "LEFT"  # track turn direction for consistency

        # Anti-shake: action smoothing + re-plan throttle
        self._action_vote: deque = deque(maxlen=nav_cfg.action_smooth_window)
        self._last_replan_node: int | None = None
        self._replan_cooldown: int = 0

        # Frame-cache: avoid redundant DINOv2 calls (Bug 9)
        self._last_fpv_hash: int | None = None

        super().__init__()

    # ── Game hooks ───────────────────────────────────────────────────
    def reset(self) -> None:
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        pygame.init()
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT,
        }

    def see(self, fpv) -> None:
        if fpv is None or len(fpv.shape) < 3:
            return
        self.fpv = fpv
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(
            f"AutoNav | step {self.step_count} | "
            f"node {getattr(self.planner, 'current_node', '?')} | "
            f"goal {getattr(self.planner, 'goal_node', '?')}"
        )
        surface = pygame.image.frombuffer(
            fpv[:, :, ::-1].tobytes(), fpv.shape[1::-1], "RGB",
        )
        self.screen.blit(surface, (0, 0))
        pygame.display.update()

    def set_target_images(self, images) -> None:
        super().set_target_images(images)
        self._show_targets()

    def pre_exploration(self) -> None:
        log.info("Camera K = %s", self.get_camera_intrinsic_matrix())

    # ── Pipeline build ───────────────────────────────────────────────
    def pre_navigation(self) -> None:
        super().pre_navigation()
        ensure_dirs()
        log.info("=" * 60)
        log.info("PRE-NAVIGATION: building autonomous pipeline")
        t0 = time.time()
        try:
            self._load_models()
            self._load_trajectory()
            self._extract_features()
            self._build_navigation()
            self._setup_goal()
            self.autonomous = True
            log.info(
                "Ready in %.1fs  (%d nodes, %d edges, goal=%s)",
                time.time() - t0,
                len(self.features),
                self.nav_graph.G.number_of_edges(),
                self.planner.goal_node,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            log.warning("[WARN] Autonomous init failed: %s  → keyboard mode", exc)
            self.autonomous = False
        log.info("=" * 60)

    # ── Action ───────────────────────────────────────────────────────
    def act(self) -> Action:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                return Action.QUIT
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return Action.QUIT
                if not self.autonomous and ev.key in self.keymap:
                    self.last_act |= self.keymap[ev.key]
                elif ev.key == pygame.K_t:
                    self._show_targets()
            if ev.type == pygame.KEYUP:
                if not self.autonomous and ev.key in self.keymap:
                    self.last_act ^= self.keymap[ev.key]

        if not self.autonomous or self.fpv is None:
            return self.last_act

        self.step_count += 1

        # ── Extract feature (with frame-hash cache to skip redundant DINOv2) ──
        fpv_hash = hash(self.fpv.tobytes()[::1000])
        if fpv_hash != self._last_fpv_hash:
            self._current_feat = extract_single_feature(
                self.feat_model, self.fpv, self.device,
            )
            self._last_fpv_hash = fpv_hash

        # ── Localize first so current_node is valid for stuck detection ──
        p = self.planner
        p.current_node = self.localizer.localize(self._current_feat)

        # ── Stuck detection: feed node ID (int) into buffer + check recovery ──
        # CRITICAL: only append during normal navigation, NOT during active recovery.
        # If we append while _recovery_queue is non-empty the buffer fills with
        # "near-same node" entries from the recovery moves themselves → the window
        # is already full of stuck-looking data the moment recovery finishes →
        # re-triggers in 2 steps → infinite recovery loop (seen as phase 44+).
        if not p._recovery_queue:
            p.stuck_buffer.append(p.current_node)
        recovery = p.check_stuck()
        if recovery:
            self._action_vote.clear()
            log.info("[RECOVERY] step %d → %s  (phase=%d queue=%d)",
                     self.step_count, recovery,
                     p.recovery_phase, len(p._recovery_queue))
            self.action_history.append(ACTION_TO_IDX.get(recovery, 0))
            return _ACT_MAP[recovery]

        # ── Check-in: use planner's full check-in logic (similarity + graph dist) ──
        if p.should_checkin(self._current_feat, self.nav_graph):
            best_sim = max(float(np.dot(self._current_feat, gf))
                          for gf in p.goal_features) if p.goal_features else 0
            log.info("[CHECKIN] step %d  sim=%.3f", self.step_count, best_sim)
            return Action.CHECKIN

        # ── Plan path to goal (throttled to reduce oscillation) ──
        self._replan_cooldown = max(0, self._replan_cooldown - 1)
        node_jumped = (
            self._last_replan_node is None
            or abs(p.current_node - self._last_replan_node) > nav_cfg.replan_node_thresh
        )
        if p.goal_node is not None and (
            p.current_path is None
            or self._replan_cooldown == 0
            or node_jumped
        ):
            p.current_path = self.nav_graph.find_path(
                p.current_node, p.goal_node,
                goal_feature=p.goal_feature_fused,
            )
            p.path_index = 0
            self._last_replan_node = p.current_node
            self._replan_cooldown = nav_cfg.replan_cooldown

        # ── Get the FIRST real action from the path ──
        # We only take ONE step, then re-localize next frame.
        action_name = self._get_first_path_action()
        action_source = "path" if action_name is not None else None

        # ── Fallback: action predictor ──
        if action_name is None and self.action_predictor is not None \
                and p.goal_feature_fused is not None:
            # Bug 2: if next hop is visual, aim at that node's feature
            waypoint_feat = self._get_next_waypoint_feat()
            if waypoint_feat is not None:
                goal_for_predictor = waypoint_feat
                action_source = "predictor(visual-wp)"
            elif p.goal_features:
                # Bug 5: use single best-matching goal view (not fused avg)
                sims = [float(np.dot(self._current_feat, gf))
                        for gf in p.goal_features]
                goal_for_predictor = p.goal_features[int(np.argmax(sims))]
                action_source = "predictor(goal-view)"
            else:
                goal_for_predictor = p.goal_feature_fused
                action_source = "predictor(fused)"
            idx = self.action_predictor.predict_action(
                self._current_feat, goal_for_predictor,
                list(self.action_history), self.device,
            )
            action_name = IDX_TO_ACTION[idx]

        if action_name is None:
            action_name = "FORWARD"
            action_source = "default"

        # ── Action smoothing: majority vote over recent window ──
        self._action_vote.append(action_name)
        smoothed = Counter(self._action_vote).most_common(1)[0][0]

        # ── Logging (every 10 steps) ──
        if self.step_count % 10 == 0:
            candidates, scores = self.localizer.query_top_k(
                self._current_feat, 3)
            top3_str = " ".join(f"{c}:{s:.3f}" for c, s in
                                zip(candidates[:3], scores[:3]))
            path_len = len(p.current_path) if p.current_path else 0
            goal_sim = max(
                (float(np.dot(self._current_feat, gf)) for gf in p.goal_features),
                default=0.0,
            )
            log.info(
                "[step %d] src=%-22s raw=%-8s out=%-8s "
                "node=%-5d goal=%-5d path=%-4d pidx=%-4d "
                "gsim=%.3f  loc=[%s]",
                self.step_count, action_source, action_name, smoothed,
                p.current_node if p.current_node is not None else -1,
                p.goal_node if p.goal_node is not None else -1,
                path_len, p.path_index,
                goal_sim, top3_str,
            )

        self.action_history.append(ACTION_TO_IDX.get(smoothed, 0))
        return _ACT_MAP[smoothed]

    def _get_first_path_action(self) -> str | None:
        """
        Return the next physical (temporal) action from the current path and
        advance ``path_index``.

        IMPORTANT: if the very next edge is a VISUAL shortcut we return None
        immediately — *without* incrementing path_index — so the action
        predictor can steer toward that node's feature instead of leaping to
        a temporal edge that belongs to a completely different physical
        location (the old "skip-visual" loop was the primary wall-hit cause).
        """
        p = self.planner
        if not p.current_path or len(p.current_path) < 2:
            return None
        if p.path_index >= len(p.current_path) - 1:
            return None

        a, b = p.current_path[p.path_index], p.current_path[p.path_index + 1]
        edge = self.nav_graph.get_edge_action(a, b)

        if edge == "VISUAL":
            # Let the action predictor navigate toward this node's feature.
            # Do NOT increment path_index yet — it advances when we replan
            # and localisation confirms we've reached ~b.
            return None

        p.path_index += 1
        return edge if edge in _ACT_MAP else None

    def _get_next_waypoint_feat(self) -> np.ndarray | None:
        """If next path edge is a visual shortcut, return that node's
        feature as a mini-goal for the action predictor (Bug 2 fix)."""
        p = self.planner
        if not p.current_path or p.path_index >= len(p.current_path) - 1:
            return None
        a = p.current_path[p.path_index]
        b = p.current_path[p.path_index + 1]
        if self.nav_graph.get_edge_action(a, b) == "VISUAL":
            return self.features[b]
        return None

    # ── Internals ────────────────────────────────────────────────────
    def _load_models(self) -> None:
        log.info("[1/5] Loading models...")
        self.feat_model = PlaceFeatureExtractor(freeze_backbone=True)
        if os.path.exists(paths.projection_head):
            self.feat_model.load_heads(paths.projection_head, self.device)
            log.info("      projection head loaded from %s", paths.projection_head)
        self.feat_model.to(self.device).eval()

        if os.path.exists(paths.action_predictor):
            self.action_predictor = ActionPredictor()
            self.action_predictor.load_state_dict(
                torch.load(paths.action_predictor, map_location=self.device,
                            weights_only=True))
            self.action_predictor.to(self.device).eval()
            log.info("      action predictor loaded from %s", paths.action_predictor)
        else:
            log.info("      no action predictor found — graph-only mode")

    def _load_trajectory(self) -> None:
        log.info("[2/5] Loading trajectory...")
        self.motion_frames, self.file_list = load_trajectory()
        log.info("      %d frames", len(self.motion_frames))

    def _extract_features(self) -> None:
        log.info("[3/5] Extracting features...")
        ch = compute_cache_hash()
        cache = os.path.join(paths.cache_dir, f"dinov2_features_{ch}.npy")
        if os.path.exists(cache):
            self.features = np.load(cache)
            if self.features.shape[0] == len(self.file_list):
                log.info("      cached %s", self.features.shape)
                return

        imgs = [cv2.imread(os.path.join(paths.image_dir, f)) for f in self.file_list]
        self.features = extract_features_batch(
            self.feat_model, imgs,
            batch_size=nav_cfg.feature_batch_size, device=self.device,
        )
        np.save(cache, self.features)

    def _build_navigation(self) -> None:
        log.info("[4/5] Building graph + index...")
        ch = compute_cache_hash()
        gcache = os.path.join(paths.cache_dir, f"nav_graph_{ch}.pkl")
        if os.path.exists(gcache):
            try:
                self.nav_graph = NavigationGraph.load(gcache, self.features)
                log.info("      graph loaded from cache")
            except Exception:
                self.nav_graph = None

        if self.nav_graph is None:
            self.nav_graph = NavigationGraph(self.features, self.motion_frames)
            self.nav_graph.save(gcache)

        self.localizer = Localizer(self.features)
        self.planner = GoalPlanner(self.localizer, self.features)

    def _setup_goal(self) -> None:
        log.info("[5/5] Setting up goal...")
        targets = self.get_target_images()
        if targets:
            self.planner.setup_goal(targets, self.feat_model, self.device)
            log.info("      goal_node=%s  candidates=%s",
                     self.planner.goal_node, self.planner.goal_candidates)


    def _show_targets(self) -> None:
        targets = self.get_target_images()
        if not targets:
            return
        img = cv2.vconcat([cv2.hconcat(targets[:2]), cv2.hconcat(targets[2:])])
        h, w = img.shape[:2]
        cv2.line(img, (w // 2, 0), (w // 2, h), (0, 0, 0), 2)
        cv2.line(img, (0, h // 2), (w, h // 2), (0, 0, 0), 2)
        F = cv2.FONT_HERSHEY_SIMPLEX
        for lbl, pos in [("Front", (10, 25)), ("Right", (w//2+10, 25)),
                         ("Back", (10, h//2+25)), ("Left", (w//2+10, h//2+25))]:
            cv2.putText(img, lbl, pos, F, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("Target Images", img)
        cv2.imwrite("target.jpg", img)
        cv2.waitKey(1)


# ── CLI ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    _fh = logging.FileHandler("vis_nav_player.log", mode="w")
    _fh.setFormatter(_fmt)
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(_fmt)
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(_fh)
    logging.root.addHandler(_sh)
    import vis_nav_game as vng

    logging.info(f"player.py → vis_nav_game {vng.core.__version__}")
    vng.play(the_player=KeyboardPlayerPyGame())
