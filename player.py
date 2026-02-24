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
import time
from collections import deque

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
        self.action_history: deque = deque(maxlen=3)

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
        print(f"Camera K = {self.get_camera_intrinsic_matrix()}")

    # ── Pipeline build ───────────────────────────────────────────────
    def pre_navigation(self) -> None:
        super().pre_navigation()
        ensure_dirs()
        print("\n" + "=" * 60)
        print("PRE-NAVIGATION: building autonomous pipeline")
        print("=" * 60)
        t0 = time.time()
        try:
            self._load_models()
            self._load_trajectory()
            self._extract_features()
            self._build_navigation()
            self._setup_goal()
            self.autonomous = True
            print(f"\nReady in {time.time()-t0:.1f}s  "
                  f"({len(self.features)} nodes, "
                  f"{self.nav_graph.G.number_of_edges()} edges, "
                  f"goal={self.planner.goal_node})")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[WARN] Autonomous init failed: {exc}  → keyboard mode")
            self.autonomous = False
        print("=" * 60)

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
        feat = extract_single_feature(self.feat_model, self.fpv, self.device)

        # 1. check-in?
        if self.planner.should_checkin(feat, self.nav_graph):
            print(f"[CHECKIN] step {self.step_count}")
            return Action.CHECKIN

        # 2. stuck?
        recovery = self.planner.check_stuck()
        if recovery is not None:
            print(f"[RECOVERY] step {self.step_count} → {recovery}")
            return _ACT_MAP[recovery]

        # 3. re-plan?
        p = self.planner
        if (p.current_path is None
                or p.path_index >= len(p.current_path) - 1
                or self.step_count % nav_cfg.relocalize_interval == 0):
            self.planner.localize_and_plan(feat, self.nav_graph)

        # 4. follow path
        action = self._next_action()
        name = _ACT_NAMES.get(action, "FORWARD")
        if name in ACTION_TO_IDX:
            self.action_history.append(ACTION_TO_IDX[name])
        return action

    # ── Internals ────────────────────────────────────────────────────
    def _load_models(self) -> None:
        print("[1/5] Loading models...")
        self.feat_model = PlaceFeatureExtractor(freeze_backbone=True)
        if os.path.exists(paths.projection_head):
            self.feat_model.load_heads(paths.projection_head, self.device)
        self.feat_model.to(self.device).eval()

        if os.path.exists(paths.action_predictor):
            self.action_predictor = ActionPredictor()
            self.action_predictor.load_state_dict(
                torch.load(paths.action_predictor, map_location=self.device,
                            weights_only=True))
            self.action_predictor.to(self.device).eval()

    def _load_trajectory(self) -> None:
        print("[2/5] Loading trajectory...")
        self.motion_frames, self.file_list = load_trajectory()
        print(f"      {len(self.motion_frames)} frames")

    def _extract_features(self) -> None:
        print("[3/5] Extracting features...")
        ch = compute_cache_hash()
        cache = os.path.join(paths.cache_dir, f"dinov2_features_{ch}.npy")
        if os.path.exists(cache):
            self.features = np.load(cache)
            if self.features.shape[0] == len(self.file_list):
                print(f"      cached {self.features.shape}")
                return

        imgs = [cv2.imread(os.path.join(paths.image_dir, f)) for f in self.file_list]
        self.features = extract_features_batch(
            self.feat_model, imgs,
            batch_size=nav_cfg.feature_batch_size, device=self.device,
        )
        np.save(cache, self.features)

    def _build_navigation(self) -> None:
        print("[4/5] Building graph + index...")
        ch = compute_cache_hash()
        gcache = os.path.join(paths.cache_dir, f"nav_graph_{ch}.pkl")
        if os.path.exists(gcache):
            try:
                self.nav_graph = NavigationGraph.load(gcache, self.features)
            except Exception:
                self.nav_graph = None

        if self.nav_graph is None:
            self.nav_graph = NavigationGraph(self.features, self.motion_frames)
            self.nav_graph.save(gcache)

        self.localizer = Localizer(self.features)
        self.planner = GoalPlanner(self.localizer, self.features)

    def _setup_goal(self) -> None:
        print("[5/5] Setting up goal...")
        targets = self.get_target_images()
        if targets:
            self.planner.setup_goal(targets, self.feat_model, self.device)

    def _next_action(self) -> Action:
        p = self.planner
        if p.current_path is None or p.path_index >= len(p.current_path) - 1:
            return self._fallback()
        a, b = p.current_path[p.path_index], p.current_path[p.path_index + 1]
        p.path_index += 1
        edge = self.nav_graph.get_edge_action(a, b)
        if edge == "VISUAL":
            return self._next_action() if p.path_index < len(p.current_path) - 1 \
                else self._fallback()
        return _ACT_MAP.get(edge, self._fallback())

    def _fallback(self) -> Action:
        if self.action_predictor is not None and self.planner.goal_feature_fused is not None:
            feat = extract_single_feature(self.feat_model, self.fpv, self.device)
            idx = self.action_predictor.predict_action(
                feat, self.planner.goal_feature_fused,
                list(self.action_history), self.device,
            )
            return _ACT_MAP[IDX_TO_ACTION[idx]]
        r = random.random()
        if r < 0.5:   return Action.FORWARD
        if r < 0.7:   return Action.LEFT
        if r < 0.9:   return Action.RIGHT
        return Action.BACKWARD

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
    logging.basicConfig(
        filename="vis_nav_player.log", filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    import vis_nav_game as vng

    logging.info(f"player.py → vis_nav_game {vng.core.__version__}")
    vng.play(the_player=KeyboardPlayerPyGame())
