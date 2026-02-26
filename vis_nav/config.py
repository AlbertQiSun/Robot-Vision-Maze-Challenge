"""
Centralised configuration for the entire vis_nav system.

Every tunable constant lives here — nothing is hard-coded in other
modules.  Import what you need:

    from vis_nav.config import PathCfg, FeatureCfg, NavCfg
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# ── Paths ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PathCfg:
    """All file / directory paths the system uses."""
    cache_dir: str = "cache"
    model_dir: str = "models"
    image_dir: str = "data/images"
    data_info: str = "data/data_info.json"
    texture_dir: str = "data/textures"

    projection_head: str = os.path.join("models", "projection_head.pth")
    projection_full: str = os.path.join("models", "projection_head_full.pth")
    action_predictor: str = os.path.join("models", "action_predictor.pth")
    training_log: str = os.path.join("models", "training_log.json")


# ── Backbone & Feature Extraction ───────────────────────────────────────
@dataclass(frozen=True)
class FeatureCfg:
    """DINOv2 backbone + projection hyper-parameters."""
    backbone_name: str = "dinov2_vitb14"
    backbone_dim: int = 768          # ViT-B/14 embed dim
    projection_hidden: int = 512
    projection_dim: int = 256        # final descriptor dim
    projection_dropout: float = 0.1

    input_size: int = 224            # DINOv2 input resolution
    gem_p_init: float = 3.0          # initial GeM power
    gem_eps: float = 1e-6

    imagenet_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, ...] = (0.229, 0.224, 0.225)


# ── Navigation Graph ───────────────────────────────────────────────────
@dataclass(frozen=True)
class GraphCfg:
    """Topological graph construction parameters."""
    temporal_fwd_weight: float = 1.0
    temporal_bwd_weight: float = 4.0    # moderate backtrack penalty (was 10.0)
    visual_edge_weight: float = 0.3     # prefer visual shortcuts over backtracking

    min_shortcut_gap: int = 30
    global_top_k: int = 500
    per_node_top_k: int = 3
    per_node_sim_threshold: float = 0.80
    heuristic_scale: float = 5.0


# ── Autonomous Navigation ──────────────────────────────────────────────
@dataclass(frozen=True)
class NavCfg:
    """Online navigation controller parameters."""
    subsample_rate: int = 1
    feature_batch_size: int = 64

    # FAISS
    faiss_top_k: int = 10

    # Re-localisation
    relocalize_interval: int = 5      # fast relocalization for carrot-on-a-stick tracking

    # Localizer — Gaussian motion-prior + temporal feature averaging
    localizer_motion_sigma: float = 12.0   # Gaussian prior std in nodes (~1–2 physical steps)
    localizer_feat_avg_window: int = 3     # frames to average before FAISS query
    localizer_min_score: float = 0.60      # below this AND large jump → hold previous node
    # kept for dead-reckoning fallback check only (no longer used as hard cutoff)
    localizer_temporal_window: int = 20    # max jump allowed for dead-reckoning fallback

    # Stuck detection — node-spread based (replaces feature-similarity)
    stuck_window: int = 20             # history length (nodes) — need 20 normal steps to fill
    stuck_node_spread: int = 15        # if max-min node spread < this over the window → stuck
    stuck_patience: int = 5            # consecutive detections before recovery (was 2, too eager)
    post_recovery_cooldown: int = 40   # normal steps to skip after recovery before re-evaluating

    # Check-in (calibrated to competition maze — exploration maze scores ~0.82+,
    #           competition maze only reaches 0.22–0.51 throughout navigation)
    checkin_sim_threshold: float = 0.52   # above this → +1 confidence (was 0.82, unreachable)
    checkin_min_sim: float = 0.35         # below this → hard reset, definitely not at goal
    checkin_confidence_needed: int = 5
    checkin_graph_dist: int = 3

    # Re-ranking
    use_reranking: bool = True
    rerank_top_k: int = 20

    # Action smoothing & re-plan throttling
    action_smooth_window: int = 3   # majority-vote window
    replan_cooldown: int = 5        # replan every 6 steps; localization flicker was causing
                                    # plan thrashing (was 1 → back-and-forth observed in logs)
    replan_node_thresh: int = 20    # only force-replan if node shifts > 20 frames


# ── Training ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TrainCfg:
    """Training hyper-parameters (projection head + action predictor)."""
    # Multi-Similarity Loss
    ms_alpha: float = 2.0
    ms_beta: float = 50.0
    ms_base: float = 0.5

    # PK Sampler
    pk_P: int = 16
    pk_K: int = 4

    # Phase A (frozen backbone)
    lr_heads: float = 5e-4
    epochs_phase_a: int = 50
    weight_decay: float = 1e-4
    cosine_T0: int = 10

    # Phase B (unfreeze last-N blocks)
    lr_backbone: float = 5e-6
    epochs_phase_b: int = 30
    unfreeze_blocks: int = 2

    # Dataset
    subsample_rate: int = 1           # use ALL frames (was 2)
    positive_range: int = 8           # wider "same place" window (was 5)
    synthetic_ratio: float = 0.15

    # Action predictor
    action_lr: float = 1e-3
    action_epochs: int = 50
    action_n_goals: int = 10
    action_min_gap: int = 50
    action_max_gap: int = 200


# ── Action mapping ─────────────────────────────────────────────────────
ACTION_NAMES = ("FORWARD", "LEFT", "RIGHT", "BACKWARD")
ACTION_TO_IDX = {name: idx for idx, name in enumerate(ACTION_NAMES)}
IDX_TO_ACTION = {idx: name for idx, name in enumerate(ACTION_NAMES)}
PURE_ACTIONS = set(ACTION_NAMES)

REVERSE_ACTION = {
    "FORWARD": "BACKWARD",
    "BACKWARD": "FORWARD",
    "LEFT": "RIGHT",
    "RIGHT": "LEFT",
}


# ── Convenience singletons ─────────────────────────────────────────────
paths = PathCfg()
feat_cfg = FeatureCfg()
graph_cfg = GraphCfg()
nav_cfg = NavCfg()
train_cfg = TrainCfg()
