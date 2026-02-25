"""
Learned action predictor — primary navigation controller.

Uses a Siamese architecture to encode current and goal features,
computes their relational difference, and predicts the next action.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vis_nav.config import feat_cfg
from vis_nav.utils import get_device


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class ActionPredictor(nn.Module):
    """
    Siamese Relational Action Predictor.

    1. Encodes current and goal features through a shared Siamese network.
    2. Computes relational features (difference and product).
    3. Fuses with action history.
    4. Decodes into action probabilities using Residual blocks.
    """

    def __init__(
        self,
        descriptor_dim: int = feat_cfg.projection_dim,
        n_actions: int = 4,
        history_len: int = 3,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.history_len = history_len

        # ── Siamese Encoder (Shared Weights) ──
        self.encoder = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ── History Encoder ──
        self.history_enc = nn.Sequential(
            nn.Linear(n_actions * history_len, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # ── Relational Decoder ──
        # Input: current (hidden), goal (hidden), diff (hidden), prod (hidden), hist (64)
        fuse_dim = hidden_dim * 4 + 64
        
        self.decoder = nn.Sequential(
            nn.Linear(fuse_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_actions)
        )

    def forward(
        self,
        current_feat: torch.Tensor,
        goal_feat: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Siamese encoding
        c_enc = self.encoder(current_feat)
        g_enc = self.encoder(goal_feat)

        # 2. Relational features
        diff = c_enc - g_enc
        prod = c_enc * g_enc

        # 3. History encoding
        h_enc = self.history_enc(action_history)

        # 4. Fusion and decoding
        fused = torch.cat([c_enc, g_enc, diff, prod, h_enc], dim=-1)
        return self.decoder(fused)

    # ── Convenient single-sample inference ───────────────────────────
    @torch.no_grad()
    def predict_action(
        self,
        current_feat: np.ndarray,
        goal_feat: np.ndarray,
        action_history: list[int],
        device: Optional[torch.device] = None,
    ) -> int:
        device = device or get_device()
        self.eval()

        cf = torch.tensor(
            np.array(current_feat, dtype=np.float32, copy=True)
        ).unsqueeze(0).to(device)
        gf = torch.tensor(
            np.array(goal_feat, dtype=np.float32, copy=True)
        ).unsqueeze(0).to(device)

        ah = torch.zeros(1, self.n_actions * self.history_len, device=device)
        for i, a in enumerate(action_history[-self.history_len:]):
            if 0 <= a < self.n_actions:
                ah[0, i * self.n_actions + a] = 1.0

        logits = self.forward(cf, gf, ah)
        return int(logits.argmax(dim=-1).item())
