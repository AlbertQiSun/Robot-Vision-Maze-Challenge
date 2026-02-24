"""
Learned action predictor — fallback controller when graph planning fails.

``(current_descriptor, goal_descriptor, action_history) → action``
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from vis_nav.config import feat_cfg
from vis_nav.utils import get_device


class ActionPredictor(nn.Module):
    """
    Small MLP that maps (current_feat ‖ goal_feat ‖ action_history) → action.

    Input dims : 256 + 256 + 12 = 524  (default)
    Output dims: 4  (FORWARD, LEFT, RIGHT, BACKWARD)
    """

    def __init__(
        self,
        descriptor_dim: int = feat_cfg.projection_dim,
        n_actions: int = 4,
        history_len: int = 3,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.history_len = history_len
        input_dim = descriptor_dim * 2 + n_actions * history_len

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_actions),
        )

    def forward(
        self,
        current_feat: torch.Tensor,
        goal_feat: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        """(B, D), (B, D), (B, n_actions*history_len) → (B, n_actions)."""
        return self.net(
            torch.cat([current_feat, goal_feat, action_history], dim=-1)
        )

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

        cf = torch.tensor(current_feat, dtype=torch.float32).unsqueeze(0).to(device)
        gf = torch.tensor(goal_feat, dtype=torch.float32).unsqueeze(0).to(device)

        ah = torch.zeros(1, self.n_actions * self.history_len, device=device)
        for i, a in enumerate(action_history[-self.history_len :]):
            if 0 <= a < self.n_actions:
                ah[0, i * self.n_actions + a] = 1.0

        return int(self.forward(cf, gf, ah).argmax(dim=-1).item())
