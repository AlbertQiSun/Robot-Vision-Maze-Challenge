"""
DINOv2 ViT-B/14  +  GeM Pooling  +  Projection MLP.

Produces 256-dim L2-normalised place descriptors from RGB images.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vis_nav.config import feat_cfg as C
from vis_nav.utils import get_device


# ── GeM Pooling ─────────────────────────────────────────────────────────
class GeMPooling(nn.Module):
    """Generalised Mean Pooling over patch tokens.

    ``pool(X) = (mean(x_i^p))^(1/p)``

    *p* is a learnable scalar (initialised to 3).
    """

    def __init__(self, p: float = C.gem_p_init, eps: float = C.gem_eps):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) → (B, D)"""
        x = x.clamp(min=self.eps)
        return x.pow(self.p).mean(dim=1).pow(1.0 / self.p)


# ── Projection MLP ──────────────────────────────────────────────────────
class ProjectionMLP(nn.Module):
    """``backbone_dim → hidden → output_dim``, L2-normalised."""

    def __init__(
        self,
        input_dim: int = C.backbone_dim,
        hidden_dim: int = C.projection_hidden,
        output_dim: int = C.projection_dim,
        dropout: float = C.projection_dropout,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, input_dim) → (B, output_dim) L2-normalised."""
        return F.normalize(self.net(x), p=2, dim=-1)


# ── Full Feature Extractor ──────────────────────────────────────────────
class PlaceFeatureExtractor(nn.Module):
    """
    End-to-end place descriptor.

    ``Image (224×224) → DINOv2 patch tokens → GeM → Projection → (B, 256)``
    """

    def __init__(
        self,
        backbone_name: str = C.backbone_name,
        projection_dim: int = C.projection_dim,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", backbone_name, pretrained=True,
        )
        self.backbone_dim = self.backbone.embed_dim

        if freeze_backbone:
            self.freeze_backbone()

        self.gem = GeMPooling()
        self.projection = ProjectionMLP(
            input_dim=self.backbone_dim,
            output_dim=projection_dim,
        )

    # ── Freeze / unfreeze helpers ────────────────────────────────────
    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def unfreeze_last_n_blocks(self, n: int = 2) -> None:
        self.freeze_backbone()
        for block in self.backbone.blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        if hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

    # ── Forward ──────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) → (B, projection_dim) L2-normalised."""
        bb_grad = self.training and any(
            p.requires_grad for p in self.backbone.parameters()
        )
        with torch.set_grad_enabled(bb_grad):
            tokens = self.backbone.forward_features(x)["x_norm_patchtokens"]
        return self.projection(self.gem(tokens))

    def extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.backbone.forward_features(x)["x_norm_patchtokens"]

    # ── Serialisation ────────────────────────────────────────────────
    def save_heads(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "gem": self.gem.state_dict(),
                "projection": self.projection.state_dict(),
                "backbone_dim": self.backbone_dim,
                "projection_dim": self.projection.net[-1].out_features,
            },
            path,
        )

    def load_heads(self, path: str, device: Optional[torch.device] = None) -> None:
        device = device or get_device()
        state = torch.load(path, map_location=device, weights_only=True)
        self.gem.load_state_dict(state["gem"])
        self.projection.load_state_dict(state["projection"])

    def save_full(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "backbone": self.backbone.state_dict(),
                "gem": self.gem.state_dict(),
                "projection": self.projection.state_dict(),
                "backbone_dim": self.backbone_dim,
                "projection_dim": self.projection.net[-1].out_features,
            },
            path,
        )

    def load_full(self, path: str, device: Optional[torch.device] = None) -> None:
        device = device or get_device()
        state = torch.load(path, map_location=device, weights_only=True)
        self.backbone.load_state_dict(state["backbone"])
        self.gem.load_state_dict(state["gem"])
        self.projection.load_state_dict(state["projection"])
