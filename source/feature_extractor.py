"""
Feature Extractor: DINOv2 ViT-B/14 + GeM Pooling + Projection MLP.

Produces 256-dim L2-normalized place descriptors from RGB images.
Works on CUDA, MPS (Apple Silicon), and CPU.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BACKBONE_DIM = 768          # DINOv2 ViT-B/14 output dimension
PROJECTION_DIM = 256        # final descriptor dimension
INPUT_SIZE = 224             # DINOv2 input resolution


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# GeM Pooling
# ---------------------------------------------------------------------------
class GeMPooling(nn.Module):
    """Generalized Mean Pooling.

    pool(X) = (1/N * sum(x_i^p))^(1/p)

    When p=1, reduces to average pooling.
    When p→∞, approaches max pooling.
    Learnable p allows the network to find the optimal emphasis.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) — patch tokens from ViT
        Returns:
            (B, D) — pooled global descriptor
        """
        # Clamp to avoid numerical issues
        x = x.clamp(min=self.eps)
        # GeM: (mean of x^p)^(1/p)
        x_pow = x.pow(self.p)
        pooled = x_pow.mean(dim=1)  # (B, D)
        return pooled.pow(1.0 / self.p)


# ---------------------------------------------------------------------------
# Projection MLP
# ---------------------------------------------------------------------------
class ProjectionMLP(nn.Module):
    """Two-layer projection: backbone_dim → hidden → output_dim, L2-normalized."""

    def __init__(self, input_dim: int = BACKBONE_DIM,
                 hidden_dim: int = 512,
                 output_dim: int = PROJECTION_DIM,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim)
        Returns:
            (B, output_dim) — L2-normalized
        """
        x = self.net(x)
        return F.normalize(x, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Full Feature Extractor
# ---------------------------------------------------------------------------
class PlaceFeatureExtractor(nn.Module):
    """
    Complete place recognition feature extractor.

    Pipeline:
        Image (224×224) → DINOv2 ViT-B/14 → patch tokens (B, 256, 768)
        → GeM pooling → (B, 768)
        → Projection MLP → (B, 256) L2-normalized
    """

    def __init__(self, backbone_name: str = "dinov2_vitb14",
                 projection_dim: int = PROJECTION_DIM,
                 freeze_backbone: bool = True):
        super().__init__()

        # Load DINOv2 backbone from torch.hub
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", backbone_name, pretrained=True
        )
        self.backbone_dim = self.backbone.embed_dim  # 768 for ViT-B

        # Freeze backbone initially
        if freeze_backbone:
            self.freeze_backbone()

        # GeM pooling over patch tokens
        self.gem = GeMPooling(p=3.0)

        # Projection head
        self.projection = ProjectionMLP(
            input_dim=self.backbone_dim,
            hidden_dim=512,
            output_dim=projection_dim,
        )

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def unfreeze_last_n_blocks(self, n: int = 2):
        """Unfreeze the last N transformer blocks for fine-tuning."""
        # First freeze everything
        self.freeze_backbone()
        # Then unfreeze last N blocks
        for block in self.backbone.blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        # Also unfreeze norm
        if hasattr(self.backbone, 'norm'):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) — normalized RGB images
        Returns:
            (B, projection_dim) — L2-normalized place descriptors
        """
        # Get intermediate features (patch tokens without CLS)
        with torch.set_grad_enabled(self.training and any(
                p.requires_grad for p in self.backbone.parameters())):
            features = self.backbone.forward_features(x)
            patch_tokens = features["x_norm_patchtokens"]  # (B, N_patches, 768)

        # GeM pool patch tokens → global descriptor
        global_desc = self.gem(patch_tokens)  # (B, 768)

        # Project to compact descriptor
        descriptor = self.projection(global_desc)  # (B, 256) L2-normalized

        return descriptor

    def extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Extract raw patch tokens (for potential local feature use)."""
        with torch.no_grad():
            features = self.backbone.forward_features(x)
            return features["x_norm_patchtokens"]

    def save_heads(self, path: str):
        """Save only the trainable heads (GeM + Projection)."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        state = {
            "gem": self.gem.state_dict(),
            "projection": self.projection.state_dict(),
            "backbone_dim": self.backbone_dim,
            "projection_dim": self.projection.net[-1].out_features,
        }
        torch.save(state, path)
        print(f"Saved heads to {path}")

    def load_heads(self, path: str, device: Optional[torch.device] = None):
        """Load trained heads (GeM + Projection)."""
        if device is None:
            device = get_device()
        state = torch.load(path, map_location=device, weights_only=True)
        self.gem.load_state_dict(state["gem"])
        self.projection.load_state_dict(state["projection"])
        print(f"Loaded heads from {path} (dim={state['projection_dim']})")

    def save_full(self, path: str):
        """Save full model (backbone + heads) — used after fine-tuning."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        state = {
            "backbone": self.backbone.state_dict(),
            "gem": self.gem.state_dict(),
            "projection": self.projection.state_dict(),
            "backbone_dim": self.backbone_dim,
            "projection_dim": self.projection.net[-1].out_features,
        }
        torch.save(state, path)
        print(f"Saved full model to {path}")

    def load_full(self, path: str, device: Optional[torch.device] = None):
        """Load full model (backbone + heads)."""
        if device is None:
            device = get_device()
        state = torch.load(path, map_location=device, weights_only=True)
        self.backbone.load_state_dict(state["backbone"])
        self.gem.load_state_dict(state["gem"])
        self.projection.load_state_dict(state["projection"])
        print(f"Loaded full model from {path}")


# ---------------------------------------------------------------------------
# Image Preprocessing
# ---------------------------------------------------------------------------
def get_inference_transform() -> T.Compose:
    """Transform for inference / feature extraction."""
    return T.Compose([
        T.ToPILImage(),
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_train_transform() -> T.Compose:
    """Heavy augmentation transform for training."""
    return T.Compose([
        T.ToPILImage(),
        # Spatial
        T.RandomResizedCrop(INPUT_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        T.RandomPerspective(distortion_scale=0.15, p=0.3),
        # Photometric
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        T.RandomGrayscale(p=0.05),
        T.GaussianBlur(kernel_size=7, sigma=(0.1, 3.0)),
        T.ToTensor(),
        # Corruption
        T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        # Normalize
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Action Predictor MLP
# ---------------------------------------------------------------------------
class ActionPredictor(nn.Module):
    """
    Small MLP that predicts navigation action from current + goal descriptors.

    Input:  current_feature (256) + goal_feature (256) + last_3_actions (12) = 524
    Output: P(FORWARD), P(LEFT), P(RIGHT), P(BACKWARD) = 4
    """

    def __init__(self, descriptor_dim: int = PROJECTION_DIM,
                 n_actions: int = 4, history_len: int = 3):
        super().__init__()
        input_dim = descriptor_dim * 2 + n_actions * history_len  # 256+256+12=524
        self.n_actions = n_actions
        self.history_len = history_len

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_actions),
        )

    def forward(self, current_feat: torch.Tensor,
                goal_feat: torch.Tensor,
                action_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            current_feat: (B, 256) — current place descriptor
            goal_feat: (B, 256) — goal place descriptor
            action_history: (B, 12) — one-hot encoded last 3 actions
        Returns:
            (B, 4) — action logits
        """
        x = torch.cat([current_feat, goal_feat, action_history], dim=-1)
        return self.net(x)

    def predict_action(self, current_feat: np.ndarray,
                       goal_feat: np.ndarray,
                       action_history: list[int],
                       device: Optional[torch.device] = None) -> int:
        """Predict best action for a single observation."""
        if device is None:
            device = get_device()

        self.eval()
        with torch.no_grad():
            cf = torch.tensor(current_feat, dtype=torch.float32).unsqueeze(0).to(device)
            gf = torch.tensor(goal_feat, dtype=torch.float32).unsqueeze(0).to(device)

            # Encode action history as one-hot
            ah = torch.zeros(1, self.n_actions * self.history_len, device=device)
            for i, a in enumerate(action_history[-self.history_len:]):
                if 0 <= a < self.n_actions:
                    ah[0, i * self.n_actions + a] = 1.0

            logits = self.forward(cf, gf, ah)
            return int(logits.argmax(dim=-1).item())


# ---------------------------------------------------------------------------
# Batch Feature Extraction Helper
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_features_batch(
    model: PlaceFeatureExtractor,
    images: list[np.ndarray],
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Extract features for a list of BGR images.

    Args:
        model: PlaceFeatureExtractor (should be on device already)
        images: list of BGR numpy images (any size)
        batch_size: inference batch size
        device: torch device
        show_progress: show tqdm progress bar

    Returns:
        (N, projection_dim) float32 numpy array, L2-normalized
    """
    if device is None:
        device = get_device()

    model.eval()
    model.to(device)
    transform = get_inference_transform()

    all_features = []
    iterator = range(0, len(images), batch_size)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="Extracting features",
                        total=(len(images) + batch_size - 1) // batch_size)

    for start in iterator:
        batch_imgs = images[start:start + batch_size]
        tensors = []
        for img in batch_imgs:
            if img is None or len(img.shape) < 3:
                # Zero tensor for invalid images
                tensors.append(torch.zeros(3, INPUT_SIZE, INPUT_SIZE))
            else:
                # Convert BGR → RGB
                rgb = img[:, :, ::-1].copy()
                tensors.append(transform(rgb))
        batch = torch.stack(tensors).to(device)
        features = model(batch)  # (B, 256)
        all_features.append(features.cpu().numpy())

    return np.vstack(all_features).astype(np.float32)


@torch.no_grad()
def extract_single_feature(
    model: PlaceFeatureExtractor,
    image: np.ndarray,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Extract feature for a single BGR image. Returns (256,) float32."""
    if device is None:
        device = get_device()

    model.eval()
    transform = get_inference_transform()

    if image is None or len(image.shape) < 3:
        return np.zeros(PROJECTION_DIM, dtype=np.float32)

    rgb = image[:, :, ::-1].copy()
    tensor = transform(rgb).unsqueeze(0).to(device)
    feat = model(tensor)
    return feat.cpu().numpy().flatten().astype(np.float32)
