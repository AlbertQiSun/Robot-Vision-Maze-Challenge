"""
Image transforms and batch feature-extraction helpers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2 as T

from vis_nav.config import feat_cfg as C


def _numpy_to_pil(img: np.ndarray) -> Image.Image:
    """Convert a numpy RGB array to a PIL Image (avoids broken T.ToPILImage)."""
    return Image.fromarray(img.astype(np.uint8))


# ── Transforms ──────────────────────────────────────────────────────────
def get_inference_transform():
    """Deterministic resize + normalise for inference."""
    return T.Compose([
        T.Lambda(_numpy_to_pil),
        T.Resize((C.input_size, C.input_size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=C.imagenet_mean, std=C.imagenet_std),
    ])


def get_train_transform():
    """Heavy augmentation for metric-learning training."""
    return T.Compose([
        T.Lambda(_numpy_to_pil),
        # Spatial
        T.RandomResizedCrop(C.input_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        T.RandomPerspective(distortion_scale=0.15, p=0.3),
        # Photometric
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        T.RandomGrayscale(p=0.05),
        # Convert to tensor first — GaussianBlur on PIL is broken with numpy>=1.26
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.GaussianBlur(kernel_size=7, sigma=(0.1, 3.0)),
        # Corruption
        T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        # Normalise
        T.Normalize(mean=C.imagenet_mean, std=C.imagenet_std),
    ])


# ── Batch helpers ───────────────────────────────────────────────────────
@torch.no_grad()
def extract_features_batch(
    model: "PlaceFeatureExtractor",  # noqa: F821 (forward ref)
    images: list[np.ndarray],
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Extract descriptors for a list of BGR numpy images.

    Returns ``(N, projection_dim)`` float32 array, L2-normalised.
    """
    from vis_nav.utils import get_device

    device = device or get_device()
    model.eval()
    model.to(device)
    transform = get_inference_transform()

    all_feats: list[np.ndarray] = []
    n_batches = (len(images) + batch_size - 1) // batch_size
    iterator: range | object = range(0, len(images), batch_size)

    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="Extracting features", total=n_batches)

    for start in iterator:
        tensors = []
        for img in images[start : start + batch_size]:
            if img is None or len(img.shape) < 3:
                tensors.append(torch.zeros(3, C.input_size, C.input_size))
            else:
                tensors.append(transform(img[:, :, ::-1].copy()))  # BGR → RGB
        batch = torch.stack(tensors).to(device)
        # np.array(copy=True) launders the torch-created ndarray so that
        # downstream consumers (FAISS, torch.from_numpy, etc.) accept it.
        all_feats.append(np.array(model(batch).cpu().numpy(), copy=True))

    return np.vstack(all_feats).astype(np.float32)


@torch.no_grad()
def extract_single_feature(
    model: "PlaceFeatureExtractor",  # noqa: F821
    image: np.ndarray,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Extract descriptor for one BGR image → ``(projection_dim,)``."""
    from vis_nav.utils import get_device

    device = device or get_device()
    model.eval()

    if image is None or len(image.shape) < 3:
        return np.zeros(C.projection_dim, dtype=np.float32)

    transform = get_inference_transform()
    tensor = transform(image[:, :, ::-1].copy()).unsqueeze(0).to(device)
    # np.array(copy=True) launders the torch-created ndarray (see utils.to_numpy)
    raw = model(tensor).cpu().numpy()
    return np.array(raw, copy=True).flatten().astype(np.float32)
