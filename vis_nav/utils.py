"""
Shared utilities: device detection, caching, trajectory I/O.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from vis_nav.config import paths, nav_cfg, PURE_ACTIONS


# ── Numpy / Torch compatibility ──────────────────────────────────────────
def to_numpy(x) -> np.ndarray:
    """
    Safely convert *x* to a genuine ``numpy.ndarray``.

    PyTorch 2.10 + numpy ≥1.26 have a type-identity mismatch:
    ``Tensor.numpy()`` returns an array whose ``type()`` is not recognised
    by ``torch.from_numpy``, FAISS ``swig_ptr``, etc.  Passing through
    ``np.array(..., copy=True)`` produces a "clean" array that every
    library accepts.
    """
    if isinstance(x, np.ndarray):
        return x                       # already a real numpy array
    # Covers torch-created arrays and any other array-like
    return np.array(x, copy=True)


# ── Device ──────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    """Return the best available PyTorch device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Cache hash ──────────────────────────────────────────────────────────
def compute_cache_hash(
    data_info_path: str = paths.data_info,
    subsample_rate: int = nav_cfg.subsample_rate,
    model_path: str = paths.projection_head,
) -> str:
    """Deterministic short hash that keys disk caches."""
    h = hashlib.md5()
    h.update(data_info_path.encode())
    h.update(str(subsample_rate).encode())
    if os.path.exists(model_path):
        h.update(str(os.path.getmtime(model_path)).encode())
    if os.path.exists(data_info_path):
        h.update(str(os.path.getmtime(data_info_path)).encode())
    return h.hexdigest()[:12]


# ── Trajectory loading ─────────────────────────────────────────────────
def load_trajectory(
    data_info_path: str = paths.data_info,
    subsample_rate: int = nav_cfg.subsample_rate,
) -> tuple[list[dict], list[str]]:
    """
    Load and filter exploration trajectory from ``data_info.json``.

    Returns
    -------
    motion_frames : list[dict]
        Each dict has keys ``step``, ``image``, ``action``.
    file_list : list[str]
        Image filenames corresponding to *motion_frames*.
    """
    if not os.path.exists(data_info_path):
        raise FileNotFoundError(f"No data_info.json at {data_info_path}")

    with open(data_info_path) as f:
        raw = json.load(f)

    all_motion = [
        {"step": d["step"], "image": d["image"], "action": d["action"][0]}
        for d in raw
        if len(d["action"]) == 1 and d["action"][0] in PURE_ACTIONS
    ]

    motion_frames = all_motion[::subsample_rate]
    file_list = [m["image"] for m in motion_frames]

    return motion_frames, file_list


# ── Directory helpers ───────────────────────────────────────────────────
def ensure_dirs() -> None:
    """Create standard output directories if they don't exist."""
    os.makedirs(paths.cache_dir, exist_ok=True)
    os.makedirs(paths.model_dir, exist_ok=True)
