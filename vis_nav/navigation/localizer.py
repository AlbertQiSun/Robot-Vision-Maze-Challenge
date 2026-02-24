"""
Visual localiser: FAISS index + temporal consistency filter.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from vis_nav.config import nav_cfg as C


class Localizer:
    """Maintains a FAISS index and provides temporally-consistent localisation."""

    def __init__(self, features: np.ndarray):
        self.features = features
        self.faiss_index = None
        self.prev_nodes: deque[int] = deque(maxlen=5)

        self._build_index()

    # ── FAISS ────────────────────────────────────────────────────────
    def _build_index(self) -> None:
        try:
            import faiss

            dim = self.features.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.features.astype(np.float32))
        except ImportError:
            self.faiss_index = None

    # ── Query ────────────────────────────────────────────────────────
    def localize(self, query_feat: np.ndarray, top_k: int = C.faiss_top_k) -> int:
        """
        Return the database node that best matches *query_feat*,
        with temporal consistency filtering.
        """
        candidates, scores = self._raw_query(query_feat, top_k)

        best_node = candidates[0]
        best_score = scores[0]

        if self.prev_nodes:
            prev = self.prev_nodes[-1]
            for node, score in zip(candidates, scores):
                if abs(node - prev) < 20 and score > best_score * 0.95:
                    best_node = node
                    best_score = score
                    break

        self.prev_nodes.append(best_node)
        return best_node

    def query_top_k(
        self, query_feat: np.ndarray, k: int,
    ) -> tuple[list[int], list[float]]:
        """Raw top-k without temporal filtering."""
        return self._raw_query(query_feat, k)

    # ── internals ────────────────────────────────────────────────────
    def _raw_query(
        self, feat: np.ndarray, k: int,
    ) -> tuple[list[int], list[float]]:
        if self.faiss_index is not None:
            D, I = self.faiss_index.search(
                feat.reshape(1, -1).astype(np.float32), k,
            )
            return I[0].tolist(), D[0].tolist()

        sims = self.features @ feat
        idx = np.argsort(sims)[-k:][::-1]
        return idx.tolist(), [float(sims[i]) for i in idx]
