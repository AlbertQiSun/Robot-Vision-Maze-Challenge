"""
Visual localiser: FAISS index + temporal feature averaging + Gaussian motion prior.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from vis_nav.config import nav_cfg as C


class Localizer:
    """Maintains a FAISS index and provides temporally-consistent localisation.

    Two-stage improvement over the original single-frame FAISS winner-take-all:

    1. **Temporal feature averaging** — the last ``localizer_feat_avg_window``
       feature vectors are averaged and re-normalised before the FAISS query.
       Consecutive frames share the dominant "place signature" but have
       independent per-frame noise; the mean cancels that noise, giving a
       cleaner query that generalises better across the exploration→competition
       domain gap (where single-frame scores drop to 0.22–0.51).

    2. **Gaussian motion-prior re-scoring** — instead of a hard binary window
       (accept / reject based on ``|node − prev| < window``), every FAISS
       candidate is multiplied by a Gaussian prior centred at the previous
       node with std ``localizer_motion_sigma``.  This:
         • Naturally prefers nearby candidates without completely discarding
           distant high-confidence ones.
         • Replaces the hard threshold that caused "snap-to-wrong-place" jumps
           whenever no candidate fell inside the narrow window.
         • The dead-reckoning fallback is retained as a last resort:  if the
           best raw FAISS score is below ``localizer_min_score`` *and* the
           motion-prior winner is far from the previous node, hold the
           previous estimate rather than leaping to a low-confidence match.
    """

    def __init__(self, features: np.ndarray) -> None:
        self.features = features
        self.faiss_index = None
        self.prev_nodes: deque[int] = deque(maxlen=5)

        # Rolling buffer of recent feature vectors for query averaging
        self._feat_buffer: deque[np.ndarray] = deque(
            maxlen=C.localizer_feat_avg_window
        )

        self._build_index()

    # ── FAISS ────────────────────────────────────────────────────────
    def _build_index(self) -> None:
        try:
            import faiss

            dim = self.features.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            feats = np.ascontiguousarray(self.features, dtype=np.float32)
            self.faiss_index.add(feats)
        except ImportError:
            self.faiss_index = None

    # ── Query ────────────────────────────────────────────────────────
    def localize(self, query_feat: np.ndarray, top_k: int = C.faiss_top_k) -> int:
        """Return the best-matching database node with motion-prior filtering."""

        # ── 1. Temporal feature averaging ────────────────────────────
        self._feat_buffer.append(query_feat)
        if len(self._feat_buffer) >= 2:
            stacked = np.stack(self._feat_buffer).astype(np.float32)
            avg = stacked.mean(axis=0)
            norm = float(np.linalg.norm(avg))
            query = (avg / norm if norm > 1e-9 else avg).astype(np.float32)
        else:
            query = query_feat

        # ── 2. Raw FAISS query — extra candidates for re-scoring ─────
        candidates, scores = self._raw_query(query, top_k * 2)
        best_raw_score = float(scores[0])

        # ── 3. Gaussian motion-prior re-scoring ──────────────────────
        if self.prev_nodes:
            prev = int(self.prev_nodes[-1])
            sigma = float(C.localizer_motion_sigma)

            best_combined = -1.0
            best_node = int(candidates[0])

            for node, score in zip(candidates, scores):
                dist = abs(int(node) - prev)
                prior = float(np.exp(-0.5 * (dist / sigma) ** 2))
                combined = float(score) * prior
                if combined > best_combined:
                    best_combined = combined
                    best_node = int(node)

            # Dead-reckoning fallback: if FAISS confidence is very low AND
            # the winner is a large jump, hold the previous estimate.
            if (
                best_raw_score < C.localizer_min_score
                and abs(best_node - prev) > C.localizer_temporal_window
            ):
                best_node = prev
        else:
            best_node = int(candidates[0])

        self.prev_nodes.append(best_node)
        return best_node

    def query_top_k(
        self, query_feat: np.ndarray, k: int
    ) -> tuple[list[int], list[float]]:
        """Raw top-k without temporal filtering (used for logging)."""
        return self._raw_query(query_feat, k)

    # ── Internals ────────────────────────────────────────────────────
    def _raw_query(
        self, feat: np.ndarray, k: int
    ) -> tuple[list[int], list[float]]:
        if self.faiss_index is not None:
            q = np.ascontiguousarray(feat.reshape(1, -1), dtype=np.float32)
            D, I = self.faiss_index.search(q, k)
            return I[0].tolist(), D[0].tolist()

        sims = self.features @ feat
        idx = np.argsort(sims)[-k:][::-1]
        return idx.tolist(), [float(sims[i]) for i in idx]
