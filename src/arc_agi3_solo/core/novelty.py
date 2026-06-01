"""Frame embedding + online clustering for Phase 2.

Cluster 2 of the failure-mode taxonomy (re86, tr87, wa30) has 10k+ unique
frame hashes but presumably ~100 meaningfully-distinct game states; pure-hash
BFS can't fit a level-completing path in 5 min. The fix: embed each frame
through a small random-init CNN, cluster visually-similar embeddings, and
use cluster IDs as the graph explorer's node identities. The state space
collapses, BFS becomes tractable.

Random-init CNN (no training) is a known trick — see Burda et al. 2018's
Random Network Distillation. Random features tend to preserve enough of
the input structure that visually-similar inputs land near each other in
embedding space, while being cheap and deterministic.

Pure numpy (no torch dependency) — keeps the offline-eligibility surface
clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def _conv2d(x: np.ndarray, w: np.ndarray, stride: int = 1) -> np.ndarray:
    """Naive 2D conv via im2col + matmul. x: (Cin, H, W). w: (Cout, Cin, kH, kW).

    Returns (Cout, Hout, Wout). Hout = (H - kH) / stride + 1.
    """
    cin, h, ww = x.shape
    cout, _, kh, kw = w.shape
    out_h = (h - kh) // stride + 1
    out_w = (ww - kw) // stride + 1
    # Im2col
    cols = np.zeros((cin * kh * kw, out_h * out_w), dtype=x.dtype)
    idx = 0
    for i in range(0, h - kh + 1, stride):
        for j in range(0, ww - kw + 1, stride):
            patch = x[:, i:i + kh, j:j + kw].reshape(-1)
            cols[:, idx] = patch
            idx += 1
    return (w.reshape(cout, -1) @ cols).reshape(cout, out_h, out_w)


class RandomCNNEmbedder:
    """Frozen random-init CNN: (64, 64) int frame -> 32-D L2-normalized embedding.

    Architecture: one-hot encode 16 colors -> Conv(16->16, k=5, s=2) -> ReLU
    -> Conv(16->32, k=5, s=2) -> ReLU -> global mean pool -> L2-normalize.

    Output ~14x14 spatial after the second conv (64 -> 30 -> 13 with k=5/s=2),
    mean-pooled to a single 32-D vector.

    Seeded for reproducibility — same `seed` always yields the same weights.
    """

    EMBED_DIM = 32

    def __init__(self, seed: int = 0xA3A3) -> None:
        rng = np.random.default_rng(seed)
        # He init for ReLU
        self.w1 = rng.normal(0, np.sqrt(2.0 / (16 * 5 * 5)), size=(16, 16, 5, 5)).astype(np.float32)
        self.w2 = rng.normal(0, np.sqrt(2.0 / (16 * 5 * 5)), size=(32, 16, 5, 5)).astype(np.float32)

    def embed(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 2 or frame.shape != (64, 64):
            # Pad/crop to (64, 64). Frames are always 64x64 in ARC-AGI-3 but
            # defensive.
            f = np.zeros((64, 64), dtype=np.uint8)
            h, w = min(frame.shape[0], 64), min(frame.shape[1], 64)
            f[:h, :w] = np.asarray(frame, dtype=np.uint8)[:h, :w]
            frame = f
        # One-hot 16 colors: (16, 64, 64)
        x = np.zeros((16, 64, 64), dtype=np.float32)
        for c in range(16):
            x[c] = (frame == c).astype(np.float32)
        # Conv1 -> ReLU
        x = _conv2d(x, self.w1, stride=2)  # (16, 30, 30)
        np.maximum(x, 0, out=x)
        # Conv2 -> ReLU
        x = _conv2d(x, self.w2, stride=2)  # (32, 13, 13)
        np.maximum(x, 0, out=x)
        # Global mean pool -> (32,)
        emb = x.mean(axis=(1, 2))
        # L2 normalize so distances are cosine-like
        n = np.linalg.norm(emb)
        if n > 1e-8:
            emb = emb / n
        return emb.astype(np.float32)


@dataclass
class EmbeddingClusterMap:
    """Online incremental clustering of frame embeddings.

    Each unique frame hash maps to a stable cluster ID. New hashes either
    join an existing cluster (if their embedding is within `eps` cosine
    distance) or open a new one. Same hash always returns the same cluster
    even if `eps` later admits a closer cluster — this stability is
    critical for the graph explorer.

    Cosine distance = 1 - cos_sim. With L2-normalized embeddings this is
    `0.5 * ||a - b||^2`. We work directly with L2 distance (which has the
    same ordering) for speed.
    """

    eps: float = 0.4  # max L2 distance for "same cluster"
    centers: List[np.ndarray] = field(default_factory=list)
    hash_to_cluster: Dict[str, int] = field(default_factory=dict)
    _next_id: int = 0

    def assign(self, frame_hash: str, embedding: np.ndarray) -> int:
        cached = self.hash_to_cluster.get(frame_hash)
        if cached is not None:
            return cached
        if self.centers:
            stacked = np.stack(self.centers)
            d = np.linalg.norm(stacked - embedding[None, :], axis=1)
            nearest = int(np.argmin(d))
            if d[nearest] <= self.eps:
                self.hash_to_cluster[frame_hash] = nearest
                return nearest
        # Open a new cluster
        cid = self._next_id
        self._next_id += 1
        self.centers.append(embedding.astype(np.float32, copy=True))
        self.hash_to_cluster[frame_hash] = cid
        return cid

    @property
    def n_clusters(self) -> int:
        return len(self.centers)

    @property
    def n_hashes(self) -> int:
        return len(self.hash_to_cluster)

    def collapse_ratio(self) -> float:
        """Hashes per cluster — higher = more collapsing happened."""
        if self.n_clusters == 0:
            return 0.0
        return self.n_hashes / self.n_clusters

    def reset(self) -> None:
        self.centers.clear()
        self.hash_to_cluster.clear()
        self._next_id = 0
