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
    """Frozen random-init CNN: (64, 64) int frame -> spatial L2-normalized embedding.

    Architecture: one-hot 16 colors -> Conv(16->16, k=5, s=2) -> ReLU
    -> Conv(16->32, k=5, s=2) -> ReLU -> coarse `grid`x`grid` spatial pool
    -> flatten -> L2-normalize.

    **Why coarse spatial pool, not global mean pool (v0.1 change):** global
    mean pooling is position-invariant. In several ARC-AGI-3 games (tr87,
    wa30) frames differ mainly in *where* an object sits — exactly the
    signal the agent needs — so global pooling collapsed thousands of
    distinct frames into 1-2 clusters. A coarse grid pool keeps approximate
    object position while still abstracting pixel-level noise.

    With grid=4: embedding dim = 32 channels x 16 cells = 512.

    Seeded for reproducibility — same `seed` always yields the same weights.
    """

    def __init__(self, seed: int = 0xA3A3, grid: int = 4) -> None:
        self.grid = grid
        self.embed_dim = 32 * grid * grid
        rng = np.random.default_rng(seed)
        # He init for ReLU
        self.w1 = rng.normal(0, np.sqrt(2.0 / (16 * 5 * 5)), size=(16, 16, 5, 5)).astype(np.float32)
        self.w2 = rng.normal(0, np.sqrt(2.0 / (16 * 5 * 5)), size=(32, 16, 5, 5)).astype(np.float32)

    def _spatial_pool(self, x: np.ndarray) -> np.ndarray:
        """Average-pool (C, H, W) -> (C, grid, grid) by even spatial bins."""
        c, h, w = x.shape
        g = self.grid
        # Bin edges along each axis.
        hs = np.linspace(0, h, g + 1, dtype=int)
        ws = np.linspace(0, w, g + 1, dtype=int)
        out = np.zeros((c, g, g), dtype=np.float32)
        for i in range(g):
            for j in range(g):
                block = x[:, hs[i]:hs[i + 1], ws[j]:ws[j + 1]]
                if block.size:
                    out[:, i, j] = block.mean(axis=(1, 2))
        return out

    def embed(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 2 or frame.shape != (64, 64):
            f = np.zeros((64, 64), dtype=np.uint8)
            h, w = min(frame.shape[0], 64), min(frame.shape[1], 64)
            f[:h, :w] = np.asarray(frame, dtype=np.uint8)[:h, :w]
            frame = f
        x = np.zeros((16, 64, 64), dtype=np.float32)
        for c in range(16):
            x[c] = (frame == c).astype(np.float32)
        x = _conv2d(x, self.w1, stride=2)  # (16, 30, 30)
        np.maximum(x, 0, out=x)
        x = _conv2d(x, self.w2, stride=2)  # (32, 13, 13)
        np.maximum(x, 0, out=x)
        emb = self._spatial_pool(x).reshape(-1)  # (32*grid*grid,)
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
