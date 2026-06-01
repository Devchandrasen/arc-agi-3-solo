"""Phase 2 agent: GraphAgent + CNN-based state-space collapsing.

Same graph-exploration core as Phase 1, but each frame's identity is the
cluster ID assigned by a frozen random-init CNN embedder, not the raw
Blake2B hash. Visually-similar frames collapse to one node, so the BFS
operates on the underlying *game states* rather than on every
distinct-but-equivalent pixel pattern.

Motivated by the Cluster-2 failure mode (see docs/FAILURE_MODES.md):
re86, tr87, wa30 all reach 10k+ distinct frame hashes in 5 min without
solving a single level, suggesting the per-hash graph is overcounting
states by ~100x.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from arc_agi3_solo.agents.graph_agent import GraphAgent
from arc_agi3_solo.core.novelty import EmbeddingClusterMap, RandomCNNEmbedder

logger = logging.getLogger(__name__)


class CNNGraphAgent(GraphAgent):
    """GraphAgent variant: cluster IDs (not frame hashes) drive the explorer."""

    #: L2 distance threshold for the cluster map (cosine-equivalent on
    #: L2-normalized 32D embeddings). Tuned conservatively — lower = less
    #: collapsing, higher = more aggressive merging.
    #: L2 threshold for cluster assignment. Calibrated against the
    #: actual embedding-distance distribution on re86 sample frames
    #: (median ~0.047, p25 ~0.037). 0.025 admits only the tightest
    #: ~10% of frame pairs, giving conservative collapse.
    CLUSTER_EPS: float = 0.025

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Deterministic per-game seed so embeddings are reproducible.
        seed = hash(self.game_id) & 0xFFFFFFFF
        self.embedder = RandomCNNEmbedder(seed=seed)
        self.cluster_map = EmbeddingClusterMap(eps=self.CLUSTER_EPS)
        # Capture the original raw-hash function BEFORE any monkey-patching;
        # _hash_frame needs it as a stable, non-recursive reference.
        self._raw_hash = self.frame_processor.hash_frame

    @property
    def name(self) -> str:
        return f"{super().name}.cnncluster"

    def _hash_frame(self, frame_np: np.ndarray) -> str:
        """Override the frame-identity step: embed and assign cluster.

        Returns a stable string ID of the form 'c<N>'. Same input frame
        always returns the same cluster ID; visually-similar frames merge.
        """
        # We still need a stable per-pixel hash to key the cluster map (so
        # repeated visits to the same exact frame don't re-embed). Use the
        # captured original to avoid recursion through the monkey-patch.
        raw_hash = self._raw_hash(frame_np)
        emb = self.embedder.embed(frame_np)
        cid = self.cluster_map.assign(raw_hash, emb)
        return f"c{cid}"

    def choose_action(self, frames, latest_frame):  # type: ignore[override]
        # GraphAgent.choose_action computes `hashed_frame = frame_processor.hash_frame(...)`
        # then uses it as the graph node. We override frame_processor.hash_frame
        # via monkey-patching for the duration of this call to inject our
        # cluster-ID logic without forking the parent's implementation.
        original = self.frame_processor.hash_frame
        self.frame_processor.hash_frame = self._hash_frame  # type: ignore[method-assign]
        try:
            return super().choose_action(frames, latest_frame)
        finally:
            self.frame_processor.hash_frame = original  # type: ignore[method-assign]

    def main(self) -> None:
        super().main()
        logger.info(
            "%s done: %d hashes -> %d clusters (collapse %.1fx)",
            self.game_id, self.cluster_map.n_hashes, self.cluster_map.n_clusters,
            self.cluster_map.collapse_ratio(),
        )
