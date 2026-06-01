"""Unit tests for the Phase 2 novelty module (random CNN + cluster map)."""

from __future__ import annotations

import numpy as np
import pytest

from arc_agi3_solo.core.novelty import EmbeddingClusterMap, RandomCNNEmbedder


def _checkerboard(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 16, size=(64, 64), dtype=np.uint8)


def test_embedder_is_deterministic_across_instances() -> None:
    e1 = RandomCNNEmbedder(seed=0xA3A3)
    e2 = RandomCNNEmbedder(seed=0xA3A3)
    frame = _checkerboard(7)
    assert np.allclose(e1.embed(frame), e2.embed(frame))


def test_embedder_output_is_l2_normalized() -> None:
    e = RandomCNNEmbedder(seed=42)
    for s in range(3):
        emb = e.embed(_checkerboard(s))
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-5


def test_embedder_different_seeds_give_different_features() -> None:
    e1 = RandomCNNEmbedder(seed=1)
    e2 = RandomCNNEmbedder(seed=2)
    frame = _checkerboard(11)
    assert not np.allclose(e1.embed(frame), e2.embed(frame))


def test_embedder_distinct_frames_have_distinct_embeddings() -> None:
    """Two frames that differ in dominant color should embed apart.

    Random-checkerboard frames have similar color histograms after global
    mean pooling and L2 normalization, so distance can be small (~0.08).
    For the embedder to be useful at all it should at least separate
    frames with structurally different content -- here, a mostly-red vs
    a mostly-blue frame.
    """
    e = RandomCNNEmbedder(seed=0xA3A3)
    a = np.full((64, 64), 8, dtype=np.uint8)   # mostly red (color 8)
    b = np.full((64, 64), 9, dtype=np.uint8)   # mostly blue (color 9)
    a[10:50, 10:50] = 0                        # background patch
    b[10:50, 10:50] = 0
    emb_a, emb_b = e.embed(a), e.embed(b)
    assert np.linalg.norm(emb_a - emb_b) > 0.3


def test_embedder_identical_frames_have_zero_distance() -> None:
    e = RandomCNNEmbedder(seed=0xA3A3)
    frame = _checkerboard(33)
    assert np.allclose(e.embed(frame), e.embed(frame.copy()))


def test_cluster_map_stable_assignment_per_hash() -> None:
    """Same hash always returns the same cluster ID even if a closer cluster appears later."""
    cm = EmbeddingClusterMap(eps=0.4)
    emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb_b = np.array([0.95, 0.31, 0.0], dtype=np.float32)
    emb_b /= np.linalg.norm(emb_b)
    cid_a = cm.assign("a", emb_a)
    cid_b = cm.assign("b", emb_b)
    # b is within eps of a -> joins cluster 0
    assert cid_a == 0 and cid_b == 0
    # Re-assigning a returns the cached cluster
    assert cm.assign("a", emb_a) == 0


def test_cluster_map_opens_new_cluster_when_far() -> None:
    cm = EmbeddingClusterMap(eps=0.1)
    e1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    e2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert cm.assign("a", e1) == 0
    assert cm.assign("b", e2) == 1
    assert cm.n_clusters == 2
    assert cm.n_hashes == 2


def test_cluster_map_collapse_ratio_grows_with_hashes() -> None:
    cm = EmbeddingClusterMap(eps=0.4)
    base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    # 5 hashes, all near `base` -> 1 cluster
    rng = np.random.default_rng(0)
    for i in range(5):
        noise = rng.normal(0, 0.05, size=3).astype(np.float32)
        emb = base + noise
        emb /= np.linalg.norm(emb)
        cm.assign(f"h{i}", emb)
    assert cm.n_clusters == 1
    assert cm.collapse_ratio() == pytest.approx(5.0)


def test_cluster_map_reset_clears_state() -> None:
    cm = EmbeddingClusterMap()
    cm.assign("a", np.array([1.0, 0.0], dtype=np.float32))
    cm.reset()
    assert cm.n_clusters == 0 and cm.n_hashes == 0


def test_embedder_works_on_real_arc_value_range() -> None:
    """Frames are uint8 values 0..15. Embedder must handle that range."""
    e = RandomCNNEmbedder(seed=0xA3A3)
    frame = np.zeros((64, 64), dtype=np.uint8)
    frame[10:40, 10:40] = 9  # blue blob
    frame[20:30, 50:60] = 11  # yellow blob
    emb = e.embed(frame)
    assert emb.shape == (32,)
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-5
