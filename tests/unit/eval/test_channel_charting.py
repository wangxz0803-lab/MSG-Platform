"""Tests for channel_charting metrics."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from msg_embedding.eval.channel_charting import (
    continuity,
    kendall_tau_global,
    knn_consistency,
    spearman_global,
    trustworthiness,
    tsne_2d,
    umap_2d,
)

_HAVE_SKLEARN = importlib.util.find_spec("sklearn") is not None


# ---------------------------------------------------------------------------
# Trustworthiness / Continuity
# ---------------------------------------------------------------------------

def test_trustworthiness_identity_mapping_near_one() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 8))
    tw = trustworthiness(x, x, k=5)
    assert tw >= 0.99


def test_continuity_identity_mapping_near_one() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((40, 8))
    ct = continuity(x, x, k=5)
    assert ct >= 0.99


def test_trustworthiness_random_mapping_lower() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((40, 8))
    y = rng.standard_normal((40, 2))
    tw = trustworthiness(x, y, k=5)
    assert 0.0 <= tw <= 1.0
    assert tw < 0.95


def test_trustworthiness_rejects_bad_k() -> None:
    x = np.random.default_rng(3).standard_normal((10, 4))
    with pytest.raises(ValueError):
        trustworthiness(x, x, k=10)


def test_trustworthiness_rejects_mismatched_n() -> None:
    x = np.random.default_rng(4).standard_normal((10, 4))
    y = np.random.default_rng(5).standard_normal((11, 2))
    with pytest.raises(ValueError):
        trustworthiness(x, y, k=3)


# ---------------------------------------------------------------------------
# kNN consistency
# ---------------------------------------------------------------------------

def test_knn_consistency_perfect_when_attrs_identical() -> None:
    rng = np.random.default_rng(10)
    emb = rng.standard_normal((30, 16))
    attrs = np.ones((30, 2))
    val = knn_consistency(emb, attrs, k=5)
    assert val == pytest.approx(1.0, abs=1e-6)


def test_knn_consistency_in_unit_interval_for_random_input() -> None:
    rng = np.random.default_rng(11)
    emb = rng.standard_normal((50, 16))
    attrs = rng.standard_normal((50, 3))
    val = knn_consistency(emb, attrs, k=5)
    assert 0.0 <= val <= 1.0


def test_knn_consistency_rejects_bad_shapes() -> None:
    emb = np.ones((10, 16))
    attrs = np.ones((11, 4))
    with pytest.raises(ValueError):
        knn_consistency(emb, attrs, k=3)


# ---------------------------------------------------------------------------
# Kendall / Spearman
# ---------------------------------------------------------------------------

def test_kendall_tau_identity_embedding_approx_one() -> None:
    rng = np.random.default_rng(20)
    coords = rng.standard_normal((25, 2))
    tau = kendall_tau_global(coords, coords)
    assert tau >= 0.99


def test_spearman_identity_embedding_approx_one() -> None:
    rng = np.random.default_rng(21)
    coords = rng.standard_normal((25, 2))
    rho = spearman_global(coords, coords)
    assert rho >= 0.99


def test_kendall_tau_random_between_minus1_and_1() -> None:
    rng = np.random.default_rng(22)
    emb = rng.standard_normal((25, 16))
    coords = rng.standard_normal((25, 2))
    tau = kendall_tau_global(emb, coords)
    assert -1.0 <= tau <= 1.0


def test_kendall_tau_constant_returns_zero() -> None:
    emb = np.ones((10, 4))
    coords = np.ones((10, 2))
    assert kendall_tau_global(emb, coords) == 0.0


# ---------------------------------------------------------------------------
# UMAP / t-SNE
# ---------------------------------------------------------------------------

def test_umap_2d_shape_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "umap", None)  # type: ignore[arg-type]

    rng = np.random.default_rng(30)
    emb = rng.standard_normal((20, 16))
    with pytest.warns(UserWarning):
        out = umap_2d(emb, n_neighbors=5, min_dist=0.1, seed=42)
    assert out.shape == (20, 2)
    assert np.all(np.isfinite(out))


@pytest.mark.skipif(not _HAVE_SKLEARN, reason="sklearn not installed")
def test_tsne_2d_shape() -> None:
    rng = np.random.default_rng(40)
    emb = rng.standard_normal((30, 16))
    out = tsne_2d(emb, perplexity=5, seed=42)
    assert out.shape == (30, 2)
    assert np.all(np.isfinite(out))
