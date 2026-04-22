"""Tests for latent visualization."""

from __future__ import annotations

import numpy as np
import pytest

from msg_embedding.viz.latent import (
    plot_knn_edges,
    plot_latent_scatter,
    plot_latent_scatter_plotly,
    umap_2d,
)


def test_umap_2d_shape_random_input() -> None:
    try:
        import umap  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        try:
            from sklearn.manifold import TSNE  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            pytest.skip("neither umap-learn nor scikit-learn is installed")

    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 16)).astype(np.float32)
    coords = umap_2d(x, n_neighbors=5, min_dist=0.1)
    assert coords.shape == (100, 2)
    assert np.isfinite(coords).all()


def test_umap_2d_empty() -> None:
    coords = umap_2d(np.zeros((0, 4), dtype=np.float32))
    assert coords.shape == (0, 2)


def test_plot_latent_scatter_continuous(tmp_path) -> None:
    coords = np.random.default_rng(1).standard_normal((30, 2)).astype(np.float32)
    colors = np.random.default_rng(2).standard_normal(30).astype(np.float32)
    out = tmp_path / "scatter.png"
    fig = plot_latent_scatter(
        coords, color_by=colors, color_label="sinr", title="test", out_path=out
    )
    assert out.exists()
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_latent_scatter_categorical(tmp_path) -> None:
    coords = np.random.default_rng(3).standard_normal((20, 2)).astype(np.float32)
    labels = np.array(["A", "B", "A", "C"] * 5)
    out = tmp_path / "scatter_cat.png"
    fig = plot_latent_scatter(coords, color_by=labels, color_label="link", out_path=out)
    assert out.exists()
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_latent_scatter_plotly_fallback(tmp_path) -> None:
    coords = np.random.default_rng(4).standard_normal((15, 2)).astype(np.float32)
    out = tmp_path / "scatter.html"
    fig = plot_latent_scatter_plotly(coords, title="t", out_path=out)
    assert fig is not None
    assert out.exists()


def test_plot_knn_edges(tmp_path) -> None:
    rng = np.random.default_rng(5)
    embeddings = rng.standard_normal((30, 8)).astype(np.float32)
    coords = embeddings[:, :2].copy()
    out = tmp_path / "knn.png"
    fig = plot_knn_edges(coords, embeddings, k=3, out_path=out)
    assert out.exists()
    import matplotlib.pyplot as plt

    plt.close(fig)
