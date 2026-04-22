"""Latent-space visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from msg_embedding.core.logging import get_logger  # noqa: E402

_log = get_logger(__name__)


def umap_2d(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Project ``embeddings`` to 2-D (UMAP > TSNE > random projection)."""
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
    n = embeddings.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)

    try:
        import umap  # type: ignore[import-untyped]

        effective_nn = max(2, min(n_neighbors, n - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=effective_nn,
            min_dist=min_dist,
            random_state=random_state,
        )
        coords = reducer.fit_transform(embeddings)
        return np.asarray(coords, dtype=np.float32)
    except ImportError:
        _log.warning("umap_fallback", reason="umap-learn not installed, trying TSNE")

    try:
        from sklearn.manifold import TSNE  # type: ignore[import-untyped]

        perplexity = float(min(30.0, max(2.0, (n - 1) / 3.0)))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        coords = tsne.fit_transform(embeddings)
        return np.asarray(coords, dtype=np.float32)
    except ImportError:
        _log.warning("projection_fallback", reason="no umap-learn or sklearn, using random projection")

    rng = np.random.default_rng(random_state)
    proj = rng.standard_normal((embeddings.shape[1], 2)).astype(np.float32)
    return (embeddings.astype(np.float32) @ proj).astype(np.float32)


def plot_latent_scatter(
    coords: np.ndarray,
    color_by: np.ndarray | None = None,
    color_label: str | None = None,
    title: str | None = None,
    out_path: str | Path | None = None,
    figsize: tuple[float, float] = (6.0, 5.0),
) -> plt.Figure:
    """Static matplotlib latent-scatter."""
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (N, 2), got {coords.shape}")
    fig, ax = plt.subplots(figsize=figsize)
    if color_by is None:
        ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.7)
    else:
        color_arr = np.asarray(color_by)
        is_numeric = np.issubdtype(color_arr.dtype, np.number)
        if is_numeric:
            sc = ax.scatter(
                coords[:, 0], coords[:, 1], c=color_arr, s=8, alpha=0.8, cmap="viridis"
            )
            cbar = fig.colorbar(sc, ax=ax)
            if color_label:
                cbar.set_label(color_label)
        else:
            uniq = list(dict.fromkeys(color_arr.tolist()))
            cmap = plt.get_cmap("tab10", max(1, len(uniq)))
            for i, cat in enumerate(uniq):
                mask = color_arr == cat
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=8,
                    alpha=0.8,
                    color=cmap(i),
                    label=str(cat),
                )
            ax.legend(title=color_label, loc="best", fontsize="small")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        _log.info("latent_scatter_saved", path=str(out))
    return fig


def plot_latent_scatter_plotly(
    coords: np.ndarray,
    color_by: np.ndarray | None = None,
    color_label: str | None = None,
    title: str | None = None,
    out_path: str | Path | None = None,
) -> Any:
    """Interactive Plotly scatter (matplotlib fallback if plotly missing)."""
    try:
        import plotly.express as px  # type: ignore[import-untyped]
    except ImportError:
        _log.warning("plotly_fallback", reason="plotly not installed")
        fallback_out: Path | None = None
        if out_path is not None:
            out = Path(out_path)
            fallback_out = (
                out.with_suffix(".png") if out.suffix.lower() == ".html" else out
            )
        fig = plot_latent_scatter(
            coords,
            color_by=color_by,
            color_label=color_label,
            title=title,
            out_path=fallback_out,
        )
        if out_path is not None and Path(out_path).suffix.lower() == ".html":
            png_name = Path(fallback_out).name if fallback_out else ""
            Path(out_path).write_text(
                f"<html><body><img src='{png_name}' alt='latent scatter'/></body></html>",
                encoding="utf-8",
            )
        return fig

    import pandas as pd

    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
    color_kw: dict[str, Any] = {}
    if color_by is not None:
        df[color_label or "color"] = np.asarray(color_by)
        color_kw["color"] = color_label or "color"
    fig = px.scatter(
        df, x="x", y="y", title=title, opacity=0.8, render_mode="webgl", **color_kw
    )
    fig.update_traces(marker=dict(size=5))
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
        _log.info("plotly_scatter_saved", path=str(out))
    return fig


def plot_knn_edges(
    coords: np.ndarray,
    embeddings: np.ndarray,
    k: int = 5,
    out_path: str | Path | None = None,
    figsize: tuple[float, float] = (6.0, 5.0),
) -> plt.Figure:
    """Overlay a kNN graph on a 2-D projection."""
    if coords.shape[0] != embeddings.shape[0]:
        raise ValueError(
            "coords and embeddings must share N: "
            f"got {coords.shape[0]} vs {embeddings.shape[0]}"
        )
    n = coords.shape[0]
    k_eff = max(1, min(k, n - 1))

    diff = embeddings[:, None, :] - embeddings[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    idx = np.argpartition(dists, kth=k_eff - 1, axis=1)[:, :k_eff]

    fig, ax = plt.subplots(figsize=figsize)
    segs_x: list[float] = []
    segs_y: list[float] = []
    for i in range(n):
        for j in idx[i]:
            segs_x.extend([coords[i, 0], coords[j, 0], np.nan])
            segs_y.extend([coords[i, 1], coords[j, 1], np.nan])
    ax.plot(segs_x, segs_y, color="grey", alpha=0.25, linewidth=0.5)
    ax.scatter(coords[:, 0], coords[:, 1], s=10, color="tab:blue")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_title(f"kNN overlay (k={k_eff})")
    fig.tight_layout()
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        _log.info("knn_overlay_saved", path=str(out))
    return fig
