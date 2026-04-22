"""Latent-space geometry evaluation metrics."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.stats import kendalltau, spearmanr

from msg_embedding.core.logging import get_logger

try:
    from sklearn.neighbors import NearestNeighbors as _SkNearestNeighbors
    _HAVE_SKLEARN = True
except ImportError:  # pragma: no cover
    _SkNearestNeighbors = None  # type: ignore[assignment]
    _HAVE_SKLEARN = False

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_2d(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D [N, D], got shape {arr.shape}")
    return arr


def _check_paired(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x2 = _as_2d(x, "X_high")
    y2 = _as_2d(y, "X_low")
    if x2.shape[0] != y2.shape[0]:
        raise ValueError(
            f"X_high and X_low must share leading dim (N); got {x2.shape[0]} vs {y2.shape[0]}"
        )
    if x2.shape[0] < 3:
        raise ValueError(f"need at least 3 samples, got N={x2.shape[0]}")
    return x2, y2


def _knn_indices(x: np.ndarray, k: int, metric: str = "euclidean") -> np.ndarray:
    """Return ``[N, k]`` neighbour indices (self excluded)."""
    n = x.shape[0]
    if k >= n:
        raise ValueError(f"k={k} must be < N={n}")

    if _HAVE_SKLEARN and _SkNearestNeighbors is not None:
        nn = _SkNearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="auto")
        nn.fit(x)
        _, idx = nn.kneighbors(x, return_distance=True)
    else:
        if metric != "euclidean":
            raise RuntimeError(
                f"scipy fallback only supports metric='euclidean'; got {metric!r}"
            )
        tree = cKDTree(x)
        _, idx = tree.query(x, k=k + 1)

    trimmed = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        row = [j for j in np.asarray(idx[i]).tolist() if j != i][:k]
        if len(row) < k:
            row = np.asarray(idx[i])[1 : k + 1].tolist()
        trimmed[i] = row
    return trimmed


def _pairwise_rank_matrix(x: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Return ``[N, N]`` rank matrix: entry (i, j) = rank of j among i's neighbours."""
    n = x.shape[0]
    if _HAVE_SKLEARN and _SkNearestNeighbors is not None:
        nn = _SkNearestNeighbors(n_neighbors=n, metric=metric, algorithm="auto")
        nn.fit(x)
        _, order = nn.kneighbors(x, return_distance=True)
    else:
        if metric != "euclidean":
            raise RuntimeError(
                f"scipy fallback only supports metric='euclidean'; got {metric!r}"
            )
        tree = cKDTree(x)
        _, order = tree.query(x, k=n)
    order = np.asarray(order)
    ranks = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for r, j in enumerate(order[i]):
            ranks[i, int(j)] = r
    return ranks


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------

def trustworthiness(X_high: np.ndarray, X_low: np.ndarray, k: int = 10) -> float:
    """Lee & Verleysen trustworthiness in ``[0, 1]`` (higher = better)."""
    high, low = _check_paired(X_high, X_low)
    n = high.shape[0]
    if k <= 0 or k >= n:
        raise ValueError(f"k={k} must satisfy 0 < k < N={n}")

    high_ranks = _pairwise_rank_matrix(high)
    low_knn = _knn_indices(low, k)

    penalty = 0.0
    for i in range(n):
        for j in low_knn[i]:
            r = int(high_ranks[i, j])
            if r > k:
                penalty += r - k
    normalizer = n * k * (2 * n - 3 * k - 1)
    if normalizer <= 0:
        return 1.0
    return float(1.0 - 2.0 * penalty / normalizer)


def continuity(X_high: np.ndarray, X_low: np.ndarray, k: int = 10) -> float:
    """Continuity in ``[0, 1]`` — inverse of :func:`trustworthiness`."""
    high, low = _check_paired(X_high, X_low)
    n = high.shape[0]
    if k <= 0 or k >= n:
        raise ValueError(f"k={k} must satisfy 0 < k < N={n}")

    low_ranks = _pairwise_rank_matrix(low)
    high_knn = _knn_indices(high, k)

    penalty = 0.0
    for i in range(n):
        for j in high_knn[i]:
            r = int(low_ranks[i, j])
            if r > k:
                penalty += r - k
    normalizer = n * k * (2 * n - 3 * k - 1)
    if normalizer <= 0:
        return 1.0
    return float(1.0 - 2.0 * penalty / normalizer)


def knn_consistency(
    embeddings: np.ndarray,
    physical_attrs: np.ndarray,
    k: int = 5,
    metric: str = "euclidean",
) -> float:
    """Mean physical-attribute similarity of the ``k`` latent-space neighbours."""
    emb = _as_2d(embeddings, "embeddings")
    attr = _as_2d(physical_attrs, "physical_attrs")
    if emb.shape[0] != attr.shape[0]:
        raise ValueError(
            f"embeddings/attrs must share N; got {emb.shape[0]} vs {attr.shape[0]}"
        )
    if emb.shape[0] <= k:
        raise ValueError(f"k={k} must be < N={emb.shape[0]}")

    neighbours = _knn_indices(emb, k, metric=metric)
    sims = np.empty(emb.shape[0], dtype=np.float64)
    for i in range(emb.shape[0]):
        anchor = attr[i]
        nb = attr[neighbours[i]]
        diff = nb - anchor[None, :]
        d = np.linalg.norm(diff, axis=1)
        sims[i] = float(np.mean(1.0 / (1.0 + d)))
    return float(np.clip(np.mean(sims), 0.0, 1.0))


def kendall_tau_global(
    embeddings: np.ndarray,
    ground_truth_coords: np.ndarray,
) -> float:
    """Kendall's tau between pairwise embedding distances and GT distances."""
    emb = _as_2d(embeddings, "embeddings")
    coords = _as_2d(ground_truth_coords, "ground_truth_coords")
    if emb.shape[0] != coords.shape[0]:
        raise ValueError(
            f"N mismatch: embeddings {emb.shape[0]} vs coords {coords.shape[0]}"
        )
    d_emb = pdist(emb)
    d_gt = pdist(coords)
    if np.allclose(d_emb.std(), 0.0) or np.allclose(d_gt.std(), 0.0):
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        tau, _ = kendalltau(d_emb, d_gt)
    return float(tau) if np.isfinite(tau) else 0.0


def spearman_global(
    embeddings: np.ndarray,
    ground_truth_coords: np.ndarray,
) -> float:
    """Spearman's rho between pairwise embedding distances and GT distances."""
    emb = _as_2d(embeddings, "embeddings")
    coords = _as_2d(ground_truth_coords, "ground_truth_coords")
    if emb.shape[0] != coords.shape[0]:
        raise ValueError(
            f"N mismatch: embeddings {emb.shape[0]} vs coords {coords.shape[0]}"
        )
    d_emb = pdist(emb)
    d_gt = pdist(coords)
    if np.allclose(d_emb.std(), 0.0) or np.allclose(d_gt.std(), 0.0):
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        rho, _ = spearmanr(d_emb, d_gt)
    return float(rho) if np.isfinite(rho) else 0.0


# ---------------------------------------------------------------------------
# 2-D projections
# ---------------------------------------------------------------------------

def _pca_2d(x: np.ndarray) -> np.ndarray:
    """Fallback 2-D PCA projection — no external dependencies."""
    x = _as_2d(x, "embeddings")
    centered = x - x.mean(axis=0, keepdims=True)
    if centered.shape[1] <= 2:
        pad = np.zeros((centered.shape[0], 2 - centered.shape[1]))
        return np.concatenate([centered, pad], axis=1).astype(np.float64)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    pcs = u[:, :2] * s[:2]
    return pcs.astype(np.float64)


def umap_2d(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Return an ``[N, 2]`` UMAP projection (PCA fallback if umap-learn missing)."""
    emb = _as_2d(embeddings, "embeddings")
    try:
        import umap  # type: ignore[import-untyped]
    except ImportError:
        warnings.warn(
            "umap-learn not installed; falling back to PCA-2D projection.",
            stacklevel=2,
        )
        _log.warning("umap_fallback", reason="umap-learn not installed")
        return _pca_2d(emb)

    n = emb.shape[0]
    safe_n_neighbors = max(2, min(int(n_neighbors), max(2, n - 1)))
    reducer = umap.UMAP(  # type: ignore[attr-defined]
        n_components=2,
        n_neighbors=safe_n_neighbors,
        min_dist=float(min_dist),
        random_state=int(seed),
    )
    return np.asarray(reducer.fit_transform(emb), dtype=np.float64)


def tsne_2d(
    embeddings: np.ndarray,
    perplexity: float = 30.0,
    seed: int = 42,
) -> np.ndarray:
    """Return an ``[N, 2]`` t-SNE projection (sklearn-backed)."""
    emb = _as_2d(embeddings, "embeddings")
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:  # pragma: no cover
        raise ImportError("tsne_2d requires scikit-learn to be installed") from exc

    n = emb.shape[0]
    safe_perp = float(min(perplexity, max(5.0, n / 3.0 - 1.0)))
    tsne = TSNE(
        n_components=2,
        perplexity=safe_perp,
        random_state=int(seed),
        init="pca",
        learning_rate="auto",
    )
    return np.asarray(tsne.fit_transform(emb), dtype=np.float64)


__all__ = [
    "trustworthiness",
    "continuity",
    "knn_consistency",
    "kendall_tau_global",
    "spearman_global",
    "umap_2d",
    "tsne_2d",
]

MetricName = Literal[
    "trustworthiness",
    "continuity",
    "knn_consistency",
    "kendall_tau",
    "spearman",
]
