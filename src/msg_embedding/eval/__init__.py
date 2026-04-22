"""msg_embedding.eval — downstream evaluation layer.

Submodules:

* :mod:`channel_charting` — latent-space geometry metrics (trustworthiness,
  continuity, kNN consistency, global rank correlation, UMAP/t-SNE).
* :mod:`prediction` — MAE reconstruction NMSE + cosine-similarity pair
  statistics + CSI compression-feedback equivalence.
* :mod:`runner` — orchestrator that ties a trained checkpoint to a
  :class:`ChannelDataset` and produces the flat ``metrics.json`` +
  ``embeddings.parquet`` artefact pair.
"""

from .prediction import (
    compression_feedback_equiv,
    cosine_distribution,
    nmse_reconstruction,
)

_LAZY_ATTRS = {
    "continuity": ("channel_charting", "continuity"),
    "trustworthiness": ("channel_charting", "trustworthiness"),
    "knn_consistency": ("channel_charting", "knn_consistency"),
    "kendall_tau_global": ("channel_charting", "kendall_tau_global"),
    "spearman_global": ("channel_charting", "spearman_global"),
    "umap_2d": ("channel_charting", "umap_2d"),
    "tsne_2d": ("channel_charting", "tsne_2d"),
    "EvalResult": ("runner", "EvalResult"),
    "run_eval": ("runner", "run_eval"),
    "get_git_sha": ("runner", "get_git_sha"),
    "is_ready": ("runner", "is_ready"),
}


def __getattr__(name: str):  # noqa: D401
    """Lazy loader for heavyweight submodules."""
    if name in _LAZY_ATTRS:
        from importlib import import_module

        mod_name, attr_name = _LAZY_ATTRS[name]
        module = import_module(f".{mod_name}", __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "continuity",
    "trustworthiness",
    "knn_consistency",
    "kendall_tau_global",
    "spearman_global",
    "umap_2d",
    "tsne_2d",
    "nmse_reconstruction",
    "cosine_distribution",
    "compression_feedback_equiv",
    "EvalResult",
    "run_eval",
    "get_git_sha",
    "is_ready",
]
