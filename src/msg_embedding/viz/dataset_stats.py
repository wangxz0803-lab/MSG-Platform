"""Manifest-level dataset statistics + codebook histograms."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from msg_embedding.core.logging import get_logger  # noqa: E402

_log = get_logger(__name__)


def _pick_column(df: pd.DataFrame, *candidates: str) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _histogram(values: np.ndarray, bins: int = 30) -> dict[str, list[float]]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"bin_edges": [], "counts": []}
    counts, edges = np.histogram(values, bins=bins)
    return {"bin_edges": edges.tolist(), "counts": counts.astype(int).tolist()}


def manifest_stats(manifest_df: pd.DataFrame) -> dict[str, Any]:
    """Compute headline dataset statistics from a manifest DataFrame."""
    out: dict[str, Any] = {"n_rows": int(len(manifest_df))}

    for key, candidates in (
        ("snr_hist", ("snr_dB", "snr_db")),
        ("sir_hist", ("sir_dB", "sir_db")),
        ("sinr_hist", ("sinr_dB", "sinr_db")),
    ):
        col = _pick_column(manifest_df, *candidates)
        if col is None:
            continue
        out[key] = _histogram(np.asarray(manifest_df[col], dtype=float))

    for key, col_name in (
        ("shard_counts", "shard_id"),
        ("source_counts", "source"),
        ("link_counts", "link"),
        ("split_counts", "split"),
        ("status_counts", "status"),
    ):
        if col_name in manifest_df.columns:
            counts = manifest_df[col_name].value_counts(dropna=False)
            out[key] = {
                (str(k) if not pd.isna(k) else "NA"): int(v) for k, v in counts.items()
            }

    return out


def plot_manifest_histograms(
    manifest_df: pd.DataFrame,
    out_path: str | Path,
    figsize: tuple[float, float] = (12.0, 8.0),
) -> plt.Figure:
    """Multi-panel matplotlib figure: SNR / SIR / SINR histograms + breakdowns."""
    panels: list[tuple[str, Any]] = []

    for label, candidates in (
        ("SNR (dB)", ("snr_dB", "snr_db")),
        ("SIR (dB)", ("sir_dB", "sir_db")),
        ("SINR (dB)", ("sinr_dB", "sinr_db")),
    ):
        col = _pick_column(manifest_df, *candidates)
        if col is not None:
            vals = np.asarray(manifest_df[col], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                panels.append((label, ("hist", vals)))

    for label, col_name in (
        ("source", "source"),
        ("link", "link"),
        ("split", "split"),
    ):
        if col_name in manifest_df.columns:
            counts = manifest_df[col_name].value_counts(dropna=False)
            if len(counts) > 0:
                panels.append((label, ("bar", counts)))

    if not panels:
        _log.warning("manifest_empty", detail="no plottable columns")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130)
        plt.close(fig)
        return fig

    n = len(panels)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    for i, (label, (kind, data)) in enumerate(panels):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        if kind == "hist":
            ax.hist(data, bins=30, color="steelblue", edgecolor="white")
            ax.set_xlabel(label)
            ax.set_ylabel("count")
        else:
            data_sorted = data.sort_values(ascending=False)
            ax.bar(
                [str(k) for k in data_sorted.index],
                data_sorted.values,
                color="coral",
                edgecolor="white",
            )
            ax.set_title(f"by {label}")
            ax.set_ylabel("count")
            for t in ax.get_xticklabels():
                t.set_rotation(30)
                t.set_ha("right")
    for i in range(len(panels), rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].axis("off")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    _log.info("manifest_histograms_saved", path=str(out))
    plt.close(fig)
    return fig


def pmi_codebook_histogram(
    pt_dir: str | Path,
    out_path: str | Path,
    max_files: int = 64,
    figsize: tuple[float, float] = (10.0, 4.0),
) -> dict[str, int]:
    """Scan ``.pt`` files and plot the PMI codebook index histogram."""
    pt_dir = Path(pt_dir)
    if not pt_dir.exists():
        _log.warning("pmi_scan_dir_missing", path=str(pt_dir))
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "no bridge_out .pt files found", ha="center", va="center")
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130)
        plt.close(fig)
        return {}

    try:
        import torch
    except ImportError:  # pragma: no cover
        _log.warning("pmi_scan_skip", reason="torch not installed")
        return {}

    counts: dict[int, int] = {}
    files = sorted(pt_dir.glob("*.pt"))[:max_files]
    if not files:
        _log.warning("pmi_no_pt_files", path=str(pt_dir))
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "no bridge_out .pt files found", ha="center", va="center")
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130)
        plt.close(fig)
        return {}

    for path in files:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:  # pragma: no cover
            continue
        if not isinstance(payload, dict):
            continue
        for key in ("pmi1", "pmi2", "pmi3", "pmi4"):
            if key not in payload:
                continue
            tensor = payload[key]
            if not hasattr(tensor, "abs"):
                continue
            mag = tensor.abs()
            if mag.ndim < 2:
                continue
            argmax = mag.reshape(mag.shape[0], -1).argmax(dim=-1).tolist()
            for idx in argmax:
                counts[int(idx)] = counts.get(int(idx), 0) + 1

    fig, ax = plt.subplots(figsize=figsize)
    if not counts:
        ax.text(0.5, 0.5, "no PMI data found in .pt files", ha="center", va="center")
    else:
        idx_sorted = sorted(counts.keys())
        vals = [counts[i] for i in idx_sorted]
        ax.bar(idx_sorted, vals, color="purple", edgecolor="white")
        ax.set_xlabel("PMI codebook index")
        ax.set_ylabel("sample count")
        ax.set_title("PMI codebook utilisation")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    _log.info("pmi_codebook_saved", path=str(out))
    return {str(k): int(v) for k, v in counts.items()}
