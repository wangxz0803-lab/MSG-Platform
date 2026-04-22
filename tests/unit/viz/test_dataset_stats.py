"""Tests for dataset_stats."""

from __future__ import annotations

import numpy as np
import pandas as pd

from msg_embedding.viz.dataset_stats import (
    manifest_stats,
    plot_manifest_histograms,
    pmi_codebook_histogram,
)


def _fake_manifest(n: int = 64) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "sample_id": np.arange(n, dtype=np.int32),
            "shard_id": rng.integers(0, 4, size=n, dtype=np.int32),
            "snr_dB": rng.uniform(-5, 25, size=n).astype(np.float32),
            "sir_dB": rng.uniform(-5, 25, size=n).astype(np.float32),
            "sinr_dB": rng.uniform(-10, 20, size=n).astype(np.float32),
            "link": rng.choice(["UL", "DL"], size=n),
            "source": rng.choice(["quadriga", "sionna"], size=n),
            "split": rng.choice(["train", "val", "test"], size=n),
            "num_cells": rng.integers(1, 8, size=n, dtype=np.int32),
            "status": rng.choice(["ok", "pending"], size=n),
        }
    )


def test_manifest_stats_has_expected_keys() -> None:
    df = _fake_manifest()
    stats = manifest_stats(df)
    assert stats["n_rows"] == len(df)
    for key in ("snr_hist", "sir_hist", "sinr_hist"):
        assert key in stats
        assert "bin_edges" in stats[key]
        assert "counts" in stats[key]
    for key in ("shard_counts", "source_counts", "link_counts", "split_counts"):
        assert key in stats


def test_manifest_stats_missing_columns() -> None:
    df = pd.DataFrame({"sample_id": [0, 1, 2]})
    stats = manifest_stats(df)
    assert stats["n_rows"] == 3
    assert "snr_hist" not in stats
    assert "source_counts" not in stats


def test_plot_manifest_histograms_writes_file(tmp_path) -> None:
    df = _fake_manifest()
    out = tmp_path / "hist.png"
    fig = plot_manifest_histograms(df, out)
    assert out.exists()
    assert fig is not None


def test_pmi_codebook_histogram_missing_dir(tmp_path) -> None:
    out = tmp_path / "pmi.png"
    counts = pmi_codebook_histogram(tmp_path / "does_not_exist", out)
    assert counts == {}
    assert out.exists()


def test_pmi_codebook_histogram_empty_dir(tmp_path) -> None:
    (tmp_path / "empty").mkdir()
    out = tmp_path / "pmi.png"
    counts = pmi_codebook_histogram(tmp_path / "empty", out)
    assert counts == {}
    assert out.exists()
