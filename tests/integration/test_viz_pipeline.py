"""Integration: trained model -> embeddings -> visualization outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def test_latent_to_scatter_plot(trained_mae, tmp_path: Path) -> None:
    """Extract latents and produce a scatter plot PNG."""
    mae, feat_ext, ckpt_info = trained_mae
    stacked = ckpt_info["stacked_feats"]

    with torch.no_grad():
        tokens, _ = feat_ext(stacked)
        snr = stacked.get("srs_sinr", torch.zeros(tokens.shape[0]))
        latent = mae.get_latent(tokens, snr)

    z = latent.numpy()
    coords = z[:, :2]
    colors = np.arange(z.shape[0], dtype=np.float32)

    from msg_embedding.viz.latent import plot_latent_scatter

    out = tmp_path / "scatter.png"
    fig = plot_latent_scatter(coords, color_by=colors, color_label="index", out_path=out)
    assert out.exists()
    assert out.stat().st_size > 100

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_dataset_stats_from_manifest(tmp_path: Path) -> None:
    """Build manifest stats and plot histograms."""
    import pandas as pd

    from msg_embedding.viz.dataset_stats import manifest_stats, plot_manifest_histograms

    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "sample_id": np.arange(32),
            "snr_dB": rng.uniform(-5, 25, 32).astype(np.float32),
            "sir_dB": rng.uniform(-5, 25, 32).astype(np.float32),
            "sinr_dB": rng.uniform(-10, 20, 32).astype(np.float32),
            "source": rng.choice(["sionna_rt", "quadriga_real"], 32),
            "link": rng.choice(["UL", "DL"], 32),
            "split": rng.choice(["train", "val", "test"], 32),
        }
    )

    stats = manifest_stats(df)
    assert stats["n_rows"] == 32
    assert "snr_hist" in stats
    assert "source_counts" in stats

    out = tmp_path / "hist.png"
    fig = plot_manifest_histograms(df, out)
    assert out.exists()
    assert fig is not None

    import matplotlib.pyplot as plt

    plt.close(fig)
