"""Shared fixtures for integration tests."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
import pytest
import torch

from msg_embedding.data.contract import ChannelSample
from msg_embedding.features.extractor import FeatureExtractor
from msg_embedding.models.channel_mae import ChannelMAE


def _make_channel_sample(
    rng: np.random.Generator,
    *,
    link: str = "DL",
    source: str = "internal_sim",
    channel_est_mode: str = "ideal",
) -> ChannelSample:
    """Create a synthetic ChannelSample with realistic shapes."""
    n_bs, n_ue, n_rb, n_sym = 4, 2, 52, 14
    h = rng.standard_normal((n_sym, n_rb, n_bs, n_ue)) + 1j * rng.standard_normal(
        (n_sym, n_rb, n_bs, n_ue)
    )
    h = h.astype(np.complex64)
    scale = np.sqrt(np.mean(np.abs(h) ** 2))
    h = h / scale

    snr = float(rng.uniform(0, 30))
    return ChannelSample(
        h_serving_true=h,
        h_serving_est=h + 0.01 * rng.standard_normal(h.shape).astype(np.complex64),
        h_interferers=None,
        interference_signal=None,
        noise_power_dBm=-100.0,
        snr_dB=snr,
        sir_dB=float(rng.uniform(5, 20)),
        sinr_dB=snr - 3.0,
        ssb_rsrp_dBm=[-80.0, -95.0],
        link=link,
        channel_est_mode=channel_est_mode,
        serving_cell_id=0,
        ue_position=rng.uniform(-200, 200, size=3).astype(np.float64),
        source=source,
        sample_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        meta={"num_cells": 2},
    )


@pytest.fixture()
def synthetic_samples() -> list[ChannelSample]:
    """Generate 8 synthetic ChannelSamples."""
    rng = np.random.default_rng(42)
    return [_make_channel_sample(rng, link="DL" if i % 2 == 0 else "UL") for i in range(8)]


@pytest.fixture()
def trained_mae(tmp_path) -> tuple[ChannelMAE, FeatureExtractor, dict]:
    """Train a tiny MAE for 2 epochs on random data, return (model, feat_ext, ckpt_dict)."""
    rng = np.random.default_rng(99)
    feat_ext = FeatureExtractor()

    samples = [_make_channel_sample(rng) for _ in range(16)]
    from msg_embedding.data.bridge import _build_feat_dict

    feat_dicts = []
    for s in samples:
        fd, _ = _build_feat_dict(s, use_legacy_pmi=False)
        feat_dicts.append(fd)

    stacked = {}
    for key in feat_dicts[0]:
        stacked[key] = torch.cat([f[key] for f in feat_dicts], dim=0)

    with torch.no_grad():
        tokens, _ = feat_ext(stacked)

    mae = ChannelMAE(None)
    optimizer = torch.optim.Adam(mae.parameters(), lr=1e-3)

    for _ in range(2):
        mae.train()
        snr = stacked.get("srs_sinr", torch.zeros(tokens.shape[0]))
        recon = mae(tokens, snr, mask_ratio=0.3)
        loss = torch.nn.functional.mse_loss(recon, tokens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mae.eval()

    ckpt = {"model": mae.state_dict(), "epoch": 2}
    ckpt_path = tmp_path / "test_ckpt.pth"
    torch.save(ckpt, ckpt_path)

    return mae, feat_ext, {"path": ckpt_path, "data": ckpt, "stacked_feats": stacked}
