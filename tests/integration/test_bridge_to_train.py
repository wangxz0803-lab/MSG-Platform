"""Integration: ChannelSample -> bridge -> FeatureExtractor -> ChannelMAE forward."""

from __future__ import annotations

import torch

from msg_embedding.data.bridge import _build_feat_dict
from msg_embedding.data.contract import ChannelSample
from msg_embedding.features.extractor import FeatureExtractor
from msg_embedding.models.channel_mae import ChannelMAE


def test_sample_through_bridge_and_model(synthetic_samples: list[ChannelSample]) -> None:
    """Full path: 8 samples -> bridge -> tokens -> MAE forward + latent."""
    feat_dicts = []
    for s in synthetic_samples:
        fd, ctx = _build_feat_dict(s, use_legacy_pmi=False)
        feat_dicts.append(fd)
        assert isinstance(fd, dict)
        assert len(fd) > 0

    stacked = {}
    for key in feat_dicts[0]:
        stacked[key] = torch.cat([f[key] for f in feat_dicts], dim=0)

    feat_ext = FeatureExtractor()
    with torch.no_grad():
        tokens, n_special = feat_ext(stacked)

    assert tokens.shape[0] == 8
    assert tokens.shape[1] == 16
    assert tokens.shape[2] == 128

    mae = ChannelMAE(None)
    mae.eval()
    snr = stacked.get("srs_sinr", torch.zeros(tokens.shape[0]))

    with torch.no_grad():
        recon = mae(tokens, snr, mask_ratio=0.3)
        latent = mae.get_latent(tokens, snr)

    assert recon.shape == tokens.shape
    assert latent.shape == (8, 16)
    assert torch.isfinite(recon).all()
    assert torch.isfinite(latent).all()


def test_bridge_preserves_snr_metadata(synthetic_samples: list[ChannelSample]) -> None:
    """Verify that bridge output includes SNR/SINR gate features."""
    fd, ctx = _build_feat_dict(synthetic_samples[0], use_legacy_pmi=False)
    assert "srs_sinr" in fd or "pmi_sinr" in fd or any("sinr" in k.lower() for k in fd)


def test_model_gradient_flows(synthetic_samples: list[ChannelSample]) -> None:
    """Verify gradients flow through the full forward pass."""
    fd, _ = _build_feat_dict(synthetic_samples[0], use_legacy_pmi=False)
    batch = {k: v for k, v in fd.items()}

    feat_ext = FeatureExtractor()
    tokens, _ = feat_ext(batch)

    mae = ChannelMAE(None)
    mae.train()
    snr = batch.get("srs_sinr", torch.zeros(tokens.shape[0]))

    recon = mae(tokens, snr, mask_ratio=0.3)
    loss = torch.nn.functional.mse_loss(recon, tokens)
    loss.backward()

    grad_count = sum(1 for p in mae.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert grad_count > 0, "No gradients flowed through model"
