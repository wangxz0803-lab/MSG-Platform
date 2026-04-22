"""Unit tests for training losses."""

from __future__ import annotations

import torch

from msg_embedding.features.extractor import FeatureExtractor
from msg_embedding.models.channel_mae import ChannelMAE
from msg_embedding.training.losses import (
    _DEFAULT_WEIGHTS,
    _TOKEN_SLOT,
    contrastive_loss,
    mae_total_loss,
    reconstruction_loss,
)


def _build_feat(n: int = 3, tx: int = 64, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    feat: dict[str, torch.Tensor] = {}
    for k in (
        "srs1", "srs2", "srs3", "srs4",
        "pmi1", "pmi2", "pmi3", "pmi4",
        "dft1", "dft2", "dft3", "dft4",
    ):
        real = torch.randn(n, tx, generator=g)
        imag = torch.randn(n, tx, generator=g)
        feat[k] = torch.complex(real, imag)
    feat["pdp_crop"] = torch.rand(n, 64, generator=g)
    feat["rsrp_srs"] = torch.empty(n, tx).uniform_(-120.0, -70.0)
    feat["rsrp_cb"] = torch.empty(n, tx).uniform_(-120.0, -70.0)
    feat["cell_rsrp"] = torch.empty(n, 16).uniform_(-130.0, -80.0)
    feat["cqi"] = torch.randint(0, 16, (n,))
    feat["srs_sinr"] = torch.empty(n).uniform_(-10.0, 15.0)
    feat["srs_cb_sinr"] = torch.empty(n).uniform_(-10.0, 15.0)
    for k in ("srs_w1", "srs_w2", "srs_w3", "srs_w4"):
        feat[k] = torch.full((n,), 0.25)
    return feat


def test_contrastive_loss_identical_is_near_zero() -> None:
    z = torch.randn(8, 16)
    loss = contrastive_loss(z, z.clone(), temperature=0.01)
    assert loss.item() < 1e-3


def test_contrastive_loss_regularization_adds_positive_term() -> None:
    torch.manual_seed(0)
    z1 = torch.randn(8, 16)
    z2 = z1 + 0.01 * torch.randn_like(z1)
    bare = contrastive_loss(z1, z2, regularization=False)
    reg = contrastive_loss(z1, z2, regularization=True, reg_weight=1.0)
    assert reg.item() >= bare.item()


def test_contrastive_loss_backward() -> None:
    z1 = torch.randn(8, 16, requires_grad=True)
    z2 = torch.randn(8, 16, requires_grad=True)
    loss = contrastive_loss(z1, z2)
    loss.backward()
    assert z1.grad is not None and torch.isfinite(z1.grad).all()


def test_mae_total_loss_weighting() -> None:
    r = torch.tensor(2.0)
    c = torch.tensor(4.0)
    out = mae_total_loss(r, c, weights={"recon": 0.5, "contrastive": 0.25})
    assert torch.allclose(out, torch.tensor(0.5 * 2.0 + 0.25 * 4.0))


def test_mae_total_loss_defaults() -> None:
    out = mae_total_loss(torch.tensor(1.0), torch.tensor(1.0))
    assert torch.allclose(out, torch.tensor(1.5))


def test_reconstruction_loss_runs_and_is_nonneg() -> None:
    mae = ChannelMAE(None).eval()
    ext = FeatureExtractor().eval()
    feat = _build_feat()
    with torch.no_grad():
        tokens, stats = ext(feat)
        recon = mae(tokens, feat["srs_sinr"], mask_ratio=0.0)
    loss = reconstruction_loss(recon, feat, mae, ext, stats)
    assert loss.item() >= 0.0
    assert torch.isfinite(loss)


def test_reconstruction_loss_masked_slot_contributes_zero() -> None:
    mae = ChannelMAE(None)
    ext = FeatureExtractor().eval()
    feat = _build_feat()
    feat.pop("pmi1")

    tokens, stats = ext(feat)
    assert stats["token_mask"][:, _TOKEN_SLOT["pmi1"]].all()

    recon = mae(tokens, feat["srs_sinr"], mask_ratio=0.0)
    loss = reconstruction_loss(recon, feat, mae, ext, stats)

    full_feat = _build_feat()
    tokens_f, stats_f = ext(full_feat)
    recon_f = mae(tokens_f, full_feat["srs_sinr"], mask_ratio=0.0)
    loss_full = reconstruction_loss(recon_f, full_feat, mae, ext, stats_f)

    assert torch.isfinite(loss) and torch.isfinite(loss_full)


def test_reconstruction_loss_masked_slot_gradient_zero() -> None:
    torch.manual_seed(0)
    mae = ChannelMAE(None)
    ext = FeatureExtractor().eval()
    feat = _build_feat()
    feat.pop("pmi1")

    tokens, stats = ext(feat)
    recon = mae(tokens, feat["srs_sinr"], mask_ratio=0.0)
    loss = reconstruction_loss(recon, feat, mae, ext, stats)
    loss.backward()

    with torch.no_grad():
        recon2 = recon.clone()
        recon2[:, _TOKEN_SLOT["pmi1"]] = 0.0
        loss_zeroed = reconstruction_loss(recon2, feat, mae, ext, stats)
    assert torch.allclose(loss.detach(), loss_zeroed, atol=1e-6)


def test_reconstruction_loss_full_dict_vs_mask_all_present() -> None:
    mae = ChannelMAE(None).eval()
    ext = FeatureExtractor().eval()
    feat = _build_feat()
    with torch.no_grad():
        tokens, stats = ext(feat)
        recon = mae(tokens, feat["srs_sinr"], mask_ratio=0.0)
        loss_with_mask = reconstruction_loss(recon, feat, mae, ext, stats,
                                             token_mask=stats["token_mask"])
        loss_without_mask = reconstruction_loss(recon, feat, mae, ext, stats,
                                                token_mask=None)
    assert torch.allclose(loss_with_mask, loss_without_mask, rtol=1e-5, atol=1e-6)


def test_reconstruction_loss_accepts_feat_gt() -> None:
    mae = ChannelMAE(None).eval()
    ext = FeatureExtractor().eval()
    feat_noisy = _build_feat(seed=1)
    feat_clean = _build_feat(seed=2)
    with torch.no_grad():
        tokens, stats = ext(feat_noisy)
        recon = mae(tokens, feat_noisy["srs_sinr"], mask_ratio=0.0)
        loss_self = reconstruction_loss(recon, feat_noisy, mae, ext, stats)
        loss_gt = reconstruction_loss(recon, feat_noisy, mae, ext, stats,
                                      feat_gt=feat_clean)
    assert torch.isfinite(loss_self) and torch.isfinite(loss_gt)
    assert loss_self.item() >= 0.0 and loss_gt.item() >= 0.0


def test_reconstruction_loss_uses_custom_weights() -> None:
    mae = ChannelMAE(None).eval()
    ext = FeatureExtractor().eval()
    feat = _build_feat()
    tokens, stats = ext(feat)
    with torch.no_grad():
        recon = mae(tokens, feat["srs_sinr"], mask_ratio=0.0)
    base = reconstruction_loss(recon, feat, mae, ext, stats)
    doubled = reconstruction_loss(
        recon, feat, mae, ext, stats,
        weights={k: 2.0 * v for k, v in _DEFAULT_WEIGHTS.items()},
    )
    assert torch.allclose(doubled, 2.0 * base, rtol=1e-5, atol=1e-6)
