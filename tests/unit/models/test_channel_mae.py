from __future__ import annotations

import pytest
import torch

from msg_embedding.models.channel_mae import ChannelMAE


@pytest.fixture
def model():
    return ChannelMAE()


@pytest.fixture
def batch():
    B, L, D = 4, 16, 128
    return torch.randn(B, L, D)


class TestChannelMAE:
    def test_default_construction(self, model):
        assert model._token_dim == 128
        assert model._seq_len == 16
        assert model._latent_out_dim == 16

    def test_custom_cfg(self):
        cfg = {"token_dim": 64, "seq_len": 8, "latent_out_dim": 8, "n_heads": 4}
        m = ChannelMAE(cfg)
        assert m._token_dim == 64
        assert m._seq_len == 8

    def test_encode_shape(self, model, batch):
        enc = model.encode(batch)
        assert enc.shape == (4, 16, 128)

    def test_encode_with_snr(self, model, batch):
        snr = torch.randn(4)
        enc = model.encode(batch, snr_db=snr)
        assert enc.shape == (4, 16, 128)

    def test_pool_and_project_shape(self, model, batch):
        enc = model.encode(batch)
        latent = model.pool_and_project(enc)
        assert latent.shape == (4, 16)

    def test_latent_is_normalized(self, model, batch):
        enc = model.encode(batch)
        latent = model.pool_and_project(enc)
        norms = torch.norm(latent, dim=-1)
        torch.testing.assert_close(norms, torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_get_latent_shape(self, model, batch):
        latent = model.get_latent(batch)
        assert latent.shape == (4, 16)

    def test_get_latent_infer_mode(self, model, batch):
        model.eval()
        latent = model.get_latent(batch, is_infer=True)
        assert latent.shape == (4, 16)

    def test_forward_shape(self, model, batch):
        dec_out = model.forward(batch)
        assert dec_out.shape == (4, 16, 128)

    def test_forward_with_mask(self, model, batch):
        model.train()
        dec_out = model.forward(batch, mask_ratio=0.3)
        assert dec_out.shape == (4, 16, 128)

    def test_reconstruct_keys(self, model, batch):
        dec_out = model.forward(batch)
        features = model.reconstruct(dec_out)
        assert "pdp_crop" in features
        assert "srs1" in features
        assert "pmi1" in features
        assert "dft1" in features
        assert "rsrp_srs" in features
        assert "rsrp_cb" in features
        assert "cell_rsrp" in features

    def test_reconstruct_pdp_shape(self, model, batch):
        dec_out = model.forward(batch)
        features = model.reconstruct(dec_out)
        assert features["pdp_crop"].shape == (4, 64)

    def test_reconstruct_complex_shape(self, model, batch):
        dec_out = model.forward(batch)
        features = model.reconstruct(dec_out)
        assert features["srs1"].shape == (4, 64)
        assert features["srs1"].is_complex()

    def test_reconstruct_cell_rsrp_shape(self, model, batch):
        dec_out = model.forward(batch)
        features = model.reconstruct(dec_out)
        assert features["cell_rsrp"].shape == (4, 16)

    def test_reconstruct_values_bounded(self, model, batch):
        dec_out = model.forward(batch)
        features = model.reconstruct(dec_out)
        assert features["pdp_crop"].abs().max() <= 1.0
        assert features["rsrp_srs"].abs().max() <= 1.0

    def test_adapter_integration(self, model, batch):
        latent_no_adapter = model.get_latent(batch, use_adapter=False)
        latent_with_adapter = model.get_latent(batch, use_adapter=True)
        assert latent_no_adapter.shape == latent_with_adapter.shape

    def test_snr_encoder_none(self):
        cfg = {"use_snr_condition": False}
        m = ChannelMAE(cfg)
        assert m.snr_encoder is None
