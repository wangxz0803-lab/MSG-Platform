from __future__ import annotations

import pytest
import torch

from msg_embedding.features.extractor import FeatureExtractor

B = 4
ANT = 64


def _make_full_feat(batch_size: int = B, ant: int = ANT) -> dict[str, torch.Tensor]:
    def _complex(n: int) -> torch.Tensor:
        return torch.randn(batch_size, n) + 1j * torch.randn(batch_size, n)

    feat: dict[str, torch.Tensor] = {
        "pdp_crop": torch.rand(batch_size, 64),
        "srs_sinr": torch.randn(batch_size) * 10,
        "cqi": torch.randint(0, 16, (batch_size,)).float(),
        "cell_rsrp": torch.randn(batch_size, 16) * 30 - 110,
        "rsrp_srs": torch.randn(batch_size, ant) * 30 - 110,
        "rsrp_cb": torch.randn(batch_size, ant) * 30 - 110,
        "srs_w1": torch.rand(batch_size),
        "srs_w2": torch.rand(batch_size),
        "srs_w3": torch.rand(batch_size),
        "srs_w4": torch.rand(batch_size),
    }
    for prefix in ["srs", "pmi", "dft"]:
        for i in range(1, 5):
            feat[f"{prefix}{i}"] = _complex(ant)
    return feat


def _make_minimal_feat(batch_size: int = B) -> dict[str, torch.Tensor]:
    return {
        "pdp_crop": torch.rand(batch_size, 64),
        "cqi": torch.randint(0, 16, (batch_size,)).float(),
        "srs_sinr": torch.randn(batch_size) * 10,
        "srs1": torch.randn(batch_size, 64) + 1j * torch.randn(batch_size, 64),
    }


@pytest.fixture
def ext():
    return FeatureExtractor()


class TestFeatureExtractor:
    def test_default_construction(self, ext):
        assert ext._tx_ant_num_max == 64
        assert ext._token_dim == 128
        assert ext._seq_len == 16
        assert ext._cell_rsrp_dim == 16

    def test_custom_cfg(self):
        cfg = {"tx_ant_num_max": 32, "token_dim": 64, "seq_len": 8, "cell_rsrp_dim": 8}
        e = FeatureExtractor(cfg)
        assert e._tx_ant_num_max == 32
        assert e._token_dim == 64

    def test_forward_full_features_shape(self, ext):
        feat = _make_full_feat()
        tokens, stats = ext(feat)
        assert tokens.shape == (B, 16, 128)

    def test_forward_minimal_features_shape(self, ext):
        feat = _make_minimal_feat()
        tokens, stats = ext(feat)
        assert tokens.shape == (B, 16, 128)

    def test_token_mask_shape_and_type(self, ext):
        feat = _make_full_feat()
        _, stats = ext(feat)
        mask = stats["token_mask"]
        assert mask.shape == (B, 16)
        assert mask.dtype == torch.bool

    def test_all_present_no_mask(self, ext):
        feat = _make_full_feat()
        _, stats = ext(feat)
        mask = stats["token_mask"]
        assert not mask.any(), "All tokens present, mask should be all False"

    def test_missing_features_masked(self, ext):
        feat = _make_minimal_feat()
        _, stats = ext(feat)
        mask = stats["token_mask"]
        assert mask.any(), "Missing tokens should be masked True"

    def test_field_mask_tracking(self, ext):
        feat = _make_minimal_feat()
        _, stats = ext(feat)
        fm = stats["field_mask"]
        assert fm["pdp_crop"] is True
        assert fm["srs1"] is True
        assert fm["srs2"] is False
        assert fm["pmi1"] is False

    def test_norm_stats_has_rms(self, ext):
        feat = _make_full_feat()
        _, stats = ext(feat)
        assert "srs1" in stats
        assert "rms" in stats["srs1"]

    def test_smaller_antenna_padded(self):
        cfg = {"tx_ant_num_max": 64}
        e = FeatureExtractor(cfg)
        feat = {
            "cqi": torch.tensor([5.0]),
            "srs_sinr": torch.tensor([0.0]),
            "srs1": torch.randn(1, 32) + 1j * torch.randn(1, 32),
        }
        tokens, _ = e(feat)
        assert tokens.shape == (1, 16, 128)

    def test_infer_batch_info_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            FeatureExtractor._infer_batch_info({})

    def test_cell_rsrp_padding(self, ext):
        feat = {
            "cqi": torch.tensor([5.0]),
            "srs_sinr": torch.tensor([0.0]),
            "cell_rsrp": torch.randn(1, 8) * 30 - 110,
        }
        _, stats = ext(feat)
        assert stats["field_mask"]["cell_rsrp"] is True
