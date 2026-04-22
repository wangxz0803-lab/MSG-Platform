from __future__ import annotations

import torch

from msg_embedding.features.denormalize import denormalize_reconstruction
from msg_embedding.features.normalizer import ProtocolNormalizer


class TestDenormalizeReconstruction:
    def test_pdp_range(self):
        normalizer = ProtocolNormalizer()
        recon = {"pdp_crop": torch.zeros(2, 64)}
        result = denormalize_reconstruction(recon, normalizer)
        assert result["pdp_crop"].min() >= 0.0
        assert result["pdp_crop"].max() <= 1.0

    def test_pdp_boundary_values(self):
        normalizer = ProtocolNormalizer()
        recon = {"pdp_crop": torch.tensor([[-1.0] * 64, [1.0] * 64])}
        result = denormalize_reconstruction(recon, normalizer)
        torch.testing.assert_close(result["pdp_crop"][0], torch.zeros(64), atol=1e-6, rtol=0)
        torch.testing.assert_close(result["pdp_crop"][1], torch.ones(64), atol=1e-6, rtol=0)

    def test_rsrp_clamped(self):
        normalizer = ProtocolNormalizer()
        recon = {"rsrp_srs": torch.zeros(2, 64)}
        result = denormalize_reconstruction(recon, normalizer)
        assert result["rsrp_srs"].min() >= -160.0
        assert result["rsrp_srs"].max() <= -60.0

    def test_cell_rsrp_clamped(self):
        normalizer = ProtocolNormalizer()
        recon = {"cell_rsrp": torch.zeros(2, 16)}
        result = denormalize_reconstruction(recon, normalizer)
        assert result["cell_rsrp"].min() >= -160.0
        assert result["cell_rsrp"].max() <= -60.0

    def test_complex_rms_recovery(self):
        normalizer = ProtocolNormalizer()
        rms = torch.tensor([[2.0], [3.0]])
        recon = {"srs1": torch.complex(torch.ones(2, 64), torch.ones(2, 64))}
        norm_stats = {"srs1": {"rms": rms.squeeze(-1)}}
        result = denormalize_reconstruction(recon, normalizer, norm_stats)
        assert result["srs1"].is_complex()
        assert result["srs1"].shape == (2, 64)

    def test_complex_without_stats(self):
        normalizer = ProtocolNormalizer()
        recon = {"dft1": torch.complex(torch.ones(2, 64), torch.zeros(2, 64))}
        result = denormalize_reconstruction(recon, normalizer)
        torch.testing.assert_close(result["dft1"], recon["dft1"])

    def test_empty_recon(self):
        normalizer = ProtocolNormalizer()
        result = denormalize_reconstruction({}, normalizer)
        assert len(result) == 0
