from __future__ import annotations

import torch
import torch.testing

from msg_embedding.features.normalizer import ProtocolNormalizer, db2lin, lin2db


class TestDb2Lin:
    def test_zero_db(self):
        assert db2lin(torch.tensor(0.0)).item() == 1.0

    def test_ten_db(self):
        torch.testing.assert_close(db2lin(torch.tensor(10.0)), torch.tensor(10.0))

    def test_negative_db(self):
        torch.testing.assert_close(db2lin(torch.tensor(-10.0)), torch.tensor(0.1))


class TestLin2Db:
    def test_one_linear(self):
        torch.testing.assert_close(lin2db(torch.tensor(1.0)), torch.tensor(0.0), atol=1e-5, rtol=0)

    def test_ten_linear(self):
        torch.testing.assert_close(
            lin2db(torch.tensor(10.0)), torch.tensor(10.0), atol=1e-5, rtol=0
        )

    def test_roundtrip(self):
        x = torch.tensor(5.0)
        torch.testing.assert_close(lin2db(db2lin(x)), x, atol=1e-5, rtol=0)


class TestProtocolNormalizer:
    def test_normalize_db_key(self):
        norm = ProtocolNormalizer()
        val = torch.tensor([-110.0])
        result = norm.normalize(val, "rsrp_srs")
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_normalize_linear_key(self):
        norm = ProtocolNormalizer()
        val = torch.tensor([0.0])
        result = norm.normalize(val, "srs_sinr")
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_normalize_discrete(self):
        norm = ProtocolNormalizer()
        val = torch.tensor([7.0])
        result = norm.normalize(val, "cqi")
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_unknown_key_passthrough(self):
        norm = ProtocolNormalizer()
        val = torch.tensor([42.0])
        result = norm.normalize(val, "nonexistent_key")
        torch.testing.assert_close(result, val)

    def test_denormalize_roundtrip_db(self):
        norm = ProtocolNormalizer()
        original = torch.tensor([-100.0, -80.0, -140.0])
        normalized = norm.normalize(original, "rsrp_srs")
        recovered = norm.denormalize(normalized, "rsrp_srs")
        torch.testing.assert_close(recovered, original, atol=1e-4, rtol=1e-4)

    def test_denormalize_roundtrip_discrete(self):
        norm = ProtocolNormalizer()
        original = torch.tensor([0.0, 7.0, 15.0])
        normalized = norm.normalize(original, "cqi")
        recovered = norm.denormalize(normalized, "cqi")
        torch.testing.assert_close(recovered, original, atol=0.6, rtol=0)

    def test_boundary_values(self):
        norm = ProtocolNormalizer()
        low = norm.normalize(torch.tensor([-160.0]), "rsrp_srs")
        high = norm.normalize(torch.tensor([-60.0]), "rsrp_srs")
        torch.testing.assert_close(low, torch.tensor([-1.0]), atol=1e-5, rtol=0)
        torch.testing.assert_close(high, torch.tensor([1.0]), atol=1e-5, rtol=0)

    def test_batch_normalize(self):
        norm = ProtocolNormalizer()
        batch = torch.tensor([-150.0, -110.0, -70.0])
        result = norm.normalize(batch, "cell_rsrp")
        assert result.shape == (3,)
        assert result.min() >= -1.0
        assert result.max() <= 1.0
