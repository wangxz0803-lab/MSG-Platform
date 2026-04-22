from __future__ import annotations

import torch

from msg_embedding.models.adapters import LatentAdapter


class TestLatentAdapter:
    def test_default_construction(self):
        adapter = LatentAdapter()
        total = sum(p.numel() for p in adapter.parameters())
        assert 1400 < total < 1500

    def test_forward_shape(self):
        adapter = LatentAdapter(dim=16)
        z = torch.randn(4, 16)
        out = adapter(z)
        assert out.shape == (4, 16)

    def test_residual_starts_at_zero(self):
        adapter = LatentAdapter(dim=16)
        z = torch.randn(4, 16)
        out = adapter(z)
        torch.testing.assert_close(out, z, atol=1e-6, rtol=1e-6)

    def test_custom_dims(self):
        adapter = LatentAdapter(dim=32, hidden_dim=64, dropout=0.0)
        z = torch.randn(2, 32)
        out = adapter(z)
        assert out.shape == (2, 32)

    def test_gradient_flow(self):
        adapter = LatentAdapter(dim=16)
        z = torch.randn(2, 16, requires_grad=True)
        out = adapter(z)
        loss = out.sum()
        loss.backward()
        assert z.grad is not None
