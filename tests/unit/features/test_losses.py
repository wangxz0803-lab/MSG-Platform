from __future__ import annotations

import torch

from msg_embedding.features.losses import contrastive_loss_fn


class TestContrastiveLoss:
    def test_returns_scalar(self):
        z1 = torch.randn(8, 16)
        z2 = torch.randn(8, 16)
        loss = contrastive_loss_fn(z1, z2)
        assert loss.ndim == 0

    def test_positive_loss(self):
        z1 = torch.randn(8, 16)
        z2 = torch.randn(8, 16)
        loss = contrastive_loss_fn(z1, z2)
        assert loss.item() > 0

    def test_identical_pairs_low_loss(self):
        z = torch.randn(8, 16)
        loss = contrastive_loss_fn(z, z)
        random_loss = contrastive_loss_fn(z, torch.randn(8, 16))
        assert loss.item() < random_loss.item()

    def test_with_regularization(self):
        z1 = torch.randn(8, 16)
        z2 = torch.randn(8, 16)
        loss_no_reg = contrastive_loss_fn(z1, z2, use_regularization=False)
        loss_reg = contrastive_loss_fn(z1, z2, use_regularization=True)
        assert loss_reg.item() != loss_no_reg.item()

    def test_temperature_scaling(self):
        z1 = torch.randn(8, 16)
        z2 = torch.randn(8, 16)
        loss_cold = contrastive_loss_fn(z1, z2, temperature=0.01)
        loss_warm = contrastive_loss_fn(z1, z2, temperature=1.0)
        assert loss_cold.item() != loss_warm.item()

    def test_gradients_flow(self):
        z1 = torch.randn(8, 16, requires_grad=True)
        z2 = torch.randn(8, 16, requires_grad=True)
        loss = contrastive_loss_fn(z1, z2)
        loss.backward()
        assert z1.grad is not None
        assert z2.grad is not None
