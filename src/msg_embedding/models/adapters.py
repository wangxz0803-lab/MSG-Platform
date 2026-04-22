from __future__ import annotations

import torch
import torch.nn as nn


class LatentAdapter(nn.Module):
    """Dual-layer wide latent space adapter (~1,451 parameters).

    Designed for rapid online fine-tuning with 5K+ samples.
    Two-layer structure provides hierarchical adaptation without
    catastrophic forgetting of pretrained representations.
    """

    def __init__(self, dim: int = 16, hidden_dim: int = 40, dropout: float = 0.15) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.down1 = nn.Linear(dim, hidden_dim)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.up = nn.Linear(hidden_dim, dim)

        self.scale1 = nn.Parameter(torch.zeros(1))
        self.scale2 = nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_norm = self.norm1(z)
        h = self.down1(z_norm)
        h = self.act1(h)
        h = self.dropout1(h)
        h = h * torch.tanh(self.scale1)

        h = self.norm2(h)
        residual = self.up(h)

        return z + torch.tanh(self.scale2) * residual
