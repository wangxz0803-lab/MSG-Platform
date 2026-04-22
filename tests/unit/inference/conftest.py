"""Shared fixtures for inference tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from msg_embedding.models.channel_mae import ChannelMAE


@pytest.fixture(scope="session")
def channel_mae_ckpt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    model = ChannelMAE(None)
    model.eval()
    tmp = tmp_path_factory.mktemp("ckpt")
    ckpt_path = tmp / "ckpt_best.pth"
    payload = {
        "model": model.state_dict(),
        "epoch": 0,
        "step": 0,
        "best_val_loss": 0.0,
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


@pytest.fixture(scope="session")
def sample_feat_dict_factory():
    def _make(n: int = 2) -> dict[str, torch.Tensor]:
        rng = torch.Generator().manual_seed(42)
        feat: dict[str, torch.Tensor] = {}
        for k in (
            "srs1", "srs2", "srs3", "srs4",
            "pmi1", "pmi2", "pmi3", "pmi4",
            "dft1", "dft2", "dft3", "dft4",
        ):
            real = torch.randn(n, 64, generator=rng, dtype=torch.float32)
            imag = torch.randn(n, 64, generator=rng, dtype=torch.float32)
            feat[k] = torch.complex(real, imag)

        feat["pdp_crop"] = torch.rand(n, 64, generator=rng, dtype=torch.float32)
        feat["rsrp_srs"] = torch.empty(n, 64).uniform_(-120.0, -70.0)
        feat["rsrp_cb"] = torch.empty(n, 64).uniform_(-120.0, -70.0)
        feat["cell_rsrp"] = torch.full((n, 16), -110.0, dtype=torch.float32)
        feat["cqi"] = torch.randint(0, 16, (n,), generator=rng,
                                    dtype=torch.int64)
        feat["srs_sinr"] = torch.empty(n).uniform_(-10.0, 15.0)
        feat["srs_cb_sinr"] = torch.empty(n).uniform_(-10.0, 15.0)
        for k in ("srs_w1", "srs_w2", "srs_w3", "srs_w4"):
            feat[k] = torch.full((n,), 0.25, dtype=torch.float32)
        return feat

    return _make
