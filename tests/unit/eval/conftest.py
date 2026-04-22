"""Shared fixtures for eval tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

_N, _BS = 4, 8


@pytest.fixture(scope="module")
def small_feature_dict() -> dict[str, torch.Tensor]:
    """Minimal feature dict matching PROTOCOL_SPEC ranges."""
    n = _N
    bs = _BS
    rng = np.random.default_rng(44)

    def _cx(shape: tuple[int, ...]) -> torch.Tensor:
        arr = (
            rng.standard_normal(shape, dtype=np.float32)
            + 1j * rng.standard_normal(shape, dtype=np.float32)
        ).astype(np.complex64)
        return torch.from_numpy(arr)

    feat: dict[str, torch.Tensor] = {}
    for k in (
        "srs1", "srs2", "srs3", "srs4",
        "pmi1", "pmi2", "pmi3", "pmi4",
        "dft1", "dft2", "dft3", "dft4",
    ):
        feat[k] = _cx((n, bs))

    feat["pdp_crop"] = torch.rand(n, 64, dtype=torch.float32)
    feat["rsrp_srs"] = torch.empty(n, 64, dtype=torch.float32).uniform_(-120.0, -70.0)
    feat["rsrp_cb"] = torch.empty(n, 64, dtype=torch.float32).uniform_(-120.0, -70.0)
    feat["cell_rsrp"] = torch.full((n, 16), -110.0, dtype=torch.float32)

    feat["cqi"] = torch.randint(0, 16, (n,), dtype=torch.int64)
    feat["srs_sinr"] = torch.empty(n, dtype=torch.float32).uniform_(-10.0, 15.0)
    feat["srs_cb_sinr"] = torch.empty(n, dtype=torch.float32).uniform_(-10.0, 15.0)
    for k in ("srs_w1", "srs_w2", "srs_w3", "srs_w4"):
        feat[k] = torch.full((n,), 0.25, dtype=torch.float32)
    return feat
