"""Shared fixtures for the data-package unit tests."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from msg_embedding.data.contract import ChannelSample


def _make_sample(
    *,
    source: str = "internal_sim",
    link: str = "UL",
    serving_cell_id: int = 0,
    channel_est_mode: str = "ls_linear",
    snr_dB: float = 10.0,
    sir_dB: float | None = 5.0,
    sinr_dB: float = 3.0,
    ue_position: tuple[float, float, float] | None = (10.0, 20.0, 1.5),
    shape: tuple[int, int, int, int] = (2, 4, 2, 2),
) -> ChannelSample:
    T, RB, BS, UE = shape
    rng = np.random.default_rng(abs(hash((source, link, serving_cell_id))) % (2**32))
    h_true = (
        rng.standard_normal((T, RB, BS, UE)) + 1j * rng.standard_normal((T, RB, BS, UE))
    ).astype(np.complex64)
    h_est = h_true + (
        0.01 * (rng.standard_normal(h_true.shape) + 1j * rng.standard_normal(h_true.shape))
    ).astype(np.complex64)
    pos = None if ue_position is None else np.asarray(ue_position, dtype=np.float64)
    return ChannelSample(
        h_serving_true=h_true,
        h_serving_est=h_est,
        noise_power_dBm=-100.0,
        snr_dB=snr_dB,
        sir_dB=sir_dB,
        sinr_dB=sinr_dB,
        link=link,
        channel_est_mode=channel_est_mode,
        serving_cell_id=serving_cell_id,
        ue_position=pos,
        source=source,
        sample_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        meta={},
    )


@pytest.fixture
def make_sample() -> Callable[..., ChannelSample]:
    """Factory for quickly building valid :class:`ChannelSample` objects."""
    return _make_sample


@pytest.fixture
def write_sample_pt(tmp_path) -> Callable[[ChannelSample, str], Path]:
    """Factory that pickles a sample to a .pt file and returns the path."""

    def _write(sample: ChannelSample, name: str = "sample.pt") -> Path:
        import torch

        path = tmp_path / name
        torch.save(sample.to_dict(), path)
        return path

    return _write


@pytest.fixture
def manifest_row_factory() -> Callable[..., dict]:
    """Factory that builds canonical manifest row dicts for tests."""

    def _row(
        *,
        uuid_: str | None = None,
        source: str = "internal_sim",
        link: str = "UL",
        snr_dB: float = 10.0,
        sinr_dB: float = 3.0,
        serving_cell_id: int = 0,
        ue_x: float = 10.0,
        ue_y: float = 20.0,
        ue_z: float = 1.5,
        path: str = "/tmp/fake.pt",
        status: str = "succeeded",
        split: str = "unassigned",
        channel_est_mode: str = "ls_linear",
    ) -> dict:
        return {
            "uuid": uuid_ or str(uuid.uuid4()),
            "source": source,
            "link": link,
            "snr_dB": snr_dB,
            "sinr_dB": sinr_dB,
            "serving_cell_id": serving_cell_id,
            "ue_x": ue_x,
            "ue_y": ue_y,
            "ue_z": ue_z,
            "path": path,
            "status": status,
            "split": split,
            "channel_est_mode": channel_est_mode,
        }

    return _row
