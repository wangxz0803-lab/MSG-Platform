"""Unit tests for :mod:`msg_embedding.data.sources.sionna_rt`.

Most of the tests target :class:`SionnaRTMockSource` — the Sionna-free
fallback that runs on any CI host. The tests marked with ``@pytest.mark.sionna``
exercise the real :class:`SionnaRTSource` and will be skipped automatically
when Sionna 2.0 is not importable.
"""

from __future__ import annotations

import numpy as np
import pytest

from msg_embedding.data.contract import ChannelSample
from msg_embedding.data.sources import SOURCE_REGISTRY
from msg_embedding.data.sources.sionna_rt import (
    _SIONNA_AVAILABLE,
    SionnaRTMockSource,
    SionnaRTSource,
)

# ---------------------------------------------------------------------------
# Mock-source tests (run everywhere)
# ---------------------------------------------------------------------------


def _mock_config(**overrides: object) -> dict[str, object]:
    cfg: dict[str, object] = {
        "num_samples": 3,
        "num_cells": 3,
        "num_ofdm_symbols": 2,
        "num_rb": 4,
        "num_bs_ant": 2,
        "num_ue_ant": 2,
        "serving_cell_id": 0,
        "channel_est_mode": "ls_linear",
        "link": "DL",
        "pilot_type": "csi_rs_gold",
        "scenario": "munich",
        "seed": 7,
    }
    cfg.update(overrides)
    return cfg


def test_registry_has_both_sources() -> None:
    """Both real + mock sources must be registered under their own names."""
    assert "sionna_rt" in SOURCE_REGISTRY
    assert SOURCE_REGISTRY["sionna_rt"] is SionnaRTSource
    assert "sionna_rt_mock" in SOURCE_REGISTRY
    assert SOURCE_REGISTRY["sionna_rt_mock"] is SionnaRTMockSource


def test_mock_describe_reports_mock_flag() -> None:
    src = SionnaRTMockSource(_mock_config())
    info = src.describe()
    assert info["source"] == "sionna_rt_mock"
    assert info["mock"] is True
    # `available` reflects the underlying Sionna install status.
    assert info["available"] is _SIONNA_AVAILABLE
    assert info["expected_sample_count"] == 3
    assert info["dimensions"]["K"] == 3


def test_mock_iter_samples_yields_valid_channel_samples() -> None:
    src = SionnaRTMockSource(_mock_config())
    samples = list(src.iter_samples())
    assert len(samples) == 3
    for s in samples:
        assert isinstance(s, ChannelSample)
        # contract source is the canonical 'sionna_rt', with meta['mock']=True.
        assert s.source == "sionna_rt"
        assert s.meta.get("mock") is True
        assert s.h_serving_true.dtype == np.complex64
        assert s.h_serving_est.dtype == np.complex64
        assert s.h_serving_true.shape == (2, 4, 2, 2)
        assert s.h_interferers is not None
        assert s.h_interferers.shape == (2, 2, 4, 2, 2)  # K-1 = 2
        assert s.link == "DL"
        assert s.channel_est_mode == "ls_linear"


def test_mock_single_cell_has_no_interferers() -> None:
    src = SionnaRTMockSource(_mock_config(num_cells=1, serving_cell_id=0))
    s = next(iter(src.iter_samples()))
    assert s.h_interferers is None
    assert s.serving_cell_id == 0


def test_mock_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="serving_cell_id"):
        SionnaRTMockSource(_mock_config(num_cells=2, serving_cell_id=5))
    with pytest.raises(ValueError, match="channel_est_mode"):
        SionnaRTMockSource(_mock_config(channel_est_mode="bogus"))
    with pytest.raises(ValueError, match="link"):
        SionnaRTMockSource(_mock_config(link="SL"))


def test_mock_seed_is_deterministic() -> None:
    a = list(SionnaRTMockSource(_mock_config(seed=42)).iter_samples())
    b = list(SionnaRTMockSource(_mock_config(seed=42)).iter_samples())
    assert len(a) == len(b)
    np.testing.assert_array_equal(a[0].h_serving_true, b[0].h_serving_true)
    np.testing.assert_array_equal(a[0].h_serving_est, b[0].h_serving_est)


def test_mock_snr_sir_clamped_to_contract_bounds() -> None:
    """Huge requested SNRs/SIRs must still pass `ChannelSample` validation."""
    src = SionnaRTMockSource(_mock_config(snr_dB=80.0, sir_dB=80.0))
    s = next(iter(src.iter_samples()))
    assert -50.0 <= s.snr_dB <= 50.0
    assert s.sir_dB is not None and -50.0 <= s.sir_dB <= 50.0
    assert -50.0 <= s.sinr_dB <= 50.0


# ---------------------------------------------------------------------------
# Real-source tests (ImportError surface + smoke)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_SIONNA_AVAILABLE, reason="Sionna is installed; fallback test skipped.")
def test_real_source_falls_back_when_sionna_missing() -> None:
    """On a CI host without Sionna, the real source falls back to TDL (no ImportError)."""
    src = SionnaRTSource({"scenario": "munich", "num_cells": 2})
    assert not src._use_real_sionna


@pytest.mark.sionna
def test_real_source_smoke_describe() -> None:
    """Real-Sionna smoke test — only runs when Sionna 2.0 is importable."""
    try:
        import sionna  # type: ignore  # noqa: F401
        import sionna.rt  # type: ignore  # noqa: F401
    except ImportError:
        pytest.skip("Sionna 2.0 not installed.")

    cfg = {
        "scenario": "munich",
        "num_cells": 2,
        "num_ues": 1,
        "num_samples": 1,
        "num_bs_ant": 2,
        "num_ue_ant": 2,
        "fft_size": 48,
        "num_ofdm_symbols": 2,
        "device": "cpu",
        "channel_est_mode": "ideal",
        "pilot_type": "csi_rs_gold",
        "link": "DL",
    }
    src = SionnaRTSource(cfg)
    info = src.describe()
    assert info["available"] is True
    assert info["source"] == "sionna_rt"
    assert info["dimensions"]["K"] == 2
