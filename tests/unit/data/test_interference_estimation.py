"""Tests for the shared interference-aware channel estimation module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from msg_embedding.data.sources._interference_estimation import (
    InterferenceEstResult,
    _generate_interferer_pilots_csirs,
    _generate_interferer_pilots_srs,
    estimate_channel_with_interference,
    estimate_paired_channels,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

T, RB, BS, UE = 4, 12, 4, 2


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def h_serving(rng):
    """Unit-power complex Gaussian serving channel [T, RB, BS, UE]."""
    h = (rng.standard_normal((T, RB, BS, UE)) + 1j * rng.standard_normal((T, RB, BS, UE))) / math.sqrt(2)
    return h.astype(np.complex64)


@pytest.fixture()
def h_interferers(rng):
    """Two interferer channels [2, T, RB, BS, UE]."""
    h = (rng.standard_normal((2, T, RB, BS, UE)) + 1j * rng.standard_normal((2, T, RB, BS, UE))) / math.sqrt(2)
    return h.astype(np.complex64)


@pytest.fixture()
def pilots_srs():
    from msg_embedding.data.sources.internal_sim import _generate_pilots_srs
    return _generate_pilots_srs(RB, cell_id=0)


@pytest.fixture()
def pilots_csirs():
    from msg_embedding.data.sources.internal_sim import _generate_pilots_csirs
    return _generate_pilots_csirs(RB, cell_id=0)


# ---------------------------------------------------------------------------
# Pilot generation
# ---------------------------------------------------------------------------

class TestInterfererPilots:
    def test_srs_pilots_shape(self):
        seq = _generate_interferer_pilots_srs(RB, cell_id=1, ue_index=0)
        assert seq.shape == (RB,)
        assert seq.dtype == np.complex128

    def test_srs_different_ues_different_sequences(self):
        seq0 = _generate_interferer_pilots_srs(RB, cell_id=1, ue_index=0)
        seq1 = _generate_interferer_pilots_srs(RB, cell_id=1, ue_index=1)
        assert not np.allclose(seq0, seq1), "Different UEs should use different ZC roots"

    def test_csirs_pilots_shape(self):
        seq = _generate_interferer_pilots_csirs(RB, cell_id=5)
        assert seq.shape == (RB,)
        assert seq.dtype == np.complex128

    def test_csirs_different_cells_different_sequences(self):
        seq0 = _generate_interferer_pilots_csirs(RB, cell_id=0)
        seq1 = _generate_interferer_pilots_csirs(RB, cell_id=1)
        assert not np.allclose(seq0, seq1)


# ---------------------------------------------------------------------------
# Ideal mode (bypass)
# ---------------------------------------------------------------------------

class TestIdealMode:
    def test_ideal_returns_true_channel(self, h_serving, rng, pilots_srs):
        result = estimate_channel_with_interference(
            h_serving_true=h_serving,
            h_interferers=None,
            pilots_serving=pilots_srs,
            interferer_cell_ids=None,
            direction="UL",
            snr_dB=20.0,
            rng=rng,
            est_mode="ideal",
        )
        assert isinstance(result, InterferenceEstResult)
        assert result.h_est.shape == h_serving.shape
        assert result.sir_dB is None
        np.testing.assert_allclose(result.h_est, h_serving, atol=1e-5)


# ---------------------------------------------------------------------------
# Noise-only estimation (no interferers)
# ---------------------------------------------------------------------------

class TestNoiseOnly:
    def test_noise_only_shape(self, h_serving, rng, pilots_srs):
        result = estimate_channel_with_interference(
            h_serving_true=h_serving,
            h_interferers=None,
            pilots_serving=pilots_srs,
            interferer_cell_ids=None,
            direction="UL",
            snr_dB=30.0,
            rng=rng,
            est_mode="ls_linear",
        )
        assert result.h_est.shape == h_serving.shape
        assert result.h_est.dtype == np.complex64
        assert result.sir_dB is None

    def test_high_snr_close_to_true(self, rng, pilots_srs):
        # Use a time-constant channel to avoid interpolation error
        # dominating the NMSE (sparse time pilots with T=4, spacing=4 → 1 sample)
        h_const = np.broadcast_to(
            (rng.standard_normal((1, RB, BS, UE)) + 1j * rng.standard_normal((1, RB, BS, UE))) / math.sqrt(2),
            (T, RB, BS, UE),
        ).astype(np.complex64).copy()
        result = estimate_channel_with_interference(
            h_serving_true=h_const,
            h_interferers=None,
            pilots_serving=pilots_srs,
            interferer_cell_ids=None,
            direction="UL",
            snr_dB=40.0,
            rng=np.random.default_rng(99),
            est_mode="ls_linear",
        )
        nmse = float(np.mean(np.abs(result.h_est - h_const) ** 2) / np.mean(np.abs(h_const) ** 2))
        assert nmse < 0.05, f"High-SNR noise-only NMSE should be low, got {nmse}"


# ---------------------------------------------------------------------------
# Interference-aware estimation
# ---------------------------------------------------------------------------

class TestInterferenceAware:
    def test_ul_with_interference_returns_sir(self, h_serving, h_interferers, rng, pilots_srs):
        result = estimate_channel_with_interference(
            h_serving_true=h_serving,
            h_interferers=h_interferers,
            pilots_serving=pilots_srs,
            interferer_cell_ids=[1, 2],
            direction="UL",
            snr_dB=30.0,
            rng=rng,
            est_mode="ls_linear",
            num_interfering_ues=2,
        )
        assert result.h_est.shape == h_serving.shape
        assert result.sir_dB is not None
        assert -50 <= result.sir_dB <= 50

    def test_dl_with_interference_returns_sir(self, h_serving, h_interferers, rng, pilots_csirs):
        result = estimate_channel_with_interference(
            h_serving_true=h_serving,
            h_interferers=h_interferers,
            pilots_serving=pilots_csirs,
            interferer_cell_ids=[10, 20],
            direction="DL",
            snr_dB=30.0,
            rng=rng,
            est_mode="ls_linear",
        )
        assert result.h_est.shape == h_serving.shape
        assert result.sir_dB is not None

    def test_interference_degrades_estimate(self, h_serving, h_interferers, rng, pilots_srs):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        clean = estimate_channel_with_interference(
            h_serving_true=h_serving,
            h_interferers=None,
            pilots_serving=pilots_srs,
            interferer_cell_ids=None,
            direction="UL",
            snr_dB=30.0,
            rng=rng1,
            est_mode="ls_linear",
        )
        dirty = estimate_channel_with_interference(
            h_serving_true=h_serving,
            h_interferers=h_interferers,
            pilots_serving=pilots_srs,
            interferer_cell_ids=[1, 2],
            direction="UL",
            snr_dB=30.0,
            rng=rng2,
            est_mode="ls_linear",
            num_interfering_ues=3,
        )
        nmse_clean = float(np.mean(np.abs(clean.h_est - h_serving) ** 2) / np.mean(np.abs(h_serving) ** 2))
        nmse_dirty = float(np.mean(np.abs(dirty.h_est - h_serving) ** 2) / np.mean(np.abs(h_serving) ** 2))
        assert nmse_dirty > nmse_clean, (
            f"Interference should degrade estimation: dirty NMSE={nmse_dirty:.4f} <= clean NMSE={nmse_clean:.4f}"
        )

    def test_empty_interferers_same_as_none(self, h_serving, rng, pilots_srs):
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)

        r1 = estimate_channel_with_interference(
            h_serving_true=h_serving,
            h_interferers=None,
            pilots_serving=pilots_srs,
            interferer_cell_ids=None,
            direction="UL",
            snr_dB=25.0,
            rng=rng1,
            est_mode="ls_linear",
        )
        h_empty = np.zeros((0, T, RB, BS, UE), dtype=np.complex64)
        r2 = estimate_channel_with_interference(
            h_serving_true=h_serving,
            h_interferers=h_empty,
            pilots_serving=pilots_srs,
            interferer_cell_ids=[],
            direction="UL",
            snr_dB=25.0,
            rng=rng2,
            est_mode="ls_linear",
        )
        np.testing.assert_allclose(r1.h_est, r2.h_est, atol=1e-5)


# ---------------------------------------------------------------------------
# Paired channel estimation
# ---------------------------------------------------------------------------

class TestPairedChannels:
    def test_paired_returns_all_keys(self, h_serving, h_interferers, rng):
        result = estimate_paired_channels(
            h_dl_true=h_serving,
            h_interferers_dl=h_interferers,
            serving_cell_id=0,
            interferer_cell_ids=[1, 2],
            snr_dB=25.0,
            rng=rng,
            est_mode="ls_linear",
            num_interfering_ues=2,
        )
        required = {"h_ul_true", "h_ul_est", "h_dl_true", "h_dl_est", "ul_sir_dB", "dl_sir_dB", "num_interfering_ues"}
        assert set(result.keys()) == required

    def test_paired_shapes(self, h_serving, h_interferers, rng):
        result = estimate_paired_channels(
            h_dl_true=h_serving,
            h_interferers_dl=h_interferers,
            serving_cell_id=0,
            interferer_cell_ids=[1, 2],
            snr_dB=25.0,
            rng=rng,
        )
        # UL shape has transposed antenna dims relative to DL
        assert result["h_dl_true"].shape == (T, RB, BS, UE)
        assert result["h_dl_est"].shape == (T, RB, BS, UE)
        assert result["h_ul_true"].shape == (T, RB, UE, BS)
        assert result["h_ul_est"].shape == (T, RB, UE, BS)

    def test_ul_is_conjugate_transpose_of_dl(self, h_serving, rng):
        result = estimate_paired_channels(
            h_dl_true=h_serving,
            h_interferers_dl=None,
            serving_cell_id=0,
            interferer_cell_ids=None,
            snr_dB=25.0,
            rng=rng,
            reciprocity_noise_scale=0.0,
        )
        h_ul_expected = np.conj(h_serving.transpose(0, 1, 3, 2))
        np.testing.assert_allclose(
            result["h_ul_true"], h_ul_expected, atol=1e-5,
            err_msg="UL true should be conjugate transpose of DL true (zero reciprocity noise)",
        )

    def test_paired_no_interferers(self, h_serving, rng):
        result = estimate_paired_channels(
            h_dl_true=h_serving,
            h_interferers_dl=None,
            serving_cell_id=0,
            interferer_cell_ids=None,
            snr_dB=25.0,
            rng=rng,
        )
        assert result["ul_sir_dB"] is None
        assert result["dl_sir_dB"] is None
        assert result["num_interfering_ues"] is None

    def test_paired_with_interferers_has_sir(self, h_serving, h_interferers, rng):
        result = estimate_paired_channels(
            h_dl_true=h_serving,
            h_interferers_dl=h_interferers,
            serving_cell_id=0,
            interferer_cell_ids=[1, 2],
            snr_dB=25.0,
            rng=rng,
        )
        assert result["ul_sir_dB"] is not None
        assert result["dl_sir_dB"] is not None
        assert result["num_interfering_ues"] == 3  # default
