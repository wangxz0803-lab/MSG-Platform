"""Tests for the bridge's paired UL/DL token computation split."""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone

import numpy as np
import pytest
import torch

from msg_embedding.data.bridge import _build_feat_dict, sample_to_features
from msg_embedding.data.contract import ChannelSample
from msg_embedding.features.extractor import FeatureExtractor

T, RB, BS, UE = 4, 12, 4, 2


def _rng():
    return np.random.default_rng(42)


def _make_channel(rng, shape):
    h = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2)
    return h.astype(np.complex64)


def _make_paired_sample(rng) -> ChannelSample:
    h_dl = _make_channel(rng, (T, RB, BS, UE))
    h_dl_est = (h_dl + 0.1 * _make_channel(rng, (T, RB, BS, UE))).astype(np.complex64)
    h_ul = np.conj(h_dl.transpose(0, 1, 3, 2)).astype(np.complex64)
    h_ul_est = (h_ul + 0.1 * _make_channel(rng, (T, RB, UE, BS))).astype(np.complex64)

    return ChannelSample(
        h_serving_true=h_dl,
        h_serving_est=h_dl_est,
        noise_power_dBm=-100.0,
        snr_dB=20.0,
        sinr_dB=18.0,
        sir_dB=25.0,
        ssb_rsrp_dBm=[-80.0, -95.0],
        link="DL",
        channel_est_mode="ls_linear",
        serving_cell_id=0,
        ue_position=np.array([10.0, 20.0, 1.5]),
        source="internal_sim",
        sample_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        meta={},
        link_pairing="paired",
        h_ul_true=h_ul,
        h_ul_est=h_ul_est,
        h_dl_true=h_dl,
        h_dl_est=h_dl_est,
        ul_sir_dB=22.0,
        dl_sir_dB=25.0,
        num_interfering_ues=2,
    )


def _make_single_sample(rng) -> ChannelSample:
    h = _make_channel(rng, (T, RB, BS, UE))
    return ChannelSample(
        h_serving_true=h,
        h_serving_est=(h + 0.01 * _make_channel(rng, (T, RB, BS, UE))).astype(np.complex64),
        noise_power_dBm=-100.0,
        snr_dB=20.0,
        sinr_dB=18.0,
        link="DL",
        channel_est_mode="ls_linear",
        serving_cell_id=0,
        source="internal_sim",
        sample_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        meta={},
    )


class TestBuildFeatDict:
    def test_single_mode_backward_compat(self):
        rng = _rng()
        sample = _make_single_sample(rng)
        feat, ctx = _build_feat_dict(sample, use_legacy_pmi=False)
        assert "pdp_crop" in feat
        assert "srs1" in feat
        assert "pmi1" in feat
        assert "dft1" in feat
        assert ctx["link_pairing"] == "single"

    def test_paired_mode_produces_all_tokens(self):
        rng = _rng()
        sample = _make_paired_sample(rng)
        feat, ctx = _build_feat_dict(sample, use_legacy_pmi=False)
        assert ctx["link_pairing"] == "paired"
        for key in ["pdp_crop", "srs1", "srs2", "srs3", "srs4",
                     "pmi1", "pmi2", "pmi3", "pmi4",
                     "dft1", "dft2", "dft3", "dft4",
                     "rsrp_srs", "rsrp_cb", "cell_rsrp",
                     "srs_w1", "srs_w2", "srs_w3", "srs_w4",
                     "srs_sinr", "srs_cb_sinr", "cqi"]:
            assert key in feat, f"Missing key: {key}"

    def test_paired_uses_different_channels_for_ul_dl(self):
        rng = _rng()
        sample = _make_paired_sample(rng)

        # Build feat with actual paired split
        feat_paired, _ = _build_feat_dict(sample, use_legacy_pmi=False)

        # Build feat forcing DL channel for everything (old behavior)
        feat_dl_only, _ = _build_feat_dict(
            sample, use_legacy_pmi=False,
            h_ul_override=sample.h_dl_est,
            h_dl_override=sample.h_dl_est,
        )

        # SRS tokens should differ (UL vs DL channel)
        srs_paired = feat_paired["srs1"].numpy()
        srs_dl = feat_dl_only["srs1"].numpy()
        assert not np.allclose(srs_paired, srs_dl, atol=1e-3), \
            "SRS tokens should differ between paired (UL) and DL-only"

        # PMI tokens should be the same (both use DL)
        pmi_paired = feat_paired["pmi1"].numpy()
        pmi_dl = feat_dl_only["pmi1"].numpy()
        np.testing.assert_allclose(pmi_paired, pmi_dl, atol=1e-5,
                                   err_msg="PMI tokens should be identical (both from DL)")

    def test_gt_override_uses_true_channels(self):
        rng = _rng()
        sample = _make_paired_sample(rng)

        feat_est, _ = _build_feat_dict(sample, use_legacy_pmi=False)
        feat_gt, _ = _build_feat_dict(
            sample, use_legacy_pmi=False,
            h_ul_override=sample.h_ul_true.transpose(0, 1, 3, 2),
            h_dl_override=sample.h_dl_true,
        )

        # GT and est should produce different tokens (est has noise)
        pdp_est = feat_est["pdp_crop"].numpy()
        pdp_gt = feat_gt["pdp_crop"].numpy()
        assert not np.allclose(pdp_est, pdp_gt, atol=1e-3), \
            "GT PDP should differ from estimated PDP"


class TestSampleToFeatures:
    def test_paired_produces_gt_tokens(self):
        rng = _rng()
        sample = _make_paired_sample(rng)
        fe = FeatureExtractor()
        fe.eval()

        tokens, norm_stats = sample_to_features(sample, fe, use_legacy_pmi=False)
        assert tokens.shape == (1, 16, 128)
        assert norm_stats["gt_tokens"] is not None
        assert norm_stats["gt_tokens"].shape == (1, 16, 128)
        assert norm_stats["bridge_context"]["link_pairing"] == "paired"

    def test_single_has_no_gt_tokens(self):
        rng = _rng()
        sample = _make_single_sample(rng)
        fe = FeatureExtractor()
        fe.eval()

        tokens, norm_stats = sample_to_features(sample, fe, use_legacy_pmi=False)
        assert tokens.shape == (1, 16, 128)
        assert norm_stats["gt_tokens"] is None
        assert norm_stats["bridge_context"]["link_pairing"] == "single"

    def test_gt_tokens_closer_to_true_than_est(self):
        rng = _rng()
        sample = _make_paired_sample(rng)
        fe = FeatureExtractor()
        fe.eval()

        tokens_est, norm_stats = sample_to_features(sample, fe, use_legacy_pmi=False)
        tokens_gt = norm_stats["gt_tokens"]

        # GT tokens should be deterministically different from est tokens
        diff = float(torch.mean((tokens_est - tokens_gt) ** 2))
        assert diff > 1e-6, f"GT and est tokens should differ, got MSE={diff}"
