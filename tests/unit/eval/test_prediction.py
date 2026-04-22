"""Tests for prediction metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from msg_embedding.eval.prediction import (
    compression_feedback_equiv,
    cosine_distribution,
    nmse_reconstruction,
)
from msg_embedding.features.extractor import FeatureExtractor
from msg_embedding.models.channel_mae import ChannelMAE

# ---------------------------------------------------------------------------
# Reconstruction NMSE
# ---------------------------------------------------------------------------

def test_nmse_reconstruction_finite(small_feature_dict: dict[str, torch.Tensor]) -> None:
    mae = ChannelMAE(None).eval()
    fe = FeatureExtractor().eval()

    sample = {"feat": small_feature_dict, "feat_gt": small_feature_dict}
    out = nmse_reconstruction(mae, fe, [sample])

    assert "nmse_dB" in out
    assert "nmse_per_feature" in out
    assert np.isfinite(out["nmse_dB"])
    assert isinstance(out["nmse_per_feature"], dict)
    assert out["n_samples_total"] == small_feature_dict["cqi"].shape[0]


def test_nmse_reconstruction_requires_feat() -> None:
    mae = ChannelMAE(None).eval()
    fe = FeatureExtractor().eval()
    with pytest.raises(ValueError):
        nmse_reconstruction(mae, fe, [{}])


# ---------------------------------------------------------------------------
# Cosine distribution
# ---------------------------------------------------------------------------

def test_cosine_distribution_separation_margin_positive() -> None:
    rng = np.random.default_rng(0)
    anchors = rng.standard_normal((4, 8))
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
    partners = np.empty_like(anchors)
    partners[:2] = anchors[:2] + 1e-3 * rng.standard_normal((2, 8))
    partners[2:] = rng.standard_normal((2, 8))
    partners[2:] /= np.linalg.norm(partners[2:], axis=1, keepdims=True)

    embeddings = np.empty((8, 8))
    embeddings[0::2] = anchors
    embeddings[1::2] = partners
    labels = ["clean_noisy", "clean_noisy", "diff_ue", "diff_ue"]

    out = cosine_distribution(embeddings, labels)
    assert out["clean_noisy_n"] == 2
    assert out["diff_ue_n"] == 2
    assert out["clean_noisy_mean"] > out["diff_ue_mean"]
    assert out["separation_margin"] > 0


def test_cosine_distribution_bad_shape_raises() -> None:
    with pytest.raises(ValueError):
        cosine_distribution(np.zeros((3, 4)), ["clean_noisy"])


def test_cosine_distribution_missing_label_is_nan() -> None:
    emb = np.eye(4)
    out = cosine_distribution(emb, ["clean_noisy", "clean_noisy"])
    assert np.isnan(out["diff_ue_mean"])
    assert out["diff_ue_n"] == 0
    assert out["separation_margin"] == 0.0


# ---------------------------------------------------------------------------
# Compression feedback equivalence
# ---------------------------------------------------------------------------

def test_compression_feedback_equiv_runs(small_feature_dict: dict[str, torch.Tensor]) -> None:
    mae = ChannelMAE(None).eval()
    fe = FeatureExtractor().eval()

    sample = {"feat": small_feature_dict, "feat_gt": small_feature_dict}
    out = compression_feedback_equiv(mae, fe, [sample], baseline="pmi_type1")

    assert set(out.keys()) == {
        "sinr_diff_dB",
        "compression_ratio",
        "latent_sinr_dB",
        "baseline_sinr_dB",
    }
    assert np.isfinite(out["sinr_diff_dB"])
    assert out["compression_ratio"] >= 1.0


def test_compression_feedback_equiv_rejects_unknown_baseline(
    small_feature_dict: dict[str, torch.Tensor],
) -> None:
    mae = ChannelMAE(None).eval()
    fe = FeatureExtractor().eval()
    sample = {"feat": small_feature_dict}
    with pytest.raises(ValueError):
        compression_feedback_equiv(mae, fe, [sample], baseline="bogus")


def test_compression_feedback_equiv_svd_rank1(
    small_feature_dict: dict[str, torch.Tensor],
) -> None:
    mae = ChannelMAE(None).eval()
    fe = FeatureExtractor().eval()
    sample = {"feat": small_feature_dict, "feat_gt": small_feature_dict}
    out = compression_feedback_equiv(mae, fe, [sample], baseline="svd_rank1")
    assert np.isfinite(out["latent_sinr_dB"])
