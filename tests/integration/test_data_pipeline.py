"""Integration: data source -> ChannelSample -> bridge -> shard -> dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from msg_embedding.data.bridge import _build_feat_dict
from msg_embedding.data.contract import ChannelSample


def test_channel_sample_serialization_roundtrip(synthetic_samples) -> None:
    """to_dict -> from_dict should preserve all fields."""
    for sample in synthetic_samples[:3]:
        d = sample.to_dict()
        restored = ChannelSample.from_dict(d)
        assert restored.link == sample.link
        assert restored.source == sample.source
        assert abs(restored.snr_dB - sample.snr_dB) < 1e-4
        np.testing.assert_allclose(
            np.abs(restored.h_serving_true),
            np.abs(sample.h_serving_true),
            atol=1e-5,
        )


def test_channel_sample_torch_save_load(synthetic_samples, tmp_path: Path) -> None:
    """Save/load via torch.save preserves ChannelSample."""
    sample = synthetic_samples[0]
    path = tmp_path / "sample.pt"
    torch.save(sample.to_dict(), path)

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    restored = ChannelSample.from_dict(loaded)
    assert restored.sample_id == sample.sample_id
    assert restored.link == sample.link


def test_bridge_produces_shardable_output(synthetic_samples, tmp_path: Path) -> None:
    """Bridge output can be stacked into a shard and saved/loaded."""
    feat_dicts = []
    for s in synthetic_samples:
        fd, _ = _build_feat_dict(s, use_legacy_pmi=False)
        feat_dicts.append(fd)

    stacked = {}
    for key in feat_dicts[0]:
        stacked[key] = torch.cat([f[key] for f in feat_dicts], dim=0)

    shard_path = tmp_path / "shard_0000.pt"
    torch.save(stacked, shard_path)

    loaded = torch.load(shard_path, map_location="cpu", weights_only=False)
    for key in stacked:
        assert key in loaded
        assert loaded[key].shape == stacked[key].shape
        assert torch.allclose(loaded[key], stacked[key])


def test_multiple_sources_produce_compatible_features(synthetic_samples) -> None:
    """Samples with different links/sources should produce same token shape."""
    from msg_embedding.features.extractor import FeatureExtractor

    feat_ext = FeatureExtractor()
    shapes = []

    for s in synthetic_samples:
        fd, _ = _build_feat_dict(s, use_legacy_pmi=False)
        with torch.no_grad():
            tokens, _ = feat_ext(fd)
        shapes.append(tokens.shape)

    for shape in shapes:
        assert shape[1] == 16, f"Expected 16 tokens, got {shape[1]}"
        assert shape[2] == 128, f"Expected 128-dim tokens, got {shape[2]}"
