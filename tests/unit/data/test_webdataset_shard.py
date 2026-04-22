"""Tests for :mod:`msg_embedding.data.webdataset_shard`."""

from __future__ import annotations

import numpy as np
import pytest

from msg_embedding.data.webdataset_shard import pack_shard, stream_shard


def test_pack_then_stream_roundtrip(tmp_path, make_sample) -> None:
    samples = [make_sample(serving_cell_id=i) for i in range(5)]
    shards = pack_shard(samples, tmp_path, shard_size=1000)
    assert len(shards) == 1
    assert shards[0].exists()

    recovered = list(stream_shard(shards))
    assert len(recovered) == len(samples)
    for orig, rec in zip(samples, recovered, strict=False):
        assert rec.sample_id == orig.sample_id
        assert rec.source == orig.source
        np.testing.assert_array_equal(rec.h_serving_true, orig.h_serving_true)


def test_pack_splits_into_multiple_shards(tmp_path, make_sample) -> None:
    samples = [make_sample(serving_cell_id=i) for i in range(7)]
    shards = pack_shard(samples, tmp_path, shard_size=3)
    assert len(shards) == 3  # 3 + 3 + 1

    recovered = list(stream_shard(shards))
    assert len(recovered) == 7


def test_pack_preserves_order(tmp_path, make_sample) -> None:
    samples = [make_sample(serving_cell_id=i) for i in range(6)]
    shards = pack_shard(samples, tmp_path, shard_size=2)
    recovered = list(stream_shard(shards))
    assert [r.sample_id for r in recovered] == [s.sample_id for s in samples]


def test_pack_invalid_shard_size(tmp_path, make_sample) -> None:
    with pytest.raises(ValueError):
        pack_shard([make_sample()], tmp_path, shard_size=0)


def test_output_path_can_be_explicit_file_prefix(tmp_path, make_sample) -> None:
    samples = [make_sample() for _ in range(2)]
    # If user passes a .tar path, we use its stem as the prefix.
    output = tmp_path / "my-shards.tar"
    shards = pack_shard(samples, output, shard_size=10)
    assert len(shards) == 1
    assert shards[0].name.startswith("my-shards-")


def test_stream_empty_dir(tmp_path) -> None:
    # streaming zero shard paths yields nothing and doesn't raise
    assert list(stream_shard([])) == []
