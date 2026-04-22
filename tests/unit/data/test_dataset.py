"""Tests for :mod:`msg_embedding.data.dataset`."""

from __future__ import annotations

import pytest

pytest.importorskip("pyarrow")
torch = pytest.importorskip("torch")

from msg_embedding.data.dataset import ChannelDataset  # noqa: E402
from msg_embedding.data.manifest import Manifest  # noqa: E402


def test_len_matches_filtered_manifest(tmp_path, make_sample, write_sample_pt) -> None:
    m = Manifest(tmp_path / "m.parquet")
    rows = []
    for i in range(5):
        sample = make_sample(source="quadriga_multi")
        path = write_sample_pt(sample, f"s{i}.pt")
        rows.append(
            {
                "uuid": sample.sample_id,
                "source": sample.source,
                "link": sample.link,
                "snr_dB": sample.snr_dB,
                "sinr_dB": sample.sinr_dB,
                "serving_cell_id": sample.serving_cell_id,
                "channel_est_mode": sample.channel_est_mode,
                "path": str(path),
                "status": "succeeded",
                "split": "train",
            }
        )
    m.append(rows)

    ds = ChannelDataset(m, split="train")
    assert len(ds) == 5


def test_source_filter(tmp_path, make_sample, write_sample_pt) -> None:
    m = Manifest(tmp_path / "m.parquet")
    rows = []
    for i, src in enumerate(["quadriga_multi", "sionna_rt", "quadriga_multi", "internal_sim"]):
        sample = make_sample(source=src)
        path = write_sample_pt(sample, f"s{i}.pt")
        rows.append(
            {
                "uuid": sample.sample_id,
                "source": sample.source,
                "link": sample.link,
                "path": str(path),
                "status": "succeeded",
                "split": "train",
            }
        )
    m.append(rows)

    ds_all = ChannelDataset(m, split="train", source_filter=None)
    assert len(ds_all) == 4

    ds_multi = ChannelDataset(m, split="train", source_filter=["quadriga_multi"])
    assert len(ds_multi) == 2

    ds_two = ChannelDataset(m, split="train", source_filter=["quadriga_multi", "sionna_rt"])
    assert len(ds_two) == 3


def test_link_filter(tmp_path, make_sample, write_sample_pt) -> None:
    m = Manifest(tmp_path / "m.parquet")
    rows = []
    for i, link in enumerate(["UL", "DL", "UL", "DL", "UL"]):
        sample = make_sample(link=link)
        path = write_sample_pt(sample, f"s{i}.pt")
        rows.append(
            {
                "uuid": sample.sample_id,
                "source": sample.source,
                "link": sample.link,
                "path": str(path),
                "status": "succeeded",
                "split": "train",
            }
        )
    m.append(rows)

    assert len(ChannelDataset(m, split="train", link_filter="both")) == 5
    assert len(ChannelDataset(m, split="train", link_filter="UL")) == 3
    assert len(ChannelDataset(m, split="train", link_filter="DL")) == 2


def test_getitem_returns_tensor_payload(tmp_path, make_sample, write_sample_pt) -> None:
    m = Manifest(tmp_path / "m.parquet")
    sample = make_sample()
    path = write_sample_pt(sample)
    m.append(
        [
            {
                "uuid": sample.sample_id,
                "source": sample.source,
                "link": sample.link,
                "snr_dB": sample.snr_dB,
                "sinr_dB": sample.sinr_dB,
                "path": str(path),
                "status": "succeeded",
                "split": "train",
            }
        ]
    )

    ds = ChannelDataset(m, split="train")
    record = ds[0]

    assert record["uuid"] == sample.sample_id
    assert isinstance(record["h_true"], torch.Tensor)
    assert record["h_true"].shape == sample.h_serving_true.shape
    assert record["link"] == sample.link
    assert pytest.approx(record["sinr_dB"], rel=1e-5) == float(sample.sinr_dB)


def test_transform_is_invoked(tmp_path, make_sample, write_sample_pt) -> None:
    m = Manifest(tmp_path / "m.parquet")
    sample = make_sample()
    path = write_sample_pt(sample)
    m.append(
        [
            {
                "uuid": sample.sample_id,
                "source": sample.source,
                "link": sample.link,
                "path": str(path),
                "status": "succeeded",
                "split": "train",
            }
        ]
    )

    def _transform(record: dict) -> dict:
        record["transformed"] = True
        return record

    ds = ChannelDataset(m, split="train", transform=_transform)
    record = ds[0]
    assert record.get("transformed") is True


def test_index_out_of_range(tmp_path, make_sample, write_sample_pt) -> None:
    m = Manifest(tmp_path / "m.parquet")
    sample = make_sample()
    path = write_sample_pt(sample)
    m.append(
        [
            {
                "uuid": sample.sample_id,
                "source": sample.source,
                "link": sample.link,
                "path": str(path),
                "status": "succeeded",
                "split": "train",
            }
        ]
    )
    ds = ChannelDataset(m, split="train")
    with pytest.raises(IndexError):
        _ = ds[10]


def test_filters_out_failed_and_pending_rows(tmp_path, make_sample, write_sample_pt) -> None:
    m = Manifest(tmp_path / "m.parquet")
    # Successful sample on disk
    ok_sample = make_sample()
    ok_path = write_sample_pt(ok_sample, "ok.pt")
    # Failed row — no path even though a uuid is present
    bad_sample = make_sample()
    m.append(
        [
            {
                "uuid": ok_sample.sample_id,
                "source": ok_sample.source,
                "link": ok_sample.link,
                "path": str(ok_path),
                "status": "succeeded",
                "split": "train",
            },
            {
                "uuid": bad_sample.sample_id,
                "source": bad_sample.source,
                "link": bad_sample.link,
                "path": None,
                "status": "failed",
                "split": "train",
            },
        ]
    )
    ds = ChannelDataset(m, split="train")
    assert len(ds) == 1
