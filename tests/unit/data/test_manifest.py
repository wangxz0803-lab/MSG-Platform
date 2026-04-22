"""Tests for :mod:`msg_embedding.data.manifest`."""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("pyarrow")

from msg_embedding.data.manifest import (  # noqa: E402
    COLUMNS,
    MANIFEST_SCHEMA,
    Manifest,
    compute_content_hash,
)


def test_schema_has_all_phase6_columns() -> None:
    expected = {
        "uuid",
        "job_id",
        "run_id",
        "source",
        "shard_id",
        "sample_id",
        "stage",
        "status",
        "link",
        "snr_dB",
        "sir_dB",
        "sinr_dB",
        "num_cells",
        "serving_cell_id",
        "ue_x",
        "ue_y",
        "ue_z",
        "channel_est_mode",
        "split",
        "hash",
        "path",
        "error_msg",
        "created_at",
        "updated_at",
    }
    assert set(COLUMNS) == expected
    assert {f.name for f in MANIFEST_SCHEMA} == expected


def test_new_manifest_is_empty(tmp_path) -> None:
    m = Manifest(tmp_path / "manifest.parquet")
    assert len(m) == 0
    assert list(m.df.columns) == COLUMNS


def test_append_and_query(tmp_path, manifest_row_factory) -> None:
    m = Manifest(tmp_path / "m.parquet")
    rows = [manifest_row_factory(source="sionna_rt") for _ in range(3)]
    rows += [manifest_row_factory(source="quadriga_multi") for _ in range(2)]
    m.append(rows)
    assert len(m) == 5

    sionna = m.query(source="sionna_rt")
    assert len(sionna) == 3

    multi_src = m.query(source=["sionna_rt", "quadriga_multi"])
    assert len(multi_src) == 5

    with pytest.raises(ValueError):
        m.query(nonexistent="x")


def test_append_is_idempotent_by_uuid(tmp_path, manifest_row_factory) -> None:
    m = Manifest(tmp_path / "m.parquet")
    row = manifest_row_factory()
    m.append([row])
    m.append([row, row])  # should skip duplicates
    assert len(m) == 1


def test_update_patches_row_and_bumps_updated_at(tmp_path, manifest_row_factory) -> None:
    m = Manifest(tmp_path / "m.parquet")
    row = manifest_row_factory(status="pending")
    m.append([row])
    before = m.df.loc[0, "updated_at"]

    m.update(row["uuid"], status="succeeded", path="/new/path.pt")
    assert m.df.loc[0, "status"] == "succeeded"
    assert m.df.loc[0, "path"] == "/new/path.pt"
    assert m.df.loc[0, "updated_at"] >= before


def test_update_missing_uuid_raises(tmp_path) -> None:
    m = Manifest(tmp_path / "m.parquet")
    with pytest.raises(KeyError):
        m.update(str(uuid.uuid4()), status="succeeded")


def test_update_unknown_column_raises(tmp_path, manifest_row_factory) -> None:
    m = Manifest(tmp_path / "m.parquet")
    row = manifest_row_factory()
    m.append([row])
    with pytest.raises(ValueError):
        m.update(row["uuid"], nonexistent_col=1)


def test_save_load_roundtrip(tmp_path, manifest_row_factory) -> None:
    path = tmp_path / "m.parquet"
    m = Manifest(path)
    rows = [manifest_row_factory() for _ in range(4)]
    m.append(rows)
    m.save()

    assert path.exists()

    m2 = Manifest(path)
    assert len(m2) == 4
    assert set(m2.df["uuid"].astype(str)) == {r["uuid"] for r in rows}


def test_save_preserves_schema_dtypes(tmp_path, manifest_row_factory) -> None:
    path = tmp_path / "m.parquet"
    m = Manifest(path)
    m.append([manifest_row_factory()])
    m.save()

    import pyarrow.parquet as pq

    table = pq.read_table(path)
    assert table.schema.field("uuid").type.equals(MANIFEST_SCHEMA.field("uuid").type)
    assert table.schema.field("snr_dB").type.equals(MANIFEST_SCHEMA.field("snr_dB").type)
    assert table.schema.field("created_at").type.equals(MANIFEST_SCHEMA.field("created_at").type)


def test_compute_split_random(tmp_path, manifest_row_factory) -> None:
    m = Manifest(tmp_path / "m.parquet")
    m.append([manifest_row_factory() for _ in range(100)])
    m.compute_split(strategy="random", seed=42)
    counts = m.df["split"].value_counts().to_dict()
    # Determinism
    assert sum(counts.values()) == 100
    assert set(counts).issubset({"train", "val", "test"})
    # Re-running with same seed produces the same assignment
    first_split = m.df["split"].tolist()
    m.compute_split(strategy="random", seed=42)
    assert m.df["split"].tolist() == first_split


def test_compute_split_by_position_keeps_groups_together(tmp_path, manifest_row_factory) -> None:
    m = Manifest(tmp_path / "m.parquet")
    rows: list[dict] = []
    # Three geographic clusters, each with multiple samples at the same location.
    for cluster_idx, (x, y) in enumerate([(0.0, 0.0), (50.0, 50.0), (-30.0, 30.0)]):
        for _ in range(10):
            rows.append(
                manifest_row_factory(
                    ue_x=x,
                    ue_y=y,
                    serving_cell_id=cluster_idx,
                )
            )
    m.append(rows)
    m.compute_split(strategy="by_position", seed=1)

    # Each cluster must be in exactly one split.
    for x, y in [(0.0, 0.0), (50.0, 50.0), (-30.0, 30.0)]:
        sub = m.df[(m.df["ue_x"] == x) & (m.df["ue_y"] == y)]
        assert (
            sub["split"].nunique() == 1
        ), f"cluster at ({x},{y}) split across {sub['split'].unique()}"


def test_compute_split_by_beam(tmp_path, manifest_row_factory) -> None:
    m = Manifest(tmp_path / "m.parquet")
    rows: list[dict] = []
    for beam in range(5):
        for _ in range(4):
            rows.append(manifest_row_factory(serving_cell_id=beam))
    m.append(rows)
    m.compute_split(strategy="by_beam", seed=7, ratios=(0.6, 0.2, 0.2))

    for beam in range(5):
        sub = m.df[m.df["serving_cell_id"] == beam]
        assert sub["split"].nunique() == 1


def test_compute_split_rejects_unknown_strategy(tmp_path, manifest_row_factory) -> None:
    m = Manifest(tmp_path / "m.parquet")
    m.append([manifest_row_factory()])
    with pytest.raises(ValueError):
        m.compute_split(strategy="magic", seed=0)  # type: ignore[arg-type]


def test_compute_split_rejects_bad_ratios(tmp_path, manifest_row_factory) -> None:
    m = Manifest(tmp_path / "m.parquet")
    m.append([manifest_row_factory()])
    with pytest.raises(ValueError):
        m.compute_split(strategy="random", seed=0, ratios=(0.5, 0.3, 0.3))


def test_to_sqlite(tmp_path, manifest_row_factory) -> None:
    m = Manifest(tmp_path / "m.parquet")
    m.append([manifest_row_factory() for _ in range(3)])
    db_path = tmp_path / "manifest.db"
    m.to_sqlite(db_path)
    assert db_path.exists()

    import sqlite3

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM manifest")
        (count,) = cursor.fetchone()
        assert count == 3
        # Check index exists
        idx = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_manifest_uuid'"
        ).fetchone()
        assert idx is not None


def test_compute_content_hash_stable(tmp_path) -> None:
    p = tmp_path / "blob.bin"
    p.write_bytes(b"MSG-Embedding phase 1.6")
    h1 = compute_content_hash(p)
    h2 = compute_content_hash(p)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex

    p2 = tmp_path / "blob2.bin"
    p2.write_bytes(b"different")
    assert compute_content_hash(p2) != h1


def test_append_empty_is_noop(tmp_path) -> None:
    m = Manifest(tmp_path / "m.parquet")
    m.append([])
    assert len(m) == 0
