"""Tests for :mod:`msg_embedding.data.parallel`.

Worker functions are defined at the top level so they pickle cleanly into
``ProcessPoolExecutor`` workers.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("pyarrow")

# The project has a platform/ directory that shadows the stdlib 'platform'
# module in subprocess workers. Skip these tests when that collision exists.
_platform_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "platform"
)
_has_platform_collision = os.path.isdir(_platform_dir)
pytestmark = pytest.mark.skipif(
    _has_platform_collision,
    reason="platform/ directory shadows stdlib platform module in subprocess workers",
)

from msg_embedding.data.manifest import Manifest  # noqa: E402
from msg_embedding.data.parallel import FAILURE_SCHEMA, parallel_process  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level workers (picklable)
# ---------------------------------------------------------------------------


def _success_worker(item: dict) -> tuple[str, dict]:
    return item["uuid"], {"status": "succeeded", "path": f"/out/{item['uuid']}.pt"}


def _failing_worker(item: dict) -> tuple[str, dict]:
    if item.get("fail", False):
        raise RuntimeError(f"simulated failure for {item['uuid']}")
    return item["uuid"], {"status": "succeeded", "path": f"/out/{item['uuid']}.pt"}


def _always_fail(item: dict) -> tuple[str, dict]:
    raise ValueError("boom")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_items_returns_zero(tmp_path) -> None:
    m = Manifest(tmp_path / "m.parquet")
    succeeded, failed = parallel_process([], _success_worker, m, num_workers=2)
    assert succeeded == 0 and failed == 0


def test_all_success(tmp_path, manifest_row_factory) -> None:
    m = Manifest(tmp_path / "m.parquet")
    rows = [manifest_row_factory(status="pending", path=None) for _ in range(6)]
    m.append(rows)
    items = [{"uuid": r["uuid"]} for r in rows]

    succeeded, failed = parallel_process(
        items,
        _success_worker,
        m,
        num_workers=2,
        chunk_size=2,
    )
    assert succeeded == 6
    assert failed == 0

    statuses = set(m.df["status"].astype(str))
    assert statuses == {"succeeded"}
    # All paths populated
    assert m.df["path"].notna().all()


def test_partial_failure_isolates_bad_samples(tmp_path, manifest_row_factory) -> None:
    m = Manifest(tmp_path / "m.parquet")
    rows = [manifest_row_factory(status="pending") for _ in range(5)]
    m.append(rows)

    items = []
    for i, r in enumerate(rows):
        items.append({"uuid": r["uuid"], "fail": i % 2 == 0})
    n_expected_fail = sum(1 for it in items if it["fail"])

    succeeded, failed = parallel_process(
        items,
        _failing_worker,
        m,
        num_workers=2,
        chunk_size=2,
    )
    assert succeeded == len(items) - n_expected_fail
    assert failed == n_expected_fail

    # Manifest reflects per-row status.
    status_counts = m.df["status"].astype(str).value_counts().to_dict()
    assert status_counts.get("failed", 0) == n_expected_fail
    assert status_counts.get("succeeded", 0) == len(items) - n_expected_fail

    # failed rows carry an error_msg.
    failed_df = m.df[m.df["status"].astype(str) == "failed"]
    assert failed_df["error_msg"].notna().all()


def test_failures_parquet_is_written(tmp_path, manifest_row_factory) -> None:
    m = Manifest(tmp_path / "m.parquet")
    rows = [manifest_row_factory(status="pending") for _ in range(3)]
    m.append(rows)
    items = [{"uuid": r["uuid"]} for r in rows]

    succeeded, failed = parallel_process(
        items,
        _always_fail,
        m,
        num_workers=2,
        chunk_size=2,
    )
    assert succeeded == 0 and failed == 3

    failures_path = tmp_path / "failures.parquet"
    assert failures_path.exists()

    import pyarrow.parquet as pq

    table = pq.read_table(failures_path)
    assert table.num_rows == 3
    assert set(table.schema.names) == set(FAILURE_SCHEMA.names)


def test_invalid_num_workers(tmp_path) -> None:
    m = Manifest(tmp_path / "m.parquet")
    with pytest.raises(ValueError):
        parallel_process([1], _success_worker, m, num_workers=0)


def test_invalid_chunk_size(tmp_path) -> None:
    m = Manifest(tmp_path / "m.parquet")
    with pytest.raises(ValueError):
        parallel_process([1], _success_worker, m, num_workers=1, chunk_size=0)


def test_manifest_saved_on_completion(tmp_path, manifest_row_factory) -> None:
    path = tmp_path / "m.parquet"
    m = Manifest(path)
    rows = [manifest_row_factory(status="pending") for _ in range(4)]
    m.append(rows)
    items = [{"uuid": r["uuid"]} for r in rows]
    parallel_process(items, _success_worker, m, num_workers=2, chunk_size=2)

    # Reload from disk to confirm save() was invoked.
    m2 = Manifest(path)
    succeeded = m2.df[m2.df["status"].astype(str) == "succeeded"]
    assert len(succeeded) == 4
