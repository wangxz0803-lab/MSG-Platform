"""Shared fixtures for backend unit tests."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

sys.modules.setdefault("platform.backend.conftest", sys.modules[__name__])

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from platform.backend import db as backend_db  # noqa: E402
from platform.backend import settings as backend_settings  # noqa: E402
from platform.backend.db import Base  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402


@pytest.fixture
def tmp_layout(tmp_path: Path) -> dict[str, Path]:
    """Build a fake artifacts/reports/worker layout under tmp_path."""
    artifacts = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    worker_queue = tmp_path / "worker" / "queue"
    worker_logs = tmp_path / "worker" / "logs"
    worker_progress = tmp_path / "worker" / "progress"
    worker_cancel = tmp_path / "worker" / "cancel"
    configs = tmp_path / "configs"
    for p in (artifacts, reports, worker_queue, worker_logs, worker_progress, worker_cancel, configs):
        p.mkdir(parents=True, exist_ok=True)

    run_a = artifacts / "run_a"
    run_a.mkdir()
    (run_a / "metadata.json").write_text(
        json.dumps({"git_sha": "abc123", "tags": ["smoke", "cpu"]}), encoding="utf-8"
    )
    (run_a / "config.yaml").write_text(
        "train:\n  batch_size: 32\nmodel:\n  token_dim: 128\n", encoding="utf-8"
    )
    (run_a / "ckpt_best.pth").write_bytes(b"fake-best")
    (run_a / "ckpt_last.pth").write_bytes(b"fake-last")
    (run_a / "model.onnx").write_bytes(b"fake-onnx")

    run_b = artifacts / "run_b"
    run_b.mkdir()
    (run_b / "metadata.json").write_text(
        json.dumps({"git_sha": "def456", "tags": ["prod"]}), encoding="utf-8"
    )
    (run_b / "ckpt_best.pth").write_bytes(b"fake-best-b")

    (reports / "run_a").mkdir()
    (reports / "run_a" / "metrics.json").write_text(
        json.dumps({"mse": 0.12, "nmse_db": -8.4, "loss": 0.05}), encoding="utf-8"
    )
    (reports / "run_b").mkdir()
    (reports / "run_b" / "metrics.json").write_text(
        json.dumps({"mse": 0.09, "nmse_db": -9.1, "loss": 0.03}), encoding="utf-8"
    )

    (configs / "_schema.json").write_text(
        json.dumps({"type": "object", "properties": {"train": {"type": "object"}}}),
        encoding="utf-8",
    )

    return {
        "root": tmp_path,
        "artifacts": artifacts,
        "reports": reports,
        "worker_queue": worker_queue,
        "worker_logs": worker_logs,
        "worker_progress": worker_progress,
        "worker_cancel": worker_cancel,
        "configs": configs,
    }


@pytest.fixture
def fake_manifest(tmp_path: Path) -> Path | None:
    """Produce a small parquet manifest (skips when pyarrow is missing)."""
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "uuid": "u-1",
                "source": "quadriga_multi",
                "shard_id": 0,
                "sample_id": 1,
                "status": "ready",
                "link": "UL",
                "snr_dB": 10.0,
                "sir_dB": 5.0,
                "sinr_dB": 4.0,
                "num_cells": 3,
                "split": "train",
                "path": "/tmp/s1",
                "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
                "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
            },
            {
                "uuid": "u-2",
                "source": "quadriga_multi",
                "shard_id": 0,
                "sample_id": 2,
                "status": "ready",
                "link": "DL",
                "snr_dB": 20.0,
                "sir_dB": 15.0,
                "sinr_dB": 14.0,
                "num_cells": 5,
                "split": "val",
                "path": "/tmp/s2",
                "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
                "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
            },
            {
                "uuid": "u-3",
                "source": "sionna_rt",
                "shard_id": 0,
                "sample_id": 3,
                "status": "ready",
                "link": "UL",
                "snr_dB": 15.0,
                "sir_dB": None,
                "sinr_dB": None,
                "num_cells": 2,
                "split": "test",
                "path": "/tmp/s3",
                "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
                "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
            },
        ]
    )
    out = tmp_path / "manifest.parquet"
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out)
    return out


@pytest.fixture
def settings_override(
    tmp_layout: dict[str, Path],
    fake_manifest: Path | None,
) -> Iterator[backend_settings.BackendSettings]:
    """Install a BackendSettings instance pointing at the tmp layout."""
    cfg_overrides: dict[str, Any] = {
        "db_url": "sqlite:///:memory:",
        "artifacts_dir": str(tmp_layout["artifacts"]),
        "reports_dir": str(tmp_layout["reports"]),
        "data_dir": str(tmp_layout["root"]),
        "bridge_out_dir": str(tmp_layout["root"]),
        "worker_queue_dir": str(tmp_layout["worker_queue"]),
        "worker_logs_dir": str(tmp_layout["worker_logs"]),
        "worker_progress_dir": str(tmp_layout["worker_progress"]),
        "worker_cancel_dir": str(tmp_layout["worker_cancel"]),
        "configs_dir": str(tmp_layout["configs"]),
        "schema_path": str(tmp_layout["configs"] / "_schema.json"),
    }
    if fake_manifest is not None:
        cfg_overrides["manifest_path"] = str(fake_manifest)
    else:
        cfg_overrides["manifest_path"] = str(tmp_layout["root"] / "missing.parquet")

    new_settings = backend_settings.BackendSettings(**cfg_overrides)
    backend_settings.reset_settings(new_settings)
    backend_db.reset_engine()
    yield new_settings
    backend_settings.reset_settings(None)
    backend_db.reset_engine()


@pytest.fixture
def client(settings_override: backend_settings.BackendSettings) -> Iterator[TestClient]:
    """Provide a TestClient wired to an in-memory SQLite DB."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(bind=engine)
    testing_session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    backend_db._engine = engine  # type: ignore[assignment]
    backend_db._SessionLocal = testing_session  # type: ignore[assignment]

    from platform.backend.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c
    engine.dispose()
    backend_db.reset_engine()
