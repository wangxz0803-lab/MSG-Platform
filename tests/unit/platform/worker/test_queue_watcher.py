"""Tests for the file-drop queue watcher."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("dramatiq")
pytest.importorskip("sqlalchemy")


def test_process_once_empty(worker_settings):
    from platform.worker.queue_watcher import process_once

    results = process_once(settings=worker_settings, actors_map={})
    assert results == []


def test_process_once_enqueues_job(worker_settings, stub_broker):
    from platform.worker.actors import ACTORS
    from platform.worker.queue_watcher import process_once

    queue_dir = Path(worker_settings.queue_dir)
    payload = {
        "job_id": "test-j1",
        "type": "eval",
        "config_overrides": {"eval.batch": 8},
        "display_name": "qw-test",
    }
    (queue_dir / "test-j1.json").write_text(json.dumps(payload), encoding="utf-8")

    results = process_once(settings=worker_settings, actors_map=ACTORS)
    assert len(results) == 1
    assert results[0]["enqueued"] is True
    assert results[0]["job_id"] == "test-j1"

    processed = Path(worker_settings.processed_dir)
    assert list(processed.glob("*.json"))


def test_process_once_malformed_json(worker_settings):
    from platform.worker.queue_watcher import process_once

    queue_dir = Path(worker_settings.queue_dir)
    (queue_dir / "bad.json").write_text("not json!", encoding="utf-8")

    results = process_once(settings=worker_settings, actors_map={})
    assert len(results) == 1
    assert results[0]["enqueued"] is False
    assert results[0]["error"] == "malformed"
