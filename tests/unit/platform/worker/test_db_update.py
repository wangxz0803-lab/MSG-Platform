"""Tests for worker db_update module."""

from __future__ import annotations

import pytest

pytest.importorskip("dramatiq")
pytest.importorskip("sqlalchemy")


def test_update_and_get_status(worker_settings):
    from platform.worker.db_update import get_job_status, update_job_status

    assert get_job_status("j1", db_url=worker_settings.db_url) is None
    update_job_status("j1", "running", db_url=worker_settings.db_url)
    assert get_job_status("j1", db_url=worker_settings.db_url) == "running"
    update_job_status("j1", "completed", db_url=worker_settings.db_url)
    assert get_job_status("j1", db_url=worker_settings.db_url) == "completed"


def test_update_progress(worker_settings):
    from platform.worker.db_update import update_job_progress, update_job_status

    update_job_status("j2", "running", db_url=worker_settings.db_url)
    update_job_progress("j2", 50.0, "epoch_5", db_url=worker_settings.db_url)


def test_update_run_id(worker_settings):
    from platform.worker.db_update import update_job_run_id, update_job_status

    update_job_status("j3", "running", db_url=worker_settings.db_url)
    update_job_run_id("j3", "run_abc", db_url=worker_settings.db_url)
