"""Shared fixtures for worker tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.modules.setdefault("platform.worker.conftest", sys.modules[__name__])

pytest.importorskip("dramatiq")
pytest.importorskip("sqlalchemy")


@pytest.fixture
def worker_settings(tmp_path: Path):
    """Isolated WorkerSettings with per-test dirs + a fresh SQLite file."""
    from platform.worker import settings as settings_mod

    queue = tmp_path / "queue"
    processed = queue / ".processed"
    logs = tmp_path / "logs"
    progress = tmp_path / "progress"
    cancel = tmp_path / "cancel"
    db_file = tmp_path / "msg.db"

    s = settings_mod.WorkerSettings(
        redis_url="redis://localhost:6379/15",
        queue_dir=str(queue),
        processed_dir=str(processed),
        log_dir=str(logs),
        progress_dir=str(progress),
        cancel_dir=str(cancel),
        db_url=f"sqlite:///{db_file.as_posix()}",
        worker_poll_interval_secs=0.05,
        queue_poll_interval_secs=0.05,
        cancel_grace_secs=1.0,
    )
    s.ensure_dirs()
    settings_mod.reset_settings(s)

    from platform.worker import db_update

    db_update.reset_engine_cache()

    yield s

    settings_mod.reset_settings(None)
    db_update.reset_engine_cache()


@pytest.fixture
def stub_broker():
    """Install a StubBroker for the test."""
    import dramatiq
    from dramatiq.brokers.stub import StubBroker

    broker = StubBroker()
    broker.emit_after("process_boot")
    dramatiq.set_broker(broker)

    import platform.worker.actors as actors_mod

    for actor in actors_mod.ACTORS.values():
        actor.broker = broker
        broker.declare_actor(actor)
        broker.declare_queue(actor.queue_name)

    yield broker
    broker.flush_all()
    broker.close()
