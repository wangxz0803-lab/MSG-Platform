"""Job dispatch service -- persist Job row + run via background thread.

Primary path: spawn TaskRunner directly in a daemon thread (no Redis needed).
Fallback: write a JSON file to the queue directory for external workers.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from msg_embedding.core.logging import get_logger

from ..models.job import Job
from ..settings import get_settings

_log = get_logger(__name__)

VALID_JOB_TYPES = frozenset(
    {"simulate", "convert", "bridge", "train", "eval", "infer", "export", "report"}
)


def _run_in_thread(job_id: str, job_type: str, overrides: dict[str, Any]) -> None:
    """Execute a job via TaskRunner in the current thread (called as daemon)."""
    try:
        from platform.worker.tasks.base import TaskRunner

        result = TaskRunner(job_id, job_type, overrides).run()
        _log.info("thread_job_finished", job_id=job_id, result=result)
    except Exception as exc:
        _log.exception("thread_job_failed", job_id=job_id, error=str(exc))
        try:
            from platform.worker.db_update import update_job_status

            update_job_status(
                job_id,
                "failed",
                finished_at=datetime.now(UTC),
                error_msg=str(exc)[:500],
            )
        except Exception:
            pass


def _spawn_job_thread(
    job_id: str, job_type: str, overrides: dict[str, Any]
) -> bool:
    """Start a daemon thread that runs the job. Returns True on success."""
    try:
        t = threading.Thread(
            target=_run_in_thread,
            args=(job_id, job_type, overrides),
            daemon=True,
            name=f"job-{job_id[:8]}",
        )
        t.start()
        _log.info("job_thread_started", job_id=job_id, job_type=job_type)
        return True
    except Exception as exc:
        _log.error("job_thread_start_failed", job_id=job_id, error=str(exc))
        return False


def _write_queue_file(
    job_id: str,
    job_type: str,
    overrides: dict[str, Any],
    display_name: str | None,
) -> None:
    """Fallback: write a JSON file for an external queue_watcher to pick up."""
    settings = get_settings()
    queue_file = settings.worker_queue_path / f"{job_id}.json"
    payload = {
        "job_id": job_id,
        "type": job_type,
        "config_overrides": overrides,
        "display_name": display_name,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    try:
        queue_file.write_text(json.dumps(payload), encoding="utf-8")
        _log.info("queue_file_fallback_ok", job_id=job_id, path=str(queue_file))
    except OSError as exc:
        _log.error("queue_file_write_failed", path=str(queue_file), error=str(exc))


def new_job_id() -> str:
    """Return a fresh uuid4 hex string."""
    return uuid.uuid4().hex


def dispatch_job(
    session: Session,
    job_type: str,
    config_overrides: dict[str, Any] | None = None,
    display_name: str | None = None,
    explicit_job_id: str | None = None,
) -> Job:
    """Create a Job row and run it in a background thread.

    Raises ``ValueError`` on unknown job types.
    """
    if job_type not in VALID_JOB_TYPES:
        raise ValueError(f"unknown job type: {job_type!r}")

    job_id = explicit_job_id or new_job_id()
    overrides = dict(config_overrides or {})

    settings = get_settings()
    settings.ensure_dirs()

    log_path = settings.worker_logs_path / f"{job_id}.log"
    job = Job(
        job_id=job_id,
        type=job_type,
        status="queued",
        display_name=display_name,
        created_at=datetime.now(UTC),
        progress_pct=0.0,
        params_json=json.dumps(overrides),
        log_path=str(log_path),
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    started = _spawn_job_thread(job_id, job_type, overrides)
    if not started:
        _log.warning("thread_start_failed_using_file_fallback", job_id=job_id)
        _write_queue_file(job_id, job_type, overrides, display_name)

    _log.info(
        "job_dispatched",
        job_id=job_id,
        job_type=job_type,
        via="thread" if started else "file",
    )
    return job


def dispatch_batch_jobs(
    session: Session,
    job_type: str,
    configs: list[dict[str, Any]],
    display_name_prefix: str | None = None,
) -> list[Job]:
    """Dispatch multiple jobs from a list of config overrides."""
    jobs = []
    for i, cfg in enumerate(configs):
        suffix = f"-{i + 1}" if len(configs) > 1 else ""
        name = f"{display_name_prefix}{suffix}" if display_name_prefix else None
        job = dispatch_job(session, job_type, config_overrides=cfg, display_name=name)
        jobs.append(job)
    return jobs


def cancel_job(session: Session, job_id: str) -> Job | None:
    """Mark job as cancelled and drop a cancel flag for the worker to observe."""
    job = session.get(Job, job_id)
    if job is None:
        return None

    settings = get_settings()
    settings.ensure_dirs()
    flag = settings.worker_cancel_path / f"{job_id}.flag"
    try:
        flag.write_text(datetime.now(UTC).isoformat(), encoding="utf-8")
    except OSError as exc:
        _log.error("cancel_flag_write_failed", path=str(flag), error=str(exc))

    job.status = "cancelled"
    job.finished_at = datetime.now(UTC)
    session.commit()
    session.refresh(job)
    return job


def read_progress(job_id: str) -> dict[str, Any]:
    """Read worker-produced progress file ``progress/<job_id>.json`` if present."""
    settings = get_settings()
    path = settings.worker_progress_path / f"{job_id}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _log.warning("progress_file_parse_failed", path=str(path), error=str(exc))
        return {}


def read_log_tail(job_id: str, tail: int = 500) -> list[str]:
    """Return the last ``tail`` lines of the worker log for ``job_id``."""
    settings = get_settings()
    path = settings.worker_logs_path / f"{job_id}.log"
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return []
    if tail > 0 and len(lines) > tail:
        lines = lines[-tail:]
    return [line.rstrip("\n") for line in lines]


def deserialize_overrides(params_json: str | None) -> dict[str, Any]:
    """Safely turn the stored JSON overrides blob back into a dict."""
    if not params_json:
        return {}
    try:
        parsed = json.loads(params_json)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
