"""Thin DB-update layer shared by the worker and the backend.

Tries to import the backend's SQLAlchemy ``Job`` model; if that fails
falls back to raw SQL against the same SQLite file. Sessions are opened
and committed per call -- cross-process safe for SQLite in WAL mode.
"""

from __future__ import annotations

from contextlib import contextmanager, suppress
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .settings import get_settings

if TYPE_CHECKING:
    from collections.abc import Iterator

_ENGINE_CACHE: dict[str, Any] = {}
_SCHEMA_READY: set[str] = set()


def _get_engine(db_url: str | None = None):
    """Lazily create and cache a SQLAlchemy engine per URL."""
    from sqlalchemy import create_engine

    url = db_url or get_settings().db_url
    if url not in _ENGINE_CACHE:
        connect_args: dict[str, Any] = {}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        _ENGINE_CACHE[url] = create_engine(url, future=True, connect_args=connect_args)
    return _ENGINE_CACHE[url]


def reset_engine_cache() -> None:
    """Dispose and clear engines -- used in tests between DB URLs."""
    for eng in _ENGINE_CACHE.values():
        with suppress(Exception):
            eng.dispose()
    _ENGINE_CACHE.clear()
    _SCHEMA_READY.clear()


def _try_import_backend_job():
    """Return the backend ``Job`` ORM class, or None if unavailable."""
    try:
        from platform.backend.models.job import Job  # type: ignore[import-not-found]

        return Job
    except Exception:
        return None


def _backend_pk_column(Job: Any) -> str:
    """Return the primary-key column name on the backend ORM model."""
    mapper = Job.__mapper__
    pk_cols = mapper.primary_key
    return pk_cols[0].key if pk_cols else "job_id"


def _ensure_schema(engine: Any, Job: Any | None) -> None:
    """Make sure the table exists on the target engine."""
    key = str(engine.url)
    if key in _SCHEMA_READY:
        return
    if Job is not None:
        with suppress(Exception):
            Job.metadata.create_all(engine)
    else:
        from sqlalchemy import text

        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id TEXT PRIMARY KEY,
                        type TEXT,
                        status TEXT,
                        started_at TEXT,
                        finished_at TEXT,
                        progress_pct REAL DEFAULT 0.0,
                        current_step TEXT,
                        error_msg TEXT,
                        log_path TEXT,
                        display_name TEXT,
                        params_json TEXT
                    )
                    """
                )
            )
    _SCHEMA_READY.add(key)


@contextmanager
def _session(db_url: str | None = None, Job: Any | None = None) -> Iterator[Any]:
    """Yield a short-lived Session; commit on success, rollback on error."""
    from sqlalchemy.orm import Session

    engine = _get_engine(db_url)
    _ensure_schema(engine, Job)
    sess = Session(engine, future=True)
    try:
        yield sess
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


def _set_if_has(obj: Any, attr: str, value: Any) -> None:
    """``setattr`` only when the attribute exists on the ORM row."""
    if hasattr(obj, attr):
        setattr(obj, attr, value)


def _new_job_kwargs(Job: Any, job_id: str, status: str) -> dict[str, Any]:
    """Build minimal kwargs satisfying NOT NULL constraints on the ORM model."""
    pk = _backend_pk_column(Job)
    kwargs: dict[str, Any] = {pk: job_id, "status": status}
    mapper = Job.__mapper__
    for col in mapper.columns:
        if col.key in kwargs:
            continue
        if col.nullable or col.default is not None or col.server_default is not None:
            continue
        py_type = getattr(col.type, "python_type", str)
        try:
            kwargs[col.key] = py_type() if callable(py_type) else ""
        except Exception:
            kwargs[col.key] = ""
    return kwargs


def update_job_status(
    job_id: str,
    status: str,
    *,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    error_msg: str | None = None,
    log_path: str | None = None,
    progress_pct: float | None = None,
    db_url: str | None = None,
) -> None:
    """Update status/timestamps on a Job row. Creates the row if missing."""
    Job = _try_import_backend_job()
    with _session(db_url, Job) as sess:
        if Job is not None:
            row = sess.get(Job, job_id)
            if row is None:
                row = Job(**_new_job_kwargs(Job, job_id, status))
                sess.add(row)
            row.status = status
            if started_at is not None:
                _set_if_has(row, "started_at", started_at)
            if finished_at is not None:
                _set_if_has(row, "finished_at", finished_at)
            if error_msg is not None:
                _set_if_has(row, "error_msg", error_msg)
            if log_path is not None:
                _set_if_has(row, "log_path", log_path)
            if progress_pct is not None:
                _set_if_has(row, "progress_pct", progress_pct)
        else:
            from sqlalchemy import text

            conn = sess.connection()
            fields: dict[str, Any] = {"status": status}
            if started_at is not None:
                fields["started_at"] = started_at.isoformat()
            if finished_at is not None:
                fields["finished_at"] = finished_at.isoformat()
            if error_msg is not None:
                fields["error_msg"] = error_msg
            if log_path is not None:
                fields["log_path"] = log_path
            if progress_pct is not None:
                fields["progress_pct"] = progress_pct

            existing = conn.execute(
                text("SELECT job_id FROM jobs WHERE job_id=:id"), {"id": job_id}
            ).fetchone()
            if existing is None:
                cols = ["job_id", *fields.keys()]
                placeholders = ", ".join(f":{c}" for c in cols)
                conn.execute(
                    text(f"INSERT INTO jobs ({', '.join(cols)}) VALUES ({placeholders})"),
                    {"job_id": job_id, **fields},
                )
            else:
                set_clause = ", ".join(f"{k}=:{k}" for k in fields)
                conn.execute(
                    text(f"UPDATE jobs SET {set_clause} WHERE job_id=:id"),
                    {"id": job_id, **fields},
                )


def update_job_progress(
    job_id: str,
    pct: float,
    step: str | None = None,
    *,
    db_url: str | None = None,
) -> None:
    """Update progress_pct / current_step only. Does not touch status."""
    Job = _try_import_backend_job()
    with _session(db_url, Job) as sess:
        if Job is not None:
            row = sess.get(Job, job_id)
            if row is None:
                row = Job(**_new_job_kwargs(Job, job_id, "running"))
                _set_if_has(row, "progress_pct", pct)
                sess.add(row)
            else:
                _set_if_has(row, "progress_pct", pct)
            if step is not None:
                _set_if_has(row, "current_step", step)
        else:
            from sqlalchemy import text

            conn = sess.connection()
            existing = conn.execute(
                text("SELECT job_id FROM jobs WHERE job_id=:id"), {"id": job_id}
            ).fetchone()
            fields: dict[str, Any] = {"progress_pct": pct}
            if step is not None:
                fields["current_step"] = step
            if existing is None:
                fields["status"] = "running"
                cols = ["job_id", *fields.keys()]
                placeholders = ", ".join(f":{c}" for c in cols)
                conn.execute(
                    text(f"INSERT INTO jobs ({', '.join(cols)}) VALUES ({placeholders})"),
                    {"job_id": job_id, **fields},
                )
            else:
                set_clause = ", ".join(f"{k}=:{k}" for k in fields)
                conn.execute(
                    text(f"UPDATE jobs SET {set_clause} WHERE job_id=:id"),
                    {"id": job_id, **fields},
                )


def update_job_run_id(
    job_id: str,
    run_id: str,
    *,
    db_url: str | None = None,
) -> None:
    """Set the ``run_id`` on a Job row (linking Job to Run)."""
    Job = _try_import_backend_job()
    with _session(db_url, Job) as sess:
        if Job is not None:
            row = sess.get(Job, job_id)
            if row is not None:
                _set_if_has(row, "run_id", run_id)
        else:
            from sqlalchemy import text

            conn = sess.connection()
            conn.execute(
                text("UPDATE jobs SET run_id=:rid WHERE job_id=:jid"),
                {"rid": run_id, "jid": job_id},
            )


def get_job_status(job_id: str, *, db_url: str | None = None) -> str | None:
    """Look up the current status -- returns None if the row is missing."""
    Job = _try_import_backend_job()
    with _session(db_url, Job) as sess:
        if Job is not None:
            row = sess.get(Job, job_id)
            return row.status if row is not None else None
        from sqlalchemy import text

        conn = sess.connection()
        row = conn.execute(
            text("SELECT status FROM jobs WHERE job_id=:id"), {"id": job_id}
        ).fetchone()
        return row[0] if row is not None else None
