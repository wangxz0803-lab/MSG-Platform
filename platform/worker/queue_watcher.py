"""File-drop queue watcher.

Watches ``platform/worker/queue/*.json`` -- each file describes a single job
written by the backend. For each file the watcher:

1. Parses the JSON.
2. Sends the matching Dramatiq actor.
3. Moves the file into ``queue/.processed/``.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from msg_embedding.core.logging import get_logger

from .settings import WorkerSettings, get_settings

_log = get_logger(__name__)


def _load_actors_map() -> dict[str, Any]:
    """Lazy import so this module can be imported in environments without dramatiq."""
    from .actors import ACTORS

    return ACTORS


def _iter_queue_files(queue_dir: Path) -> list[Path]:
    """Return sorted ``*.json`` files in ``queue_dir`` (excludes ``.processed``)."""
    if not queue_dir.exists():
        return []
    return sorted(p for p in queue_dir.glob("*.json") if p.is_file())


def _parse_job_file(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _log.error("queue_file_read_failed", path=str(path), error=str(exc))
        return None
    if not isinstance(data, dict):
        _log.error("queue_file_not_object", path=str(path))
        return None
    if "job_id" not in data or "type" not in data:
        _log.error("queue_file_missing_fields", path=str(path))
        return None
    return data


def _move_to_processed(path: Path, processed_dir: Path) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    dest = processed_dir / path.name
    if dest.exists():
        stem = path.stem
        suffix = path.suffix
        i = 1
        while True:
            candidate = processed_dir / f"{stem}.{i}{suffix}"
            if not candidate.exists():
                dest = candidate
                break
            i += 1
    shutil.move(str(path), str(dest))
    return dest


def process_once(
    *,
    settings: WorkerSettings | None = None,
    actors_map: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Scan once; enqueue every pending job file; return list of results."""
    settings = settings or get_settings()
    settings.ensure_dirs()
    actors_map = actors_map if actors_map is not None else _load_actors_map()

    results: list[dict[str, Any]] = []
    for qfile in _iter_queue_files(settings.queue_path):
        data = _parse_job_file(qfile)
        if data is None:
            dest = _move_to_processed(qfile, settings.processed_path)
            results.append(
                {
                    "path": str(qfile),
                    "job_id": None,
                    "type": None,
                    "enqueued": False,
                    "processed_path": str(dest),
                    "error": "malformed",
                }
            )
            continue

        task_type = data["type"]
        job_id = data["job_id"]
        overrides = data.get("config_overrides") or {}
        actor = actors_map.get(task_type)
        if actor is None:
            _log.error("no_actor_for_type", task_type=task_type, job_id=job_id)
            dest = _move_to_processed(qfile, settings.processed_path)
            results.append(
                {
                    "path": str(qfile),
                    "job_id": job_id,
                    "type": task_type,
                    "enqueued": False,
                    "processed_path": str(dest),
                    "error": "unknown_type",
                }
            )
            continue

        try:
            actor.send(job_id=job_id, overrides=overrides)
            enqueued = True
            err: str | None = None
        except Exception as exc:
            _log.exception("actor_send_failed", job_id=job_id, error=str(exc))
            enqueued = False
            err = str(exc)

        dest = _move_to_processed(qfile, settings.processed_path)
        results.append(
            {
                "path": str(qfile),
                "job_id": job_id,
                "type": task_type,
                "enqueued": enqueued,
                "processed_path": str(dest),
                "error": err,
            }
        )
        _log.info("job_enqueued", job_id=job_id, task_type=task_type, file=qfile.name)
    return results


def watch_forever(
    *,
    settings: WorkerSettings | None = None,
    actors_map: dict[str, Any] | None = None,
    stop_after: int | None = None,
) -> None:
    """Run the polling loop until interrupted."""
    settings = settings or get_settings()
    _log.info("queue_watcher_started", dir=str(settings.queue_path))
    ticks = 0
    try:
        while True:
            process_once(settings=settings, actors_map=actors_map)
            ticks += 1
            if stop_after is not None and ticks >= stop_after:
                return
            time.sleep(settings.queue_poll_interval_secs)
    except KeyboardInterrupt:
        _log.info("queue_watcher_stopping")


__all__ = ["process_once", "watch_forever"]
