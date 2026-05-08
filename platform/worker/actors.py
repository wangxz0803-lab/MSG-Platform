"""Dramatiq actors -- one per job type.

Importing this module requires ``dramatiq`` to be installed and a broker to
be registered (``make_redis_broker`` or ``install_stub_broker``). If no
broker is set, a StubBroker is installed automatically so that import
doesn't fail in test collection.
"""

from __future__ import annotations

from typing import Any

import dramatiq

from .broker import install_stub_broker
from .tasks.base import TaskRunner

try:
    dramatiq.get_broker()
except Exception:
    install_stub_broker()

_TIME_LIMIT_MS = 12 * 60 * 60 * 1000
_QUEUE = "msg_jobs"


@dramatiq.actor(max_retries=2, time_limit=_TIME_LIMIT_MS, queue_name=_QUEUE)
def run_convert_job(job_id: str, overrides: dict[str, Any] | None = None) -> str:
    return TaskRunner(job_id, "convert", overrides).run()


@dramatiq.actor(max_retries=2, time_limit=_TIME_LIMIT_MS, queue_name=_QUEUE)
def run_bridge_job(job_id: str, overrides: dict[str, Any] | None = None) -> str:
    return TaskRunner(job_id, "bridge", overrides).run()


@dramatiq.actor(max_retries=2, time_limit=_TIME_LIMIT_MS, queue_name=_QUEUE)
def run_eval_job(job_id: str, overrides: dict[str, Any] | None = None) -> str:
    return TaskRunner(job_id, "eval", overrides).run()


@dramatiq.actor(max_retries=2, time_limit=_TIME_LIMIT_MS, queue_name=_QUEUE)
def run_infer_job(job_id: str, overrides: dict[str, Any] | None = None) -> str:
    return TaskRunner(job_id, "infer", overrides).run()


@dramatiq.actor(max_retries=2, time_limit=_TIME_LIMIT_MS, queue_name=_QUEUE)
def run_export_job(job_id: str, overrides: dict[str, Any] | None = None) -> str:
    return TaskRunner(job_id, "export", overrides).run()


@dramatiq.actor(max_retries=2, time_limit=_TIME_LIMIT_MS, queue_name=_QUEUE)
def run_report_job(job_id: str, overrides: dict[str, Any] | None = None) -> str:
    return TaskRunner(job_id, "report", overrides).run()


@dramatiq.actor(max_retries=2, time_limit=_TIME_LIMIT_MS, queue_name=_QUEUE)
def run_simulate_job(job_id: str, overrides: dict[str, Any] | None = None) -> str:
    return TaskRunner(job_id, "simulate", overrides).run()


ACTORS: dict[str, dramatiq.Actor] = {
    "convert": run_convert_job,
    "bridge": run_bridge_job,
    "eval": run_eval_job,
    "infer": run_infer_job,
    "export": run_export_job,
    "report": run_report_job,
    "simulate": run_simulate_job,
}


def get_actor(task_type: str) -> dramatiq.Actor:
    """Return the actor registered for ``task_type``. Raises KeyError otherwise."""
    try:
        return ACTORS[task_type]
    except KeyError as exc:
        raise KeyError(f"no actor registered for task type {task_type!r}") from exc


__all__ = [
    "ACTORS",
    "get_actor",
    "run_bridge_job",
    "run_convert_job",
    "run_eval_job",
    "run_export_job",
    "run_infer_job",
    "run_report_job",
    "run_simulate_job",
]
