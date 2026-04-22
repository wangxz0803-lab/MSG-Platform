"""Report task -- wraps ``scripts/run_report.py``."""

from __future__ import annotations

from typing import Any

from .base import TaskRunner

TASK_TYPE = "report"


def run(job_id: str, overrides: dict[str, Any] | None = None) -> str:
    return TaskRunner(job_id, TASK_TYPE, overrides).run()
