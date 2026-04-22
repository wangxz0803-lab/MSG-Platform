"""MSG-Embedding worker package.

A Dramatiq + Redis task queue that executes MSG training/eval/inference
pipelines as subprocesses of the Hydra entry points under ``scripts/``.
"""

from __future__ import annotations

__all__ = ["settings", "broker", "actors", "tasks", "queue_watcher", "db_update"]
