"""ChannelHub worker package.

A thread-based task queue that executes channel data collection/eval/inference
pipelines as subprocesses of the Hydra entry points under ``scripts/``.
"""

from __future__ import annotations

__all__ = ["settings", "broker", "actors", "tasks", "queue_watcher", "db_update"]
