"""Command-line entrypoints: ``python -m platform.worker.cli ...``.

Subcommands: worker | watcher | status.
"""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

from msg_embedding.core.logging import get_logger

from .settings import get_settings

if TYPE_CHECKING:
    from collections.abc import Sequence

_log = get_logger(__name__)


def _cmd_worker(args: argparse.Namespace) -> int:
    from .broker import make_redis_broker

    make_redis_broker()

    import dramatiq

    from . import actors  # noqa: F401

    broker = dramatiq.get_broker()
    worker = dramatiq.Worker(
        broker,
        worker_threads=args.threads,
        worker_timeout=args.worker_timeout_ms,
    )
    _log.info("worker_starting", threads=args.threads, processes=args.processes)
    worker.start()
    try:
        import signal

        signal.pause() if hasattr(signal, "pause") else _windows_wait()
    except KeyboardInterrupt:
        pass
    finally:
        _log.info("worker_stopping")
        worker.stop()
    return 0


def _windows_wait() -> None:
    """Windows lacks signal.pause; fall back to a plain sleep loop."""
    import time

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return


def _cmd_watcher(args: argparse.Namespace) -> int:
    from .broker import make_redis_broker

    make_redis_broker()
    from . import actors  # noqa: F401
    from .queue_watcher import watch_forever

    watch_forever()
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    from .tasks.base import read_log_tail, read_progress

    job_id = args.job_id
    prog = read_progress(job_id)
    tail = read_log_tail(job_id, lines=args.tail_lines)
    out = {
        "job_id": job_id,
        "progress": prog,
        "log_tail": tail,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))  # noqa: T201
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="platform.worker.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pw = sub.add_parser("worker", help="run a Dramatiq worker")
    pw.add_argument("--processes", type=int, default=1)
    pw.add_argument("--threads", type=int, default=4)
    pw.add_argument("--worker-timeout-ms", type=int, default=1000)
    pw.set_defaults(func=_cmd_worker)

    ph = sub.add_parser("watcher", help="run the file-drop queue watcher")
    ph.set_defaults(func=_cmd_watcher)

    ps = sub.add_parser("status", help="print progress + log tail for a job")
    ps.add_argument("job_id")
    ps.add_argument("--tail-lines", type=int, default=40)
    ps.set_defaults(func=_cmd_status)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()
    settings.ensure_dirs()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
