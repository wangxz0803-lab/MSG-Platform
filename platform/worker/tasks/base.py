"""Base :class:`TaskRunner` used by every job-type actor.

Responsibilities: build command, spawn subprocess, stream stdout, parse
progress lines, poll for cancel flag, update DB status.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import threading
from collections import deque
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from msg_embedding.core.logging import get_logger

from ..db_update import get_job_status, update_job_progress, update_job_run_id, update_job_status
from ..settings import WorkerSettings, get_settings

_log = get_logger(__name__)

_PROGRESS_RE = re.compile(r"^\[progress\]\s+pct=([0-9.]+)(?:\s+step=(.+))?\s*$")
_RUN_ID_RE = re.compile(r"^\[run_id\]\s+(.+)\s*$")

_SCRIPT_NAMES: dict[str, str] = {
    "convert": "run_convert.py",
    "bridge": "run_bridge.py",
    "train": "run_train.py",
    "eval": "run_eval.py",
    "infer": "run_infer.py",
    "export": "run_export.py",
    "report": "run_report.py",
    "simulate": "run_simulate.py",
}


def parse_progress_line(line: str) -> tuple[float, str | None] | None:
    """Extract ``(pct, step)`` from ``[progress] pct=X step=Y``."""
    m = _PROGRESS_RE.match(line.strip())
    if m is None:
        return None
    pct = float(m.group(1))
    step = m.group(2).strip() if m.group(2) else None
    return pct, step


def _flatten_overrides(overrides: dict[str, Any], prefix: str = "") -> list[str]:
    """Convert a nested dict into Hydra ``key=value`` tokens."""
    out: list[str] = []
    for k, v in overrides.items():
        full = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.extend(_flatten_overrides(v, full))
        elif isinstance(v, list | tuple):
            out.append(f"{full}={json.dumps(v)}")
        elif isinstance(v, bool):
            out.append(f"{full}={'true' if v else 'false'}")
        elif v is None:
            out.append(f"{full}=null")
        else:
            out.append(f"{full}={v}")
    return out


def _get_ddp_world_size(overrides: dict[str, Any]) -> int:
    """Extract ``ddp.world_size`` -- defaults to 1."""
    ddp = overrides.get("ddp")
    if isinstance(ddp, dict):
        ws = ddp.get("world_size", 1)
    else:
        ws = overrides.get("ddp.world_size") or overrides.get("ddp_world_size") or 1
    try:
        return max(1, int(ws))
    except (TypeError, ValueError):
        return 1


def build_command(
    task_type: str,
    overrides: dict[str, Any],
    *,
    python_exe: str | None = None,
    repo_root: Path | None = None,
) -> list[str]:
    """Build the argv list for the subprocess."""
    if task_type not in _SCRIPT_NAMES:
        raise ValueError(f"Unknown task_type: {task_type!r}")

    settings = get_settings()
    python_exe = python_exe or settings.python_exe
    repo_root = repo_root or settings.repo_path
    script_path = repo_root / "scripts" / _SCRIPT_NAMES[task_type]

    world_size = _get_ddp_world_size(overrides)
    hydra_tokens = _flatten_overrides(overrides)

    if task_type == "train" and world_size > 1:
        cmd = [
            python_exe,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={world_size}",
            str(script_path),
            *hydra_tokens,
        ]
    else:
        cmd = [python_exe, str(script_path), *hydra_tokens]
    return cmd


class TaskRunner:
    """Runs a single job: spawn, stream, parse, cancel, update DB."""

    ERROR_TAIL_CHARS = 500
    ERROR_TAIL_LINES = 200

    def __init__(
        self,
        job_id: str,
        task_type: str,
        overrides: dict[str, Any] | None = None,
        *,
        settings: WorkerSettings | None = None,
    ) -> None:
        self.job_id = job_id
        self.task_type = task_type
        self.overrides: dict[str, Any] = dict(overrides or {})
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()

        self.log_path = self.settings.log_path / f"{job_id}.log"
        self.progress_path = self.settings.progress_path / f"{job_id}.json"
        self.cancel_flag = self.settings.cancel_path / f"{job_id}.flag"

        self._cancelled = False
        self._tail: deque[str] = deque(maxlen=self.ERROR_TAIL_LINES)
        self._run_id: str | None = None

    def _write_progress(self, pct: float, step: str | None) -> None:
        payload = {
            "job_id": self.job_id,
            "pct": pct,
            "step": step,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp = self.progress_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self.progress_path)
        try:
            update_job_progress(self.job_id, pct, step, db_url=self.settings.db_url)
        except Exception as exc:
            _log.warning("progress_db_update_failed", job_id=self.job_id, error=str(exc))

    def _cancel_requested(self) -> bool:
        return self.cancel_flag.exists()

    def _terminate(self, proc: subprocess.Popen[str]) -> None:
        _log.info("terminating_job", job_id=self.job_id, pid=proc.pid)
        try:
            proc.terminate()
        except Exception as exc:
            _log.warning("terminate_failed", job_id=self.job_id, error=str(exc))
        try:
            proc.wait(timeout=self.settings.cancel_grace_secs)
        except subprocess.TimeoutExpired:
            _log.warning("kill_after_timeout", job_id=self.job_id)
            try:
                proc.kill()
            except Exception as exc:
                _log.warning("kill_failed", job_id=self.job_id, error=str(exc))

    def _cancel_watcher(self, proc: subprocess.Popen[str], stop: threading.Event) -> None:
        while not stop.is_set():
            if self._cancel_requested():
                self._cancelled = True
                self._terminate(proc)
                return
            stop.wait(self.settings.worker_poll_interval_secs)

    def run(self) -> str:
        """Execute the job and return the final status string."""
        try:
            current = get_job_status(self.job_id, db_url=self.settings.db_url)
        except Exception:
            current = None
        if current in {"completed", "cancelled"}:
            _log.info("job_already_terminal", job_id=self.job_id, status=current)
            return current

        started = datetime.now(timezone.utc)
        update_job_status(
            self.job_id,
            "running",
            started_at=started,
            log_path=str(self.log_path),
            progress_pct=0.0,
            db_url=self.settings.db_url,
        )

        self.overrides.setdefault("project", {})
        if isinstance(self.overrides.get("project"), dict):
            self.overrides["project"]["job_id"] = self.job_id

        try:
            cmd = build_command(
                self.task_type,
                self.overrides,
                python_exe=self.settings.python_exe,
                repo_root=self.settings.repo_path,
            )
        except Exception as exc:
            _log.exception("command_build_failed", job_id=self.job_id)
            update_job_status(
                self.job_id,
                "failed",
                finished_at=datetime.now(timezone.utc),
                error_msg=str(exc)[: self.ERROR_TAIL_CHARS],
                db_url=self.settings.db_url,
            )
            return "failed"

        _log.info("job_starting", job_id=self.job_id, cmd=shlex.join(cmd))
        return_code = self._spawn_and_stream(cmd)

        finished = datetime.now(timezone.utc)
        if self._cancelled:
            update_job_status(
                self.job_id,
                "cancelled",
                finished_at=finished,
                db_url=self.settings.db_url,
            )
            self._cleanup_cancel_flag()
            return "cancelled"
        if return_code == 0:
            update_job_status(
                self.job_id,
                "completed",
                finished_at=finished,
                progress_pct=100.0,
                db_url=self.settings.db_url,
            )
            self._write_progress(100.0, "done")
            return "completed"

        tail = "".join(self._tail)[-self.ERROR_TAIL_CHARS :]
        update_job_status(
            self.job_id,
            "failed",
            finished_at=finished,
            error_msg=tail or f"subprocess exited with code {return_code}",
            db_url=self.settings.db_url,
        )
        return "failed"

    def _spawn_and_stream(self, cmd: list[str]) -> int:
        """Run ``cmd``, stream output, watch for cancellation."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        proc = subprocess.Popen(
            cmd,
            cwd=str(self.settings.repo_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            shell=False,
        )

        stop = threading.Event()
        watcher = threading.Thread(
            target=self._cancel_watcher, args=(proc, stop), daemon=True
        )
        watcher.start()

        try:
            with self.log_path.open("w", encoding="utf-8") as logf:
                if proc.stdout is not None:
                    for raw in proc.stdout:
                        self._tail.append(raw)
                        logf.write(raw)
                        logf.flush()
                        parsed = parse_progress_line(raw)
                        if parsed is not None:
                            pct, step = parsed
                            self._write_progress(pct, step)
                        rid_match = _RUN_ID_RE.match(raw.strip())
                        if rid_match is not None:
                            self._run_id = rid_match.group(1).strip()
                            try:
                                update_job_run_id(
                                    self.job_id,
                                    self._run_id,
                                    db_url=self.settings.db_url,
                                )
                            except Exception as exc:
                                _log.warning(
                                    "run_id_db_update_failed",
                                    job_id=self.job_id,
                                    error=str(exc),
                                )
                return_code = proc.wait()
        finally:
            stop.set()
            watcher.join(timeout=1.0)
        return return_code

    def _cleanup_cancel_flag(self) -> None:
        with suppress(OSError):
            self.cancel_flag.unlink(missing_ok=True)


def request_cancel(job_id: str, *, settings: WorkerSettings | None = None) -> Path:
    """Create the cancel flag file for ``job_id`` (idempotent)."""
    settings = settings or get_settings()
    settings.ensure_dirs()
    flag = settings.cancel_path / f"{job_id}.flag"
    flag.touch(exist_ok=True)
    return flag


def read_progress(
    job_id: str, *, settings: WorkerSettings | None = None
) -> dict[str, Any] | None:
    """Load the on-disk progress JSON for ``job_id`` (or None if absent)."""
    settings = settings or get_settings()
    p = settings.progress_path / f"{job_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def read_log_tail(
    job_id: str, *, lines: int = 40, settings: WorkerSettings | None = None
) -> str:
    """Return the last ``lines`` of the job log (or empty string)."""
    settings = settings or get_settings()
    p = settings.log_path / f"{job_id}.log"
    if not p.exists():
        return ""
    buf: deque[str] = deque(maxlen=lines)
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            buf.append(line)
    return "".join(buf)


__all__ = [
    "TaskRunner",
    "build_command",
    "parse_progress_line",
    "read_log_tail",
    "read_progress",
    "request_cancel",
]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(  # noqa: T201
            "usage: python -m platform.worker.tasks.base <job_id> <task_type> [json_overrides]"
        )
        raise SystemExit(2)
    _job = sys.argv[1]
    _type = sys.argv[2]
    _over = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    rc = TaskRunner(_job, _type, _over).run()
    print(rc)  # noqa: T201
