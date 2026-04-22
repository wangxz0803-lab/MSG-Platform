"""Tests for TaskRunner: progress parsing, cancel handling, command building."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("dramatiq")
pytest.importorskip("sqlalchemy")


def test_parse_progress_line_basic():
    from platform.worker.tasks.base import parse_progress_line

    assert parse_progress_line("[progress] pct=25.0 step=epoch_1/40") == (25.0, "epoch_1/40")
    assert parse_progress_line("[progress] pct=100.0 step=done") == (100.0, "done")
    assert parse_progress_line("[progress] pct=5") == (5.0, None)
    assert parse_progress_line("random line") is None
    assert parse_progress_line("") is None


def test_build_command_basic(worker_settings):
    from platform.worker.tasks.base import build_command

    cmd = build_command(
        "eval",
        {"eval": {"batch_size": 32}, "device": "cpu"},
        python_exe="/usr/bin/python",
        repo_root=Path("/repo"),
    )
    assert cmd[0] == "/usr/bin/python"
    assert cmd[1].endswith("run_eval.py")
    tail = cmd[2:]
    assert "eval.batch_size=32" in tail
    assert "device=cpu" in tail


def test_build_command_train_ddp_uses_torchrun(worker_settings):
    from platform.worker.tasks.base import build_command

    cmd = build_command(
        "train",
        {"ddp": {"world_size": 4}, "train": {"epochs": 2}},
        python_exe="python",
        repo_root=Path("/repo"),
    )
    assert "torch.distributed.run" in cmd
    assert any(tok.startswith("--nproc_per_node=4") for tok in cmd)
    assert any(tok == "train.epochs=2" for tok in cmd)


def test_build_command_unknown_type(worker_settings):
    from platform.worker.tasks.base import build_command

    with pytest.raises(ValueError):
        build_command("bogus", {})


def test_runner_success_writes_log_and_progress(worker_settings, monkeypatch):
    from platform.worker.tasks import base as base_mod

    inline = textwrap.dedent("""
        import sys
        print("starting")
        print("[progress] pct=10.0 step=setup", flush=True)
        print("[progress] pct=50.0 step=working", flush=True)
        print("done")
        sys.exit(0)
    """).strip()

    monkeypatch.setattr(
        base_mod, "build_command", lambda *a, **kw: [sys.executable, "-c", inline]
    )

    runner = base_mod.TaskRunner("job-ok", "eval", {"eval": {"n": 1}}, settings=worker_settings)
    status = runner.run()
    assert status == "completed"

    log_path = Path(worker_settings.log_dir) / "job-ok.log"
    assert log_path.exists()
    log_text = log_path.read_text()
    assert "starting" in log_text
    assert "[progress] pct=50" in log_text

    prog_path = Path(worker_settings.progress_dir) / "job-ok.json"
    assert prog_path.exists()
    data = json.loads(prog_path.read_text())
    assert data["pct"] == 100.0
    assert data["step"] == "done"


def test_runner_nonzero_exit_marks_failed(worker_settings, monkeypatch):
    from platform.worker.tasks import base as base_mod

    inline = textwrap.dedent("""
        import sys
        print("boom")
        sys.exit(3)
    """).strip()

    monkeypatch.setattr(
        base_mod, "build_command", lambda *a, **kw: [sys.executable, "-c", inline]
    )

    runner = base_mod.TaskRunner("job-fail", "eval", {}, settings=worker_settings)
    status = runner.run()
    assert status == "failed"

    from platform.worker.db_update import get_job_status

    assert get_job_status("job-fail", db_url=worker_settings.db_url) == "failed"


def test_runner_respects_cancel_flag(worker_settings, monkeypatch):
    from platform.worker.tasks import base as base_mod

    inline = textwrap.dedent("""
        import sys, time
        for i in range(60):
            print(f"tick {i}", flush=True)
            time.sleep(0.2)
        sys.exit(0)
    """).strip()

    monkeypatch.setattr(
        base_mod, "build_command", lambda *a, **kw: [sys.executable, "-c", inline]
    )

    from platform.worker.tasks.base import request_cancel

    request_cancel("job-cancel", settings=worker_settings)

    runner = base_mod.TaskRunner("job-cancel", "eval", {}, settings=worker_settings)
    status = runner.run()
    assert status == "cancelled"


def test_runner_idempotent_on_completed(worker_settings, monkeypatch):
    from platform.worker.db_update import update_job_status
    from platform.worker.tasks import base as base_mod

    update_job_status("already-done", "completed", db_url=worker_settings.db_url)

    calls: list = []

    def fail_build(*a, **kw):
        calls.append(1)
        raise AssertionError("should not be called")

    monkeypatch.setattr(base_mod, "build_command", fail_build)
    status = base_mod.TaskRunner("already-done", "eval", {}, settings=worker_settings).run()
    assert status == "completed"
    assert calls == []


def test_read_progress_and_log_tail(worker_settings):
    from platform.worker.tasks.base import read_log_tail, read_progress

    prog = Path(worker_settings.progress_dir) / "jx.json"
    prog.write_text(json.dumps({"job_id": "jx", "pct": 42.0, "step": "s"}))
    log = Path(worker_settings.log_dir) / "jx.log"
    log.write_text("a\nb\nc\nd\n")

    data = read_progress("jx", settings=worker_settings)
    assert data["pct"] == 42.0
    tail = read_log_tail("jx", lines=2, settings=worker_settings)
    assert tail == "c\nd\n"

    assert read_progress("missing", settings=worker_settings) is None
    assert read_log_tail("missing", settings=worker_settings) == ""
