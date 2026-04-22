"""Unit tests for ExperimentTracker."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from msg_embedding.training.experiment import ExperimentTracker


def test_tracker_wandb_disabled_no_exception(tmp_path: Path) -> None:
    tr = ExperimentTracker(run_id="unit-A", log_dir=tmp_path, use_wandb=False)
    tr.log_scalar("train/loss", 0.5, step=0)
    tr.log_scalar("train/loss", 0.4, step=1)
    tr.log_histogram("weights", torch.randn(10), step=1)
    tr.log_image("sample", np.zeros((4, 4), dtype=np.uint8), step=1)
    tr.log_config({"lr": 1e-4, "batch_size": 8})
    tr.close()


def test_tracker_close_is_idempotent(tmp_path: Path) -> None:
    tr = ExperimentTracker(run_id="unit-B", log_dir=tmp_path, use_wandb=False)
    tr.close()
    tr.close()


def test_tracker_context_manager(tmp_path: Path) -> None:
    with ExperimentTracker(run_id="unit-C", log_dir=tmp_path,
                           use_wandb=False) as tr:
        tr.log_scalar("loss", 0.1, step=0)


def test_tracker_non_main_rank_is_silent(tmp_path: Path) -> None:
    tr = ExperimentTracker(run_id="unit-D", log_dir=tmp_path, use_wandb=False,
                           rank=1)
    assert not (tmp_path / "unit-D").exists()
    tr.log_scalar("anything", 1.0, step=0)
    tr.log_config({"k": "v"})
    tr.close()


def test_tracker_wandb_init_failure_graceful(tmp_path: Path, monkeypatch) -> None:
    class _FakeWandb:
        Histogram = object
        Image = object

        @staticmethod
        def init(**kwargs):
            raise RuntimeError("simulated wandb failure")

    import sys
    monkeypatch.setitem(sys.modules, "wandb", _FakeWandb())

    tr = ExperimentTracker(run_id="unit-E", log_dir=tmp_path, use_wandb=True)
    tr.log_scalar("loss", 0.5, step=0)
    tr.close()


def test_tracker_tensorboard_disabled_when_missing(tmp_path: Path,
                                                   monkeypatch) -> None:
    import sys

    class _RaisingModule:
        def __getattr__(self, name: str):  # pragma: no cover
            raise ImportError("blocked")

    monkeypatch.setitem(sys.modules, "tensorboard", _RaisingModule())
    tr = ExperimentTracker(run_id="unit-F", log_dir=tmp_path, use_wandb=False)
    tr.log_scalar("x", 1.0, step=0)
    tr.close()


def test_tracker_log_scalar_casts_types(tmp_path: Path) -> None:
    tr = ExperimentTracker(run_id="unit-G", log_dir=tmp_path, use_wandb=False)
    tr.log_scalar("loss", torch.tensor(0.25).item(), step=0)
    tr.close()
