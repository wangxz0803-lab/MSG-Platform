"""Unit tests for training callbacks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from msg_embedding.training.callbacks import (
    CheckpointManager,
    EarlyStopping,
    GradNormMonitor,
    LRMonitor,
)


class _Tracker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float, int]] = []

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self.calls.append((name, float(value), int(step)))


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    mgr = CheckpointManager(run_dir=tmp_path / "runA", keep_last_n=2)
    state = {"model": {"w": torch.ones(2, 2)}, "epoch": 1, "step": 5,
             "best_val_loss": 0.5}
    path = mgr.save(state, tag="last")
    assert path.exists()

    got = mgr.load_latest()
    assert got is not None
    assert got["epoch"] == 1
    assert got["step"] == 5
    assert torch.equal(got["model"]["w"], torch.ones(2, 2))


def test_checkpoint_best_promotes(tmp_path: Path) -> None:
    mgr = CheckpointManager(run_dir=tmp_path / "runB")
    state = {"model": {}, "best_val_loss": 0.3, "epoch": 3}
    mgr.save(state, tag="last", is_best=True)
    best = mgr.load_best()
    assert best is not None
    assert best["epoch"] == 3


def test_checkpoint_metadata_written(tmp_path: Path) -> None:
    mgr = CheckpointManager(run_dir=tmp_path / "runC",
                            metadata={"git_sha": "abc123"})
    mgr.save({"model": {}, "best_val_loss": 0.4}, tag="last", is_best=True)
    meta_path = tmp_path / "runC" / "metadata.json"
    assert meta_path.exists()
    payload = json.loads(meta_path.read_text())
    assert payload["git_sha"] == "abc123"
    assert payload["best_val_loss"] == pytest.approx(0.4)
    assert payload["last_tag"] == "last"


def test_checkpoint_snapshot_pruning(tmp_path: Path) -> None:
    mgr = CheckpointManager(run_dir=tmp_path / "runD", keep_last_n=2)
    for i in range(4):
        mgr.save({"model": {}, "epoch": i}, tag=f"epoch_{i}")
    remaining = sorted((tmp_path / "runD").glob("ckpt_epoch_*.pth"))
    assert len(remaining) == 2


def test_checkpoint_load_missing_tag_returns_none(tmp_path: Path) -> None:
    mgr = CheckpointManager(run_dir=tmp_path / "runE")
    assert mgr.load_latest() is None
    assert mgr.load_best() is None
    assert mgr.load_tag("epoch_42") is None


def test_checkpoint_config_snapshot(tmp_path: Path) -> None:
    mgr = CheckpointManager(
        run_dir=tmp_path / "runF",
        config={"train": {"lr": 1e-4, "epochs": 2}},
    )
    mgr.save({"model": {}}, tag="last")
    cfg_path = tmp_path / "runF" / "config.yaml"
    assert cfg_path.exists()
    content = cfg_path.read_text()
    assert "lr" in content and "epochs" in content


def test_early_stopping_patience() -> None:
    es = EarlyStopping(patience=2, min_delta=0.0)
    assert es(0.5) is False
    assert es(0.4) is False
    assert es(0.45) is False
    assert es(0.46) is False
    assert es(0.47) is True


def test_early_stopping_min_delta() -> None:
    es = EarlyStopping(patience=1, min_delta=0.1)
    assert es(1.0) is False
    assert es(0.95) is False
    assert es(0.94) is True


def test_early_stopping_max_mode() -> None:
    es = EarlyStopping(patience=1, mode="max")
    assert es(0.1) is False
    assert es(0.2) is False
    assert es(0.15) is False
    assert es(0.14) is True


def test_early_stopping_reset() -> None:
    es = EarlyStopping(patience=1)
    for v in (0.5, 0.6, 0.7):
        es(v)
    es.reset()
    assert es.bad_epochs == 0
    assert es.best == float("inf")


def test_lr_monitor_single_group_logs_canonical_name() -> None:
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    tracker = _Tracker()
    monitor = LRMonitor(tracker=tracker)
    out = monitor.step(opt, step=0)
    assert "lr" in out and "lr/group_0" in out
    assert any(c[0] == "lr" for c in tracker.calls)
    assert any(c[0] == "lr/group_0" for c in tracker.calls)


def test_lr_monitor_multi_group_logs_per_group() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    opt = torch.optim.SGD([
        {"params": model[0].parameters(), "lr": 0.01},
        {"params": model[1].parameters(), "lr": 0.02},
    ])
    tracker = _Tracker()
    LRMonitor(tracker=tracker).step(opt, step=7)
    names = [c[0] for c in tracker.calls]
    assert "lr/group_0" in names and "lr/group_1" in names


def test_grad_norm_monitor_computes_l2() -> None:
    model = nn.Linear(3, 3, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    tracker = _Tracker()
    gn = GradNormMonitor(tracker=tracker).step(list(model.parameters()), step=0)
    assert gn == pytest.approx(model.weight.numel() ** 0.5)
    assert tracker.calls[0][0] == "grad_norm"


def test_grad_norm_monitor_no_grads_returns_zero() -> None:
    model = nn.Linear(3, 3)
    gn = GradNormMonitor().step(list(model.parameters()), step=0)
    assert gn == 0.0
