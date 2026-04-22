"""Tests for training curves visualization."""

from __future__ import annotations

import pytest

from msg_embedding.viz.training_curves import (
    plot_training_curves,
    read_tb_scalars,
)


def _write_small_tb_run(tmp_path):
    pytest.importorskip("tensorboard")
    from torch.utils.tensorboard import SummaryWriter

    logdir = tmp_path / "tb"
    writer = SummaryWriter(log_dir=str(logdir))
    for step in range(10):
        writer.add_scalar("loss/train", 1.0 / (step + 1), step)
        writer.add_scalar("lr", 1e-4, step)
        writer.add_scalar("grad_norm", 0.5 + 0.01 * step, step)
        writer.add_scalar("mask_ratio", 0.3, step)
    writer.flush()
    writer.close()
    return logdir


def test_read_tb_scalars_round_trip(tmp_path) -> None:
    logdir = _write_small_tb_run(tmp_path)
    scalars = read_tb_scalars(logdir)
    assert "loss/train" in scalars
    steps, values = scalars["loss/train"]
    assert len(steps) == 10
    assert len(values) == 10
    assert values[0] > values[-1]


def test_read_tb_scalars_missing_dir(tmp_path) -> None:
    out = read_tb_scalars(tmp_path / "does_not_exist")
    assert out == {}


def test_plot_training_curves_no_data(tmp_path) -> None:
    out_html = tmp_path / "curves.html"
    written = plot_training_curves(tmp_path / "empty_tb", out_html)
    assert written.exists()
    assert written.read_text(encoding="utf-8").startswith("<html>")


def test_plot_training_curves_with_data(tmp_path) -> None:
    logdir = _write_small_tb_run(tmp_path)
    out_html = tmp_path / "curves.html"
    written = plot_training_curves(logdir, out_html)
    assert written.exists()
    text = written.read_text(encoding="utf-8")
    assert len(text) > 100
