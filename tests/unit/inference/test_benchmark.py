"""Tests for benchmark_latency."""

from __future__ import annotations

from pathlib import Path

import torch

from msg_embedding.inference.benchmark import benchmark_latency
from msg_embedding.inference.wrapper import build_encoder_wrapper


def test_benchmark_latency_on_eager_module(channel_mae_ckpt: Path) -> None:
    wrapper = build_encoder_wrapper(ckpt_path=channel_mae_ckpt, device="cpu")
    result = benchmark_latency(
        wrapper, batch_sizes=[1, 4], warmup=1, iters=3, device="cpu"
    )
    assert set(result.keys()) == {1, 4}
    for _bs, stats in result.items():
        assert {"ms_p50", "ms_p99", "throughput_per_s"} <= set(stats.keys())
        assert stats["ms_p50"] >= 0.0
        assert stats["ms_p99"] >= stats["ms_p50"]
        assert stats["throughput_per_s"] > 0.0


def test_benchmark_latency_on_torchscript(
    channel_mae_ckpt: Path, tmp_path: Path
) -> None:
    from msg_embedding.inference.export import export_torchscript

    ts_path = tmp_path / "m.ts"
    export_torchscript(ckpt_path=channel_mae_ckpt, output_path=ts_path)
    result = benchmark_latency(
        ts_path, batch_sizes=[2], warmup=1, iters=2, device="cpu"
    )
    assert 2 in result
    assert result[2]["ms_p50"] >= 0.0


def test_benchmark_latency_device_fallback(channel_mae_ckpt: Path) -> None:
    wrapper = build_encoder_wrapper(ckpt_path=channel_mae_ckpt, device="cpu")
    result = benchmark_latency(
        wrapper, batch_sizes=(2,), warmup=1, iters=2, device=torch.device("cpu")
    )
    assert 2 in result
