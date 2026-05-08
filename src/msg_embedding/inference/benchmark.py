"""Latency / throughput benchmark for ChannelHub inference artifacts."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from msg_embedding.core.logging import get_logger

_log = get_logger(__name__)


def _load_artifact(model_or_path: Any) -> Any:
    if isinstance(model_or_path, str | Path):
        path = Path(model_or_path)
        suffix = path.suffix.lower()
        if suffix in (".onnx",):
            import onnxruntime as ort  # type: ignore
            return ort.InferenceSession(str(path),
                                        providers=["CPUExecutionProvider"])
        if suffix in (".ts", ".pt", ".torchscript"):
            return torch.jit.load(str(path), map_location="cpu")
        raise ValueError(f"unrecognised artifact suffix: {suffix}")
    return model_or_path


def _is_onnx_session(obj: Any) -> bool:
    cls_name = obj.__class__.__name__
    mod_name = obj.__class__.__module__ or ""
    return cls_name == "InferenceSession" and mod_name.startswith("onnxruntime")


def benchmark_latency(
    model_or_onnx: Any,
    batch_sizes: list[int] | tuple[int, ...] = (1, 16, 128),
    warmup: int = 10,
    iters: int = 100,
    seq_len: int = 16,
    token_dim: int = 128,
    device: str | torch.device = "cpu",
) -> dict[int, dict[str, float]]:
    """Measure per-batch latency + throughput for an inference artifact."""
    artifact = _load_artifact(model_or_onnx)
    is_onnx = _is_onnx_session(artifact)
    if not is_onnx:
        torch_device = torch.device(device)
        artifact.eval() if hasattr(artifact, "eval") else None
        if hasattr(artifact, "to"):
            artifact.to(torch_device)
    else:
        torch_device = torch.device("cpu")

    results: dict[int, dict[str, float]] = {}
    for bs in batch_sizes:
        tokens = torch.randn(bs, seq_len, token_dim, dtype=torch.float32,
                             device=torch_device)
        token_mask = torch.zeros(bs, seq_len, dtype=torch.bool,
                                 device=torch_device)

        runner = _make_runner(artifact, is_onnx)
        for _ in range(warmup):
            runner(tokens, token_mask)

        timings_ms: list[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            runner(tokens, token_mask)
            t1 = time.perf_counter()
            timings_ms.append((t1 - t0) * 1000.0)

        arr = np.asarray(timings_ms, dtype=np.float64)
        p50 = float(np.percentile(arr, 50))
        p99 = float(np.percentile(arr, 99))
        throughput = float(bs) * 1000.0 / max(float(np.mean(arr)), 1e-9)
        results[int(bs)] = {
            "ms_p50": p50,
            "ms_p99": p99,
            "throughput_per_s": throughput,
        }
        _log.info("benchmark_result", batch=bs, p50_ms=round(p50, 3),
                  p99_ms=round(p99, 3), throughput=round(throughput, 1))
    return results


def _make_runner(artifact: Any, is_onnx: bool) -> Any:
    if is_onnx:
        session = artifact

        def _run_onnx(tokens: torch.Tensor, token_mask: torch.Tensor) -> Any:
            feeds = {
                "tokens": tokens.detach().cpu().numpy(),
                "token_mask": token_mask.detach().cpu().numpy(),
            }
            return session.run(["output"], feeds)[0]

        return _run_onnx

    module: torch.nn.Module = artifact

    @torch.no_grad()
    def _run_torch(tokens: torch.Tensor, token_mask: torch.Tensor) -> Any:
        out = module(tokens, token_mask)
        if isinstance(out, torch.Tensor) and out.is_cuda:
            torch.cuda.synchronize()
        return out

    return _run_torch


__all__ = ["benchmark_latency"]
