"""ONNX + TorchScript export of the ChannelHub encoder wrapper."""

from __future__ import annotations

import contextlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from msg_embedding.core.logging import get_logger

from .wrapper import EncoderWrapper, build_encoder_wrapper

_log = get_logger(__name__)


TOKEN_LAYOUT: dict[int, str] = {
    0: "pdp",
    1: "srs1", 2: "srs2", 3: "srs3", 4: "srs4",
    5: "pmi1", 6: "pmi2", 7: "pmi3", 8: "pmi4",
    9: "dft1", 10: "dft2", 11: "dft3", 12: "dft4",
    13: "rsrp_srs", 14: "rsrp_cb", 15: "cell_rsrp",
}

DEFAULT_SEQ_LEN: int = 16
DEFAULT_TOKEN_DIM: int = 128
DEFAULT_LATENT_OUT_DIM: int = 16


def export_onnx(
    ckpt_path: str | Path,
    output_path: str | Path,
    opset: int = 17,
    dynamic_batch: bool = True,
    validate: bool = True,
    seq_len: int = DEFAULT_SEQ_LEN,
    token_dim: int = DEFAULT_TOKEN_DIM,
    use_adapter: bool = False,
) -> Path:
    """Export the encoder wrapper to an ONNX graph."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = build_encoder_wrapper(
        ckpt_path=ckpt_path, device="cpu", use_adapter=use_adapter
    )

    dummy_tokens = torch.randn(1, seq_len, token_dim, dtype=torch.float32)
    dummy_mask = torch.zeros(1, seq_len, dtype=torch.bool)

    dynamic_axes: dict[str, dict[int, str]] | None = None
    if dynamic_batch:
        dynamic_axes = {
            "tokens": {0: "batch"},
            "token_mask": {0: "batch"},
            "output": {0: "batch"},
        }

    _log.info("onnx_export_start", path=str(output_path), opset=opset,
              dynamic_batch=dynamic_batch)
    torch.onnx.export(
        wrapper,
        (dummy_tokens, dummy_mask),
        str(output_path),
        input_names=["tokens", "token_mask"],
        output_names=["output"],
        opset_version=int(opset),
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    if validate:
        _validate_onnx(output_path, wrapper, dummy_tokens, dummy_mask)
    return output_path


def _validate_onnx(
    onnx_path: Path,
    wrapper: EncoderWrapper,
    dummy_tokens: torch.Tensor,
    dummy_mask: torch.Tensor,
) -> None:
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        _log.warning("onnxruntime_not_installed")
        return

    with torch.no_grad():
        torch_out = wrapper(dummy_tokens, dummy_mask).detach().cpu().numpy()

    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    feeds = {
        "tokens": dummy_tokens.numpy(),
        "token_mask": dummy_mask.numpy(),
    }
    onnx_out = sess.run(["output"], feeds)[0]

    import numpy as np
    if not np.allclose(onnx_out, torch_out, atol=1e-4):
        diff = float(np.abs(onnx_out - torch_out).max())
        raise RuntimeError(
            f"ONNX output mismatch vs torch (max abs diff={diff:.3e})"
        )
    _log.info("onnx_validation_passed")


def export_torchscript(
    ckpt_path: str | Path,
    output_path: str | Path,
    use_adapter: bool = False,
    seq_len: int = DEFAULT_SEQ_LEN,
    token_dim: int = DEFAULT_TOKEN_DIM,
    method: str = "trace",
) -> Path:
    """Export the encoder wrapper as a TorchScript module."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = build_encoder_wrapper(
        ckpt_path=ckpt_path, device="cpu", use_adapter=use_adapter
    )
    wrapper.eval()

    if method == "script":
        scripted = torch.jit.script(wrapper)
    else:
        dummy_tokens = torch.randn(1, seq_len, token_dim, dtype=torch.float32)
        dummy_mask = torch.zeros(1, seq_len, dtype=torch.bool)
        scripted = torch.jit.trace(
            wrapper, (dummy_tokens, dummy_mask),
            strict=False, check_trace=False,
        )

    scripted.save(str(output_path))
    _log.info("torchscript_exported", path=str(output_path), method=method)
    return output_path


def _get_git_sha(repo_root: Path | None = None) -> str:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return sha.decode("ascii").strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return ""


def _read_training_metadata(ckpt_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    run_dir = ckpt_path.parent
    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            out["training_metadata"] = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        with contextlib.suppress(OSError):
            out["training_config_path"] = str(cfg_path)
    return out


def _collect_protocol_spec() -> dict[str, Any]:
    try:
        from msg_embedding.core.protocol_spec import PROTOCOL_SPEC
    except Exception:
        return {}
    spec_serialisable: dict[str, Any] = {}
    for name, entry in PROTOCOL_SPEC.items():
        ser: dict[str, Any] = {}
        for k, v in entry.items():
            ser[k] = list(v) if isinstance(v, tuple) else v
        spec_serialisable[name] = ser
    return spec_serialisable


def write_metadata(
    ckpt_path: str | Path,
    output_dir: str | Path,
    model_name: str = "msg-embedding",
    version: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Emit metadata.json describing the deployed artifact."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(ckpt_path)

    if version is None or not version:
        sha = _get_git_sha()
        version = sha if sha else time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    payload: dict[str, Any] = {
        "model_name": model_name,
        "version": version,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ckpt_path": str(ckpt_path),
        "input_spec": {
            "tokens": {
                "shape": ["batch", DEFAULT_SEQ_LEN, DEFAULT_TOKEN_DIM],
                "dtype": "float32",
            },
            "token_mask": {
                "shape": ["batch", DEFAULT_SEQ_LEN],
                "dtype": "bool",
            },
            "feature_extractor": {
                "seq_len": DEFAULT_SEQ_LEN,
                "token_dim": DEFAULT_TOKEN_DIM,
                "layout": TOKEN_LAYOUT,
            },
        },
        "output_spec": {
            "shape": ["batch", DEFAULT_LATENT_OUT_DIM],
            "dtype": "float32",
        },
        "token_layout": TOKEN_LAYOUT,
        "normalization": {
            "protocol_spec": _collect_protocol_spec(),
        },
    }
    payload.update(_read_training_metadata(ckpt_path))
    if extra:
        payload["extra"] = dict(extra)

    target = output_dir / "metadata.json"
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False),
        encoding="utf-8",
    )
    _log.info("deployment_metadata_written", path=str(target))
    return target


__all__ = [
    "DEFAULT_LATENT_OUT_DIM",
    "DEFAULT_SEQ_LEN",
    "DEFAULT_TOKEN_DIM",
    "TOKEN_LAYOUT",
    "export_onnx",
    "export_torchscript",
    "write_metadata",
]
