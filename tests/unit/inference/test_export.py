"""Tests for inference export."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from msg_embedding.inference.export import (
    TOKEN_LAYOUT,
    export_torchscript,
)
from msg_embedding.inference.wrapper import build_encoder_wrapper


def test_export_torchscript_matches_eager(
    channel_mae_ckpt: Path, tmp_path: Path
) -> None:
    ts_path = tmp_path / "model.ts"
    export_torchscript(ckpt_path=channel_mae_ckpt, output_path=ts_path)
    assert ts_path.exists()

    loaded = torch.jit.load(str(ts_path), map_location="cpu").eval()
    wrapper = build_encoder_wrapper(ckpt_path=channel_mae_ckpt, device="cpu")
    wrapper.eval()

    torch.manual_seed(0)
    tokens = torch.randn(2, 16, 128, dtype=torch.float32)
    token_mask = torch.zeros(2, 16, dtype=torch.bool)
    token_mask[0, 15] = True

    with torch.no_grad():
        eager = wrapper(tokens, token_mask)
        scripted = loaded(tokens, token_mask)

    assert scripted.shape == eager.shape == (2, 16)
    assert torch.allclose(scripted, eager, atol=1e-5)


def test_export_onnx_optional(channel_mae_ckpt: Path, tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from msg_embedding.inference.export import export_onnx

    onnx_path = tmp_path / "model.onnx"
    export_onnx(
        ckpt_path=channel_mae_ckpt,
        output_path=onnx_path,
        opset=17,
        dynamic_batch=True,
        validate=True,
    )
    assert onnx_path.exists()


def test_token_layout_is_canonical() -> None:
    assert TOKEN_LAYOUT[0] == "pdp"
    for i, name in enumerate(("srs1", "srs2", "srs3", "srs4"), start=1):
        assert TOKEN_LAYOUT[i] == name
    for i, name in enumerate(("pmi1", "pmi2", "pmi3", "pmi4"), start=5):
        assert TOKEN_LAYOUT[i] == name
    for i, name in enumerate(("dft1", "dft2", "dft3", "dft4"), start=9):
        assert TOKEN_LAYOUT[i] == name
    assert TOKEN_LAYOUT[13] == "rsrp_srs"
    assert TOKEN_LAYOUT[14] == "rsrp_cb"
    assert TOKEN_LAYOUT[15] == "cell_rsrp"
