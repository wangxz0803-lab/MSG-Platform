"""Integration: checkpoint -> inference -> export -> re-load verification."""

from __future__ import annotations

from pathlib import Path

import torch

from msg_embedding.inference.wrapper import EncoderWrapper


def test_encoder_wrapper_produces_normalized_output(trained_mae) -> None:
    """EncoderWrapper should produce L2-normalized latent vectors."""
    mae, feat_ext, ckpt_info = trained_mae
    stacked = ckpt_info["stacked_feats"]

    with torch.no_grad():
        tokens, _ = feat_ext(stacked)

    token_mask = torch.zeros(tokens.shape[0], tokens.shape[1], dtype=torch.bool)

    wrapper = EncoderWrapper(mae)
    wrapper.eval()
    with torch.no_grad():
        latent = wrapper(tokens, token_mask)

    assert latent.shape == (tokens.shape[0], 16)
    norms = torch.norm(latent, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_torchscript_export_roundtrip(trained_mae, tmp_path: Path) -> None:
    """Export to TorchScript, reload, and verify outputs match."""
    mae, feat_ext, ckpt_info = trained_mae
    stacked = ckpt_info["stacked_feats"]

    from msg_embedding.inference.export import export_torchscript

    ts_path = tmp_path / "model.ts"
    export_torchscript(
        ckpt_path=ckpt_info["path"],
        output_path=ts_path,
        use_adapter=False,
        method="trace",
    )
    assert ts_path.exists()

    loaded = torch.jit.load(str(ts_path))
    loaded.eval()

    with torch.no_grad():
        tokens, _ = feat_ext(stacked)
        token_mask = torch.zeros(tokens.shape[0], tokens.shape[1], dtype=torch.bool)

        original = EncoderWrapper(mae)
        original.eval()
        expected = original(tokens, token_mask)
        actual = loaded(tokens, token_mask)

    assert torch.allclose(expected, actual, atol=1e-5), (
        f"TorchScript output diverges: max diff = {(expected - actual).abs().max():.6f}"
    )


def test_metadata_export(trained_mae, tmp_path: Path) -> None:
    """write_metadata should produce a valid JSON file."""
    import json

    from msg_embedding.inference.export import write_metadata

    meta_path = write_metadata(ckpt_path=trained_mae[2]["path"], output_dir=tmp_path)
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "input_spec" in meta
    assert "model_name" in meta
    spec = meta["input_spec"]
    assert spec["feature_extractor"]["seq_len"] == 16
    assert spec["feature_extractor"]["token_dim"] == 128
