"""Tests for write_metadata."""

from __future__ import annotations

import json
from pathlib import Path

from msg_embedding.inference.export import write_metadata


def test_write_metadata_has_required_keys(
    channel_mae_ckpt: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "deploy"
    meta_path = write_metadata(
        ckpt_path=channel_mae_ckpt,
        output_dir=out_dir,
        model_name="msg-embedding",
        version="test-1.0",
    )
    assert meta_path.exists()
    payload = json.loads(meta_path.read_text(encoding="utf-8"))

    required_top = {
        "model_name", "version", "exported_at",
        "input_spec", "output_spec",
        "token_layout", "normalization",
    }
    missing = required_top - set(payload.keys())
    assert not missing, f"missing metadata keys: {missing}"

    assert payload["model_name"] == "msg-embedding"
    assert payload["version"] == "test-1.0"
    assert payload["output_spec"]["shape"][-1] == 16
    assert payload["output_spec"]["dtype"] == "float32"
    assert str(payload["token_layout"]["0"]) == "pdp"
    assert str(payload["token_layout"]["15"]) == "cell_rsrp"


def test_write_metadata_auto_version(
    channel_mae_ckpt: Path, tmp_path: Path
) -> None:
    meta_path = write_metadata(
        ckpt_path=channel_mae_ckpt, output_dir=tmp_path / "d"
    )
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["version"]


def test_write_metadata_includes_training_metadata(
    channel_mae_ckpt: Path, tmp_path: Path
) -> None:
    train_meta = {"best_val_loss": 0.123, "run_id": "synthetic"}
    sibling = channel_mae_ckpt.parent / "metadata.json"
    sibling.write_text(json.dumps(train_meta), encoding="utf-8")

    out_dir = tmp_path / "deploy2"
    meta_path = write_metadata(ckpt_path=channel_mae_ckpt, output_dir=out_dir)
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "training_metadata" in payload
    assert payload["training_metadata"]["run_id"] == "synthetic"
