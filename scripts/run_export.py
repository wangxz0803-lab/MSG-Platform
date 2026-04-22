"""CLI entry point for ONNX / TorchScript export + deployment metadata.

Typical usage::

    python scripts/run_export.py \
        export.ckpt=artifacts/<run>/ckpt_best.pth \
        export.output_dir=artifacts/<run>/deploy/

Produces ``model.onnx``, ``model.ts`` and ``metadata.json`` under
``export.output_dir``. ONNX validation is attempted when
:mod:`onnxruntime` is installed; otherwise a warning is logged and the
script continues so air-gapped build environments still complete.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:  # pragma: no cover - Hydra is a first-party dependency
    import hydra
    from omegaconf import DictConfig
except ImportError as exc:  # pragma: no cover
    print(f"hydra / omegaconf missing: {exc}", file=sys.stderr)
    raise

from msg_embedding.inference.export import (
    export_onnx,
    export_torchscript,
    write_metadata,
)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry: run ONNX + TorchScript export and write metadata."""
    export_cfg = getattr(getattr(cfg, "infer", None), "export", None)
    if export_cfg is None:
        raise RuntimeError(
            "config has no `infer.export` block — check configs/infer/default.yaml"
        )
    ckpt = getattr(export_cfg, "ckpt", None)
    output_dir = getattr(export_cfg, "output_dir", None)
    if not ckpt:
        raise RuntimeError(
            "export.ckpt is required — pass 'export.ckpt=artifacts/<run>/ckpt_best.pth'"
        )
    if not output_dir:
        raise RuntimeError(
            "export.output_dir is required — pass 'export.output_dir=artifacts/<run>/deploy/'"
        )

    ckpt_path = Path(str(ckpt))
    out_dir = Path(str(output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    emit_onnx = bool(getattr(export_cfg, "emit_onnx", True))
    emit_ts = bool(getattr(export_cfg, "emit_torchscript", True))
    use_adapter = bool(getattr(export_cfg, "use_adapter", False))

    if emit_onnx:
        export_onnx(
            ckpt_path=ckpt_path,
            output_path=out_dir / "model.onnx",
            opset=int(getattr(export_cfg, "opset", 17)),
            dynamic_batch=bool(getattr(export_cfg, "dynamic_batch", True)),
            validate=bool(getattr(export_cfg, "validate", True)),
            use_adapter=use_adapter,
        )

    if emit_ts:
        export_torchscript(
            ckpt_path=ckpt_path,
            output_path=out_dir / "model.ts",
            use_adapter=use_adapter,
            method=str(getattr(export_cfg, "torchscript_method", "trace")),
        )

    meta_path = write_metadata(ckpt_path=ckpt_path, output_dir=out_dir)
    print(f"export done -> {out_dir} (metadata: {meta_path})")


if __name__ == "__main__":
    main()
