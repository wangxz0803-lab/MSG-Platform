"""CLI entry point for ChannelHub batch inference.

Typical usage::

    python scripts/run_infer.py \
        infer.ckpt=artifacts/<run>/ckpt_best.pth \
        infer.input=bridge_out/ \
        infer.output=embeddings.parquet \
        infer.batch_size=64 infer.half=true

Notes
-----
* ``infer.input`` is expected to point at a directory of bridge-produced
  ``.pt`` files (see :mod:`msg_embedding.data.bridge`). Dataset-mode is
  available programmatically via :meth:`BatchInferenceRunner.infer_dataset`.
* ``infer.half=true`` is honoured only when CUDA is available; otherwise
  the runner silently falls back to fp32 on CPU.
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

from msg_embedding.inference.batch import BatchInferenceRunner


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry: run :class:`BatchInferenceRunner` on a bridge_out directory."""
    infer_cfg = getattr(cfg, "infer", None)
    if infer_cfg is None:
        raise RuntimeError("config has no `infer` group — check configs/infer/default.yaml")
    if not getattr(infer_cfg, "ckpt", None):
        raise RuntimeError(
            "infer.ckpt is required — pass e.g. "
            "'infer.ckpt=artifacts/<run>/ckpt_best.pth'"
        )
    if not getattr(infer_cfg, "input", None):
        raise RuntimeError(
            "infer.input is required — pass a directory of bridge .pt files"
        )
    if not getattr(infer_cfg, "output", None):
        raise RuntimeError("infer.output is required — target .parquet path")

    runner = BatchInferenceRunner(
        ckpt_path=Path(str(infer_cfg.ckpt)),
        device=str(getattr(infer_cfg, "device", "cpu")),
        half=bool(getattr(infer_cfg, "half", False)),
        use_adapter=bool(getattr(infer_cfg, "use_adapter", False)),
    )
    out_path = runner.infer_directory(
        bridge_out_dir=Path(str(infer_cfg.input)),
        output_parquet_path=Path(str(infer_cfg.output)),
        batch_size=int(getattr(infer_cfg, "batch_size", 64)),
        num_workers=int(getattr(infer_cfg, "num_workers", 0)),
        glob=str(getattr(infer_cfg, "glob", "*.pt")),
    )
    print(f"infer done -> {out_path}")


if __name__ == "__main__":
    main()
