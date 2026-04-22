"""CLI entry point for downstream evaluation of a trained ChannelMAE checkpoint.

Example
-------
::

    python scripts/run_eval.py \
        eval.ckpt=artifacts/<run-id>/ckpt_best.pth \
        eval.output_dir=reports \
        eval.manifest=artifacts/manifest.parquet \
        eval.split=test \
        eval.limit=128

Notes
-----
The script expects the Hydra ``eval`` group to expose (at minimum)
``ckpt``, ``output_dir`` and ``manifest`` — legacy ``eval/default.yaml``
is a stub with just ``enabled: false`` so callers pass the required
values via overrides until a concrete default.yaml lands.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:  # pragma: no cover - Hydra is a first-party dependency
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ImportError as exc:  # pragma: no cover
    print(f"hydra / omegaconf missing: {exc}", file=sys.stderr)
    raise

from msg_embedding.data.dataset import ChannelDataset
from msg_embedding.data.manifest import Manifest
from msg_embedding.eval.runner import run_eval


def _build_dataset(cfg: Any) -> ChannelDataset:
    manifest_path = Path(str(cfg.eval.manifest))
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found at {manifest_path}")
    manifest = Manifest(manifest_path)
    split = str(getattr(cfg.eval, "split", "test"))
    source_filter = getattr(cfg.eval, "source_filter", None)
    link_filter = str(getattr(cfg.eval, "link_filter", "both"))
    return ChannelDataset(
        manifest=manifest,
        split=split,
        source_filter=list(source_filter) if source_filter else None,
        link_filter=link_filter,  # type: ignore[arg-type]
    )


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra-decorated entry that runs the downstream eval pipeline."""
    eval_cfg_section = getattr(cfg, "eval", None)
    if eval_cfg_section is None or not getattr(eval_cfg_section, "ckpt", None):
        raise RuntimeError(
            "eval.ckpt is required — pass via Hydra override, e.g. "
            "'eval.ckpt=artifacts/<run-id>/ckpt_best.pth'"
        )
    ckpt_path = Path(str(cfg.eval.ckpt))
    output_dir = Path(str(getattr(cfg.eval, "output_dir", "reports")))
    device = str(getattr(cfg.eval, "device", "cpu"))

    dataset = _build_dataset(cfg)

    eval_cfg = {
        "k_neighbors": int(getattr(cfg.eval, "k_neighbors", 10)),
        "knn_k": int(getattr(cfg.eval, "knn_k", 5)),
        "limit": getattr(cfg.eval, "limit", None),
        "run_id": getattr(cfg.eval, "run_id", None),
    }
    eval_cfg["hydra_eval_block"] = OmegaConf.to_container(cfg.eval, resolve=True)

    result = run_eval(
        ckpt_path=ckpt_path,
        dataset=dataset,
        output_dir=output_dir,
        eval_cfg=eval_cfg,
        device=device,
    )
    print(f"eval done -> run_id={result.run_id}  tw={result.tw:.3f}  ct={result.ct:.3f}")


if __name__ == "__main__":
    main()
