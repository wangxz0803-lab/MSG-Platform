"""CLI entry point for dataset export.

Exports channel data from the manifest to portable packages for external
training platforms.

Usage::

    python scripts/run_dataset_export.py export.format=hdf5 export.split=train
    python scripts/run_dataset_export.py export.format=webdataset export.split=train export.shard_size=500

Progress protocol: emits ``[progress] pct=<float> step=<name>`` lines
for the worker TaskRunner to parse.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

from msg_embedding.core.logging import get_logger

_log = get_logger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_cli_overrides(args: list[str]) -> dict[str, Any]:
    """Parse ``key=value`` pairs from CLI."""
    overrides: dict[str, Any] = {}
    for arg in args:
        if "=" not in arg:
            continue
        key, val = arg.split("=", 1)
        key = key.removeprefix("export.").removeprefix("data.")
        if val.lower() == "true":
            val = True
        elif val.lower() == "false":
            val = False
        elif val.lower() in ("null", "none"):
            val = None
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
        overrides[key] = val
    return overrides


def _progress_printer(pct: float) -> None:
    """Emit progress line for the worker to parse."""
    print(f"[progress] pct={pct * 100:.1f} step=exporting", flush=True)


def main() -> int:
    overrides = _parse_cli_overrides(sys.argv[1:])
    _log.info("dataset_export_start", overrides=overrides)

    try:
        from msg_embedding.data.export import ExportConfig, export_dataset
        from msg_embedding.data.manifest import Manifest

        manifest_path = overrides.pop("manifest_path", None)
        if manifest_path is None:
            manifest_path = _PROJECT_ROOT / "bridge_out" / "manifest.parquet"
        manifest = Manifest(manifest_path)

        output_dir = overrides.pop("output_dir", None) or str(_PROJECT_ROOT / "exports")

        config = ExportConfig(
            format=overrides.get("format", "hdf5"),
            split=overrides.get("split", "train"),
            source_filter=overrides.get("source_filter"),
            link_filter=overrides.get("link_filter"),
            min_snr=overrides.get("min_snr"),
            max_snr=overrides.get("max_snr"),
            output_dir=output_dir,
            export_name=overrides.get("export_name"),
            shard_size=int(overrides.get("shard_size", 1000)),
            include_interferers=bool(overrides.get("include_interferers", False)),
        )

        print(f"[progress] pct=0 step=starting export ({config.format}, split={config.split})",
              flush=True)

        result = export_dataset(manifest, config, progress_cb=_progress_printer)

        print(f"[progress] pct=100 step=done", flush=True)
        _log.info(
            "dataset_export_done",
            path=str(result.output_path),
            num_samples=result.num_samples,
            total_bytes=result.total_bytes,
            format=result.format,
        )
        print(
            f"Export complete: {result.num_samples} samples, "
            f"{result.total_bytes / 1024 / 1024:.1f} MB -> {result.output_path}"
        )
        return 0

    except Exception as exc:
        _log.exception("dataset_export_failed", error=str(exc))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
