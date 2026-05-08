"""Multi-format dataset export engine.

Exports ChannelSample data from the manifest to portable, self-contained
packages that external training platforms can consume directly.

Supported formats
-----------------
* **HDF5** (``.h5``): one group per sample, gzip-compressed arrays, manifest
  metadata in root attrs.  Cross-language, numpy-native.
* **WebDataset** (``.tar``): tar shards with pickle-serialised dicts, ready for
  ``webdataset.WebDataset`` streaming.
* **pt_dir**: directory of raw ``.pt`` files plus a ``manifest.parquet`` subset.

All formats embed a ``README.json`` (or root attrs for HDF5) containing:
contract version, split version, export timestamp, sample counts, schema
description, and channel dimension summary.
"""

from __future__ import annotations

import json
import pickle
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np

from .contract import CONTRACT_VERSION, ChannelSample
from .manifest import Manifest

ExportFormat = Literal["hdf5", "webdataset", "pt_dir"]


@dataclass
class ExportConfig:
    """Configuration for a dataset export job."""

    format: ExportFormat = "hdf5"
    split: str | None = "train"
    source_filter: str | None = None
    link_filter: str | None = None
    min_snr: float | None = None
    max_snr: float | None = None
    output_dir: str | Path = "exports"
    export_name: str | None = None
    shard_size: int = 1000
    include_interferers: bool = False


@dataclass
class ExportResult:
    """Result of a completed export."""

    output_path: Path
    format: ExportFormat
    num_samples: int
    total_bytes: int
    split_version: int
    contract_version: str
    manifest_subset_path: Path | None = None
    shard_paths: list[Path] = field(default_factory=list)


def _build_readme(
    manifest: Manifest,
    config: ExportConfig,
    num_samples: int,
    channel_dims: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata dict embedded in every export package."""
    split_info = manifest.get_split_info()
    return {
        "msg_embedding_export": True,
        "contract_version": CONTRACT_VERSION,
        "split_version": split_info.get("version", 0),
        "split_locked": split_info.get("locked", False),
        "export_split": config.split,
        "export_format": config.format,
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "num_samples": num_samples,
        "source_filter": config.source_filter,
        "link_filter": config.link_filter,
        "include_interferers": config.include_interferers,
        "channel_dims": channel_dims or {},
        "schema_description": {
            "h_serving_true": "[T, RB, BS_ant, UE_ant] complex64 — ideal serving channel",
            "h_serving_est": "[T, RB, BS_ant, UE_ant] complex64 — estimated channel with interference",
            "h_interferers": "[K-1, T, RB, BS_ant, UE_ant] complex64 — interferer channels (optional)",
            "snr_dB": "Signal-to-noise ratio",
            "sir_dB": "Signal-to-interference ratio",
            "sinr_dB": "Signal-to-interference-plus-noise ratio",
            "ue_position": "[x, y, z] float64 metres — UE location (label for evaluation)",
            "link": "UL or DL",
            "source": "Data source (internal_sim, sionna_rt, quadriga_real, field)",
        },
        "usage": {
            "python_hdf5": "import h5py; f = h5py.File('export.h5'); sample = f['samples/0']",
            "python_pt": "from msg_embedding.data.contract import ChannelSample; s = ChannelSample.from_pt_file('sample.pt')",
            "python_webdataset": "import webdataset as wds; ds = wds.WebDataset('shard-{000000..000010}.tar')",
        },
    }


def _iter_samples(
    manifest: Manifest,
    config: ExportConfig,
    progress_cb: Any | None = None,
) -> Iterator[tuple[int, ChannelSample]]:
    """Iterate over filtered manifest rows, yielding (index, ChannelSample)."""
    df = manifest.df.copy()

    if config.split:
        df = df[df["split"] == config.split]
    if config.source_filter:
        df = df[df["source"] == config.source_filter]
    if config.link_filter:
        df = df[df["link"] == config.link_filter]
    if config.min_snr is not None:
        df = df[df["snr_dB"] >= config.min_snr]
    if config.max_snr is not None:
        df = df[df["snr_dB"] <= config.max_snr]

    df = df[df["status"].isin(["succeeded", "ready"])]
    df = df[df["path"].notna()]

    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        path = Path(row["path"])
        if not path.exists():
            continue
        try:
            sample = ChannelSample.from_pt_file(path)
        except Exception:
            continue

        if not config.include_interferers:
            sample = sample.model_copy(
                update={"h_interferers": None, "interference_signal": None}
            )
        yield i, sample

        if progress_cb and total > 0:
            progress_cb((i + 1) / total)


def _get_channel_dims(sample: ChannelSample) -> dict[str, Any]:
    """Extract channel dimension info from a sample."""
    T, RB, BS_ant, UE_ant = sample.h_serving_true.shape
    dims = {"T": T, "RB": RB, "BS_ant": BS_ant, "UE_ant": UE_ant}
    if sample.h_interferers is not None:
        dims["K_minus_1"] = sample.h_interferers.shape[0]
    return dims


def export_hdf5(
    manifest: Manifest,
    config: ExportConfig,
    progress_cb: Any | None = None,
) -> ExportResult:
    """Export dataset to a single HDF5 file."""
    import h5py

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = config.export_name or f"msg_export_{config.split or 'all'}_{config.format}"
    out_path = out_dir / f"{name}.h5"

    channel_dims: dict[str, Any] = {}
    count = 0

    with h5py.File(out_path, "w") as f:
        samples_grp = f.create_group("samples")

        for i, sample in _iter_samples(manifest, config, progress_cb):
            grp = samples_grp.create_group(str(count))
            sample.to_hdf5_group(grp)
            if count == 0:
                channel_dims = _get_channel_dims(sample)
            count += 1

        readme = _build_readme(manifest, config, count, channel_dims)
        f.attrs["readme_json"] = json.dumps(readme, ensure_ascii=False)
        f.attrs["contract_version"] = CONTRACT_VERSION
        f.attrs["num_samples"] = count
        f.attrs["split"] = config.split or "all"
        f.attrs["split_version"] = manifest.split_version

        manifest_df = manifest.df.copy()
        if config.split:
            manifest_df = manifest_df[manifest_df["split"] == config.split]
        scalars_grp = f.create_group("manifest_scalars")
        for col in ("uuid", "source", "link", "split", "snr_dB", "sir_dB",
                     "sinr_dB", "ue_x", "ue_y", "ue_z", "channel_est_mode",
                     "serving_cell_id"):
            if col in manifest_df.columns:
                vals = manifest_df[col].values
                if vals.dtype == object:
                    vals = np.array([str(v) if v is not None else "" for v in vals])
                    scalars_grp.create_dataset(col, data=vals.astype("S"))
                else:
                    vals = np.nan_to_num(vals.astype(float), nan=-999.0)
                    scalars_grp.create_dataset(col, data=vals)

    total_bytes = out_path.stat().st_size

    manifest_path = out_dir / f"{name}_manifest.parquet"
    _export_manifest_subset(manifest, config, manifest_path)

    return ExportResult(
        output_path=out_path,
        format="hdf5",
        num_samples=count,
        total_bytes=total_bytes,
        split_version=manifest.split_version,
        contract_version=CONTRACT_VERSION,
        manifest_subset_path=manifest_path,
    )


def export_webdataset(
    manifest: Manifest,
    config: ExportConfig,
    progress_cb: Any | None = None,
) -> ExportResult:
    """Export dataset as WebDataset tar shards."""
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = config.export_name or f"msg_export_{config.split or 'all'}_{config.format}"
    shard_dir = out_dir / name
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_paths: list[Path] = []
    count = 0
    shard_idx = 0
    current_tar: tarfile.TarFile | None = None
    in_shard = 0

    def _open_shard() -> tarfile.TarFile:
        nonlocal shard_idx
        p = shard_dir / f"shard-{shard_idx:06d}.tar"
        shard_paths.append(p)
        shard_idx += 1
        return tarfile.open(p, "w")

    channel_dims: dict[str, Any] = {}

    for i, sample in _iter_samples(manifest, config, progress_cb):
        if current_tar is None or in_shard >= config.shard_size:
            if current_tar is not None:
                current_tar.close()
            current_tar = _open_shard()
            in_shard = 0

        payload = sample.to_dict()
        data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        buf = BytesIO(data)

        info = tarfile.TarInfo(name=f"{count:08d}.pkl")
        info.size = len(data)
        current_tar.addfile(info, buf)

        if count == 0:
            channel_dims = _get_channel_dims(sample)

        count += 1
        in_shard += 1

    if current_tar is not None:
        current_tar.close()

    readme = _build_readme(manifest, config, count, channel_dims)
    readme_path = shard_dir / "README.json"
    readme_path.write_text(json.dumps(readme, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest_path = shard_dir / "manifest.parquet"
    _export_manifest_subset(manifest, config, manifest_path)

    total_bytes = sum(p.stat().st_size for p in shard_paths)
    total_bytes += readme_path.stat().st_size + manifest_path.stat().st_size

    return ExportResult(
        output_path=shard_dir,
        format="webdataset",
        num_samples=count,
        total_bytes=total_bytes,
        split_version=manifest.split_version,
        contract_version=CONTRACT_VERSION,
        manifest_subset_path=manifest_path,
        shard_paths=shard_paths,
    )


def export_pt_dir(
    manifest: Manifest,
    config: ExportConfig,
    progress_cb: Any | None = None,
) -> ExportResult:
    """Export dataset as a directory of raw .pt files."""
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = config.export_name or f"msg_export_{config.split or 'all'}_{config.format}"
    export_dir = out_dir / name
    export_dir.mkdir(parents=True, exist_ok=True)

    samples_dir = export_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    channel_dims: dict[str, Any] = {}
    count = 0
    total_bytes = 0

    df = manifest.df.copy()
    if config.split:
        df = df[df["split"] == config.split]
    if config.source_filter:
        df = df[df["source"] == config.source_filter]
    if config.link_filter:
        df = df[df["link"] == config.link_filter]
    df = df[df["status"].isin(["succeeded", "ready"])]
    df = df[df["path"].notna()]
    total = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        src = Path(row["path"])
        if not src.exists():
            continue
        dst = samples_dir / f"{count:08d}.pt"
        shutil.copy2(src, dst)
        total_bytes += dst.stat().st_size
        if count == 0:
            try:
                s = ChannelSample.from_pt_file(src)
                channel_dims = _get_channel_dims(s)
            except Exception:
                pass
        count += 1
        if progress_cb and total > 0:
            progress_cb((idx + 1) / total)

    readme = _build_readme(manifest, config, count, channel_dims)
    readme_path = export_dir / "README.json"
    readme_path.write_text(json.dumps(readme, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest_path = export_dir / "manifest.parquet"
    _export_manifest_subset(manifest, config, manifest_path)

    return ExportResult(
        output_path=export_dir,
        format="pt_dir",
        num_samples=count,
        total_bytes=total_bytes,
        split_version=manifest.split_version,
        contract_version=CONTRACT_VERSION,
        manifest_subset_path=manifest_path,
    )


def _export_manifest_subset(
    manifest: Manifest,
    config: ExportConfig,
    output_path: Path,
) -> None:
    """Write a filtered manifest parquet for the export."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .manifest import MANIFEST_SCHEMA

    df = manifest.df.copy()
    if config.split:
        df = df[df["split"] == config.split]
    if config.source_filter:
        df = df[df["source"] == config.source_filter]
    if config.link_filter:
        df = df[df["link"] == config.link_filter]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, schema=MANIFEST_SCHEMA, preserve_index=False)
    pq.write_table(table, output_path)


def export_dataset(
    manifest: Manifest,
    config: ExportConfig,
    progress_cb: Any | None = None,
) -> ExportResult:
    """Main entry point: export dataset in the specified format."""
    exporters = {
        "hdf5": export_hdf5,
        "webdataset": export_webdataset,
        "pt_dir": export_pt_dir,
    }
    exporter = exporters.get(config.format)
    if exporter is None:
        raise ValueError(f"unsupported export format: {config.format}")
    return exporter(manifest, config, progress_cb)


__all__ = [
    "ExportConfig",
    "ExportFormat",
    "ExportResult",
    "export_dataset",
    "export_hdf5",
    "export_webdataset",
    "export_pt_dir",
]
