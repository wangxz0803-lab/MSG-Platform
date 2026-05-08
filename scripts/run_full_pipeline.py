"""Full pipeline: Bridge -> Manifest -> Train -> Infer -> Analyze -> Report.

Reads collected samples from artifacts/collect_*/, runs bridge conversion,
registers to manifest, trains the MAE model, runs inference, generates
analysis and PPT report.

Usage:
    python scripts/run_full_pipeline.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from msg_embedding.data.bridge import _build_feat_dict
from msg_embedding.data.contract import ChannelSample
from msg_embedding.data.manifest import Manifest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_samples(collect_dir: Path, max_samples: int = 0) -> list[ChannelSample]:
    pts = sorted(collect_dir.glob("sample_*.pt"))
    if max_samples > 0:
        pts = pts[:max_samples]
    samples = []
    for p in pts:
        d = torch.load(p, map_location="cpu", weights_only=False)
        samples.append(ChannelSample.from_dict(d))
    return samples


def _bridge_samples(
    samples: list[ChannelSample],
    output_dir: Path,
    source_name: str,
) -> list[dict]:
    """Convert samples to bridge feature dicts and save as .pt shards."""
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 100
    manifest_rows: list[dict] = []

    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        feats_clean = []

        for i, s in enumerate(chunk):
            try:
                fd, ctx = _build_feat_dict(s, use_legacy_pmi=False)
                feats_clean.append(fd)

                row = {
                    "uuid": s.sample_id,
                    "source": source_name,
                    "shard_id": f"shard_{start // batch_size:04d}",
                    "sample_id": s.sample_id,
                    "stage": "featured",
                    "status": "success",
                    "link": s.link,
                    "snr_dB": float(s.snr_dB),
                    "sir_dB": float(s.sir_dB) if s.sir_dB is not None else None,
                    "sinr_dB": float(s.sinr_dB),
                    "num_cells": int(s.meta.get("num_cells", 1)),
                    "serving_cell_id": int(s.serving_cell_id),
                    "ue_x": float(s.ue_position[0]) if s.ue_position is not None else None,
                    "ue_y": float(s.ue_position[1]) if s.ue_position is not None else None,
                    "ue_z": float(s.ue_position[2]) if s.ue_position is not None else None,
                    "channel_est_mode": s.channel_est_mode,
                    "split": "unassigned",
                    "path": str(output_dir / f"shard_{start // batch_size:04d}.pt"),
                }
                manifest_rows.append(row)
            except Exception as e:
                print(f"  Bridge error sample {start + i}: {e}")
                continue

        if feats_clean:
            stacked = {}
            for key in feats_clean[0]:
                stacked[key] = torch.cat([f[key] for f in feats_clean], dim=0)

            shard_path = output_dir / f"shard_{start // batch_size:04d}.pt"
            torch.save(stacked, shard_path)

        pct = min(100, (start + len(chunk)) / len(samples) * 100)
        print(f"  Bridge [{source_name}] {pct:.0f}% ({start + len(chunk)}/{len(samples)})")

    return manifest_rows


def _compute_statistics(samples: list[ChannelSample]) -> dict:
    """Compute distribution statistics for a set of samples."""
    snrs = [s.snr_dB for s in samples]
    sinrs = [s.sinr_dB for s in samples]
    powers = [float(np.mean(np.abs(s.h_serving_true) ** 2)) for s in samples]

    stats: dict[str, Any] = {
        "count": len(samples),
        "snr_dB": {
            "mean": float(np.mean(snrs)),
            "std": float(np.std(snrs)),
            "min": float(np.min(snrs)),
            "max": float(np.max(snrs)),
            "p25": float(np.percentile(snrs, 25)),
            "p50": float(np.percentile(snrs, 50)),
            "p75": float(np.percentile(snrs, 75)),
        },
        "sinr_dB": {
            "mean": float(np.mean(sinrs)),
            "std": float(np.std(sinrs)),
            "min": float(np.min(sinrs)),
            "max": float(np.max(sinrs)),
            "p25": float(np.percentile(sinrs, 25)),
            "p50": float(np.percentile(sinrs, 50)),
            "p75": float(np.percentile(sinrs, 75)),
        },
        "channel_power": {
            "mean": float(np.mean(powers)),
            "std": float(np.std(powers)),
            "min": float(np.min(powers)),
            "max": float(np.max(powers)),
        },
        "links": dict(zip(*np.unique([s.link for s in samples], return_counts=True), strict=False)),
        "channel_est_modes": dict(
            zip(*np.unique([s.channel_est_mode for s in samples], return_counts=True), strict=False)
        ),
    }

    ue_positions = [s.ue_position for s in samples if s.ue_position is not None]
    if ue_positions:
        pos = np.array(ue_positions)
        stats["ue_position"] = {
            "x_range": [float(pos[:, 0].min()), float(pos[:, 0].max())],
            "y_range": [float(pos[:, 1].min()), float(pos[:, 1].max())],
        }

    return stats


if __name__ == "__main__":
    print("=" * 60)
    print("ChannelHub Full Pipeline")
    print("=" * 60)

    sources = {
        "internal_sim": _PROJECT_ROOT / "artifacts" / "collect_internal_sim",
        "sionna_rt": _PROJECT_ROOT / "artifacts" / "collect_sionna_rt",
        "quadriga_real": _PROJECT_ROOT / "artifacts" / "collect_quadriga_real",
    }

    bridge_out = _PROJECT_ROOT / "artifacts" / "bridge_out_5k"
    bridge_out.mkdir(parents=True, exist_ok=True)

    all_manifest_rows: list[dict] = []
    all_stats: dict[str, Any] = {}

    for name, collect_dir in sources.items():
        print(f"\n{'---' * 14}")
        print(f"Loading {name} from {collect_dir}")

        summary_path = collect_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            print(
                f"  Collection: {summary.get('saved', '?')} samples, "
                f"{summary.get('elapsed_secs', '?')}s"
            )

        samples = _load_samples(collect_dir, max_samples=5000)
        print(f"  Loaded {len(samples)} samples")

        if not samples:
            print("  SKIP: no samples found")
            continue

        meta_path = collect_dir / "run_meta.json"
        run_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        stats = _compute_statistics(samples)
        stats["config"] = run_meta.get("config", {})
        stats["source"] = name
        all_stats[name] = stats
        print(f"  SNR: {stats['snr_dB']['mean']:.1f} +/- {stats['snr_dB']['std']:.1f} dB")
        print(f"  SINR: {stats['sinr_dB']['mean']:.1f} +/- {stats['sinr_dB']['std']:.1f} dB")

        src_bridge_dir = bridge_out / name
        rows = _bridge_samples(samples, src_bridge_dir, name)
        all_manifest_rows.extend(rows)
        print(f"  Bridged {len(rows)} samples -> {src_bridge_dir}")

    print(f"\n{'---' * 14}")
    print("Building manifest...")
    manifest_path = bridge_out / "manifest.parquet"
    manifest = Manifest(str(manifest_path))
    manifest.append(all_manifest_rows)
    manifest.compute_split(strategy="random", seed=42, ratios=(0.8, 0.1, 0.1))
    manifest.save()
    total = len(manifest.df)
    print(f"  Manifest: {total} rows -> {manifest_path}")
    print(
        f"  Splits: train={len(manifest.query(split='train'))}, "
        f"val={len(manifest.query(split='val'))}, "
        f"test={len(manifest.query(split='test'))}"
    )

    stats_path = bridge_out / "dataset_stats.json"
    stats_path.write_text(json.dumps(all_stats, indent=2, default=str), encoding="utf-8")
    print(f"  Stats -> {stats_path}")

    print(f"\n{'=' * 60}")
    print("Pipeline Phase 1 (Collect + Bridge + Manifest) DONE")
    print(f"{'=' * 60}")
