"""CLI entry point for bridge processing (raw samples → 16-Token features).

Reads raw ChannelSample .pt files produced by run_simulate.py, runs each
through the Bridge feature extraction pipeline, and writes processed token
files to bridge_out/.

Usage::

    python scripts/run_bridge.py sources=["internal_sim"] output_dir=bridge_out
    python scripts/run_bridge.py sources=["sionna_rt","internal_sim"] use_legacy_pmi=true

Progress protocol: emits ``[progress] pct=<float> step=<name>`` lines
for the worker TaskRunner to parse.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from msg_embedding.core.logging import get_logger
from msg_embedding.data.bridge import sample_to_features
from msg_embedding.data.contract import ChannelSample
from msg_embedding.features.extractor import FeatureExtractor

_log = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_cli_overrides() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for arg in sys.argv[1:]:
        if "=" not in arg:
            continue
        key, val = arg.split("=", 1)
        for prefix in ("bridge.", "data."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        if val.lower() in ("true", "false"):
            overrides[key] = val.lower() == "true"
        elif val in ("null", "none"):
            overrides[key] = None
        else:
            try:
                parsed = json.loads(val)
                overrides[key] = parsed
            except (json.JSONDecodeError, ValueError):
                try:
                    if "." in val or "e" in val.lower():
                        overrides[key] = float(val)
                    else:
                        overrides[key] = int(val)
                except ValueError:
                    overrides[key] = val
    return overrides


def _emit_progress(pct: float, step: str) -> None:
    print(f"[progress] pct={pct:.1f} step={step}", flush=True)


def _find_raw_samples(sources: list[str]) -> list[Path]:
    """Find all raw .pt sample files for the given sources.

    Looks in:
    1. artifacts/simulate_out/<source>/ directories
    2. artifacts/simulate_out/ (if samples have matching source in metadata)
    3. Any artifacts/ subdirectory containing sample_*.pt files
    """
    artifacts_dir = _PROJECT_ROOT / "artifacts"
    found: list[Path] = []

    if not artifacts_dir.is_dir():
        return found

    for sim_dir in sorted(artifacts_dir.rglob("simulate_out")):
        if not sim_dir.is_dir():
            continue
        for source in sources:
            source_dir = sim_dir / source
            if source_dir.is_dir():
                found.extend(sorted(source_dir.glob("sample_*.pt")))

        for pt_file in sorted(sim_dir.glob("sample_*.pt")):
            if pt_file not in found:
                try:
                    data = torch.load(pt_file, map_location="cpu", weights_only=False)
                    file_source = data.get("source", "")
                    if file_source in sources or not sources:
                        found.append(pt_file)
                except Exception:
                    pass

    if not found:
        for pt_file in sorted(artifacts_dir.rglob("sample_*.pt")):
            if pt_file not in found:
                found.append(pt_file)

    return found


def _load_sample(path: Path) -> ChannelSample:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, ChannelSample):
        return data
    if isinstance(data, dict):
        return ChannelSample.from_dict(data)
    raise TypeError(f"Unexpected type in {path}: {type(data)}")


def _is_already_processed(sample_id: str, output_dir: Path) -> bool:
    for pt_file in output_dir.glob(f"**/bridged_{sample_id}.pt"):
        return True
    return False


def main() -> None:
    overrides = _parse_cli_overrides()

    sources_raw = overrides.get("sources", [])
    if isinstance(sources_raw, str):
        sources_raw = [sources_raw]
    sources: list[str] = list(sources_raw)

    output_dir_name = str(overrides.get("output_dir", "bridge_out"))
    output_dir = _PROJECT_ROOT / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    skip_processed = bool(overrides.get("skip_processed", True))
    use_legacy_pmi = bool(overrides.get("use_legacy_pmi", False))
    job_id = overrides.get("project.job_id") or overrides.get("job_id")

    _log.info(
        "bridge_start",
        sources=sources,
        output_dir=str(output_dir),
        skip_processed=skip_processed,
        use_legacy_pmi=use_legacy_pmi,
    )

    _emit_progress(0.0, "scanning for raw samples")

    raw_files = _find_raw_samples(sources)
    if not raw_files:
        _emit_progress(0.0, "failed: no raw samples found")
        _log.error("bridge_no_samples", sources=sources)
        raise RuntimeError(
            f"No raw sample files found for sources={sources}. "
            f"Run data collection first to generate samples."
        )

    total = len(raw_files)
    _log.info("bridge_found_samples", count=total)
    _emit_progress(1.0, f"found {total} raw samples, initializing feature extractor")

    fe = FeatureExtractor()
    fe.eval()

    processed = 0
    skipped = 0
    errors: list[dict[str, str]] = []
    t0 = time.monotonic()

    for idx, pt_path in enumerate(raw_files):
        try:
            sample = _load_sample(pt_path)

            if skip_processed and _is_already_processed(sample.sample_id, output_dir):
                skipped += 1
                pct = min(99.0, (idx + 1) / total * 100)
                _emit_progress(pct, f"skipped {sample.sample_id} (already processed)")
                continue

            tokens, norm_stats = sample_to_features(
                sample, fe, use_legacy_pmi=use_legacy_pmi,
            )

            source_subdir = output_dir / (sample.source or "unknown")
            source_subdir.mkdir(parents=True, exist_ok=True)

            out_path = source_subdir / f"bridged_{sample.sample_id}.pt"
            payload = {
                "tokens": tokens,
                "norm_stats": norm_stats,
                "sample_id": sample.sample_id,
                "source": sample.source,
                "link": sample.link,
                "snr_dB": float(sample.snr_dB),
                "sinr_dB": float(sample.sinr_dB),
                "sir_dB": float(sample.sir_dB) if sample.sir_dB is not None else None,
                "ue_position": sample.ue_position.tolist() if sample.ue_position is not None else None,
                "serving_cell_id": int(sample.serving_cell_id),
                "channel_est_mode": sample.channel_est_mode,
                "link_pairing": sample.link_pairing,
                "bridge_job_id": job_id,
                "bridged_at": datetime.now(timezone.utc).isoformat(),
                "source_path": str(pt_path),
            }
            torch.save(payload, out_path)

            processed += 1
            pct = min(99.0, (idx + 1) / total * 100)
            _emit_progress(pct, f"processed {processed}/{total}")

        except Exception as exc:
            errors.append({"file": str(pt_path), "error": str(exc)})
            _log.error("bridge_sample_error", file=str(pt_path), error=str(exc))

    elapsed = time.monotonic() - t0

    _update_manifest(output_dir, sources, job_id)

    summary = {
        "status": "completed" if not errors else "completed_with_errors",
        "processed": processed,
        "skipped": skipped,
        "total_found": total,
        "errors": len(errors),
        "elapsed_secs": round(elapsed, 2),
        "samples_per_sec": round(processed / max(elapsed, 1e-6), 2),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if errors:
        summary["error_details"] = errors[:20]

    (output_dir / "bridge_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )

    _emit_progress(100.0, "done")
    _log.info(
        "bridge_done",
        processed=processed,
        skipped=skipped,
        errors=len(errors),
        elapsed=f"{elapsed:.1f}s",
    )


def _update_manifest(output_dir: Path, sources: list[str], job_id: str | None) -> None:
    """Update manifest entries from 'raw' → 'bridged' for processed samples."""
    try:
        from msg_embedding.data.manifest import Manifest

        manifest_path = _PROJECT_ROOT / "bridge_out" / "manifest.parquet"
        if not manifest_path.exists():
            _log.info("bridge_no_manifest", path=str(manifest_path))
            return

        manifest = Manifest(manifest_path)
        manifest.load()
        df = manifest._df
        if df is None or df.empty:
            return

        bridged_ids = set()
        for pt_file in output_dir.rglob("bridged_*.pt"):
            sid = pt_file.stem.replace("bridged_", "")
            bridged_ids.add(sid)

        if not bridged_ids:
            return

        mask = df["uuid"].isin(bridged_ids) | df["sample_id"].astype(str).isin(bridged_ids)
        updated = int(mask.sum())
        if updated > 0:
            df.loc[mask, "stage"] = "bridged"
            manifest._df = df
            manifest.save()
            _log.info("manifest_updated", count=updated)

    except Exception as exc:
        _log.warning("manifest_update_failed", error=str(exc))


if __name__ == "__main__":
    main()
