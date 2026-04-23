"""CLI entry point for channel simulation data generation.

Parses key=value overrides from CLI (as sent by the platform worker),
loads YAML defaults, instantiates the data source, and writes samples.

Usage::

    python scripts/run_simulate.py source=sionna_rt num_samples=50 num_sites=7
    python scripts/run_simulate.py data.source=sionna_rt data.num_samples=100

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

from msg_embedding.core.logging import get_logger
from msg_embedding.data.sources import SOURCE_REGISTRY

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

_log = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_yaml_defaults() -> dict[str, Any]:
    """Load base defaults from configs/data/sionna_rt.yaml."""
    yaml_path = _PROJECT_ROOT / "configs" / "data" / "sionna_rt.yaml"
    if yaml_path.exists() and yaml is not None:
        with open(yaml_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _parse_cli_overrides() -> dict[str, Any]:
    """Parse key=value args from sys.argv[1:] into a flat dict.

    Strips optional ``data.`` or ``simulate.`` prefixes so both
    ``data.num_samples=50`` and ``num_samples=50`` work.
    """
    overrides: dict[str, Any] = {}
    for arg in sys.argv[1:]:
        if "=" not in arg:
            continue
        key, val = arg.split("=", 1)
        for prefix in ("data.", "simulate."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        if val.lower() in ("true", "false"):
            overrides[key] = val.lower() == "true"
        elif val in ("null", "none"):
            overrides[key] = None
        else:
            try:
                if "." in val or "e" in val.lower():
                    overrides[key] = float(val)
                else:
                    overrides[key] = int(val)
            except ValueError:
                if val.startswith("[") and val.endswith("]"):
                    try:
                        overrides[key] = json.loads(val)
                    except json.JSONDecodeError:
                        overrides[key] = val
                else:
                    overrides[key] = val
    return overrides


def _emit_progress(pct: float, step: str) -> None:
    print(f"[progress] pct={pct:.1f} step={step}", flush=True)


def _write_progress(output_dir: Path, pct: float, step: str, **extra: Any) -> None:
    payload = {"pct": pct, "step": step, "updated_at": datetime.now(timezone.utc).isoformat(), **extra}
    tmp = output_dir / "progress.json.tmp"
    final = output_dir / "progress.json"
    try:
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(final)
    except OSError:
        pass


def _save_sample(sample: Any, output_dir: Path, idx: int) -> Path:
    path = output_dir / f"sample_{idx:06d}.pt"
    payload = sample.to_dict()
    try:
        import torch

        torch.save(payload, path)
    except ImportError:
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _register_manifest(
    samples: list[Any],
    output_dir: Path,
    source_name: str,
    job_id: str | None,
) -> int:
    """Append saved samples to bridge_out/manifest.parquet so they show in datasets."""
    try:
        from msg_embedding.data.manifest import Manifest

        manifest_path = _PROJECT_ROOT / "bridge_out" / "manifest.parquet"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = Manifest(manifest_path)

        rows = []
        for idx, sample in enumerate(samples):
            pt_path = output_dir / f"sample_{idx:06d}.pt"
            row = {
                "uuid": sample.sample_id,
                "job_id": job_id,
                "run_id": None,
                "source": source_name,
                "shard_id": 0,
                "sample_id": idx,
                "stage": "raw",
                "status": "ok",
                "link": sample.link,
                "snr_dB": float(sample.snr_dB),
                "sir_dB": float(sample.sir_dB) if sample.sir_dB is not None else None,
                "sinr_dB": float(sample.sinr_dB),
                "num_cells": sample.meta.get("num_cells"),
                "serving_cell_id": int(sample.serving_cell_id),
                "ue_x": float(sample.ue_position[0]) if sample.ue_position is not None else None,
                "ue_y": float(sample.ue_position[1]) if sample.ue_position is not None else None,
                "ue_z": float(sample.ue_position[2]) if sample.ue_position is not None else None,
                "channel_est_mode": sample.channel_est_mode,
                "split": "unassigned",
                "hash": None,
                "path": str(pt_path),
                "error_msg": None,
            }
            rows.append(row)

        manifest.append(rows)
        manifest.save()
        _log.info("manifest_registered", count=len(rows), path=str(manifest_path))
        return len(rows)
    except Exception as exc:
        _log.warning("manifest_registration_failed", error=str(exc))
        return 0


def main() -> None:
    defaults = _load_yaml_defaults()
    overrides = _parse_cli_overrides()

    cfg: dict[str, Any] = {**defaults, **overrides}

    source_name = str(cfg.pop("source", "sionna_rt"))
    job_id = cfg.pop("project.job_id", None) or cfg.pop("job_id", None)
    output_dir = Path(str(cfg.pop("output_dir", "artifacts/simulate_out")))
    output_dir.mkdir(parents=True, exist_ok=True)

    num_samples = int(cfg.get("num_samples", 100))

    _log.info("simulate_start", source=source_name, samples=num_samples, output=str(output_dir))

    source_cls = SOURCE_REGISTRY.get(source_name)
    if source_cls is None:
        raise RuntimeError(
            f"Source '{source_name}' not in registry. "
            f"Available: {list(SOURCE_REGISTRY.keys())}. "
            f"If you need mock data for testing, pass source=sionna_rt_mock explicitly."
        )

    source = source_cls(cfg)

    meta = {
        "source": source_name,
        "fallback": False,
        "num_samples": num_samples,
        "output_dir": str(output_dir),
        "config": cfg,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    _emit_progress(0.0, "starting")
    _write_progress(output_dir, 0.0, "starting")

    saved = 0
    collected_samples: list[Any] = []
    errors: list[dict[str, str]] = []
    t0 = time.monotonic()

    try:
        for idx, sample in enumerate(source.iter_samples()):
            try:
                _save_sample(sample, output_dir, idx)
                collected_samples.append(sample)
                saved += 1
                pct = min(99.0, saved / max(num_samples, 1) * 100)
                _emit_progress(pct, f"sample {saved}/{num_samples}")
                _write_progress(
                    output_dir,
                    pct,
                    f"sample {saved}/{num_samples}",
                    saved=saved,
                    total=num_samples,
                    elapsed=round(time.monotonic() - t0, 1),
                )
            except Exception as exc:
                errors.append({"index": str(idx), "error": str(exc)})
                _log.error("sample_error", index=idx, error=str(exc))
    except Exception as exc:
        _log.error("simulate_fatal", error=str(exc), traceback=traceback.format_exc())
        _emit_progress(0.0, f"failed: {exc}")
        _write_progress(output_dir, 0.0, f"failed: {exc}")
        raise

    elapsed = time.monotonic() - t0
    summary = {
        "status": "completed" if not errors else "completed_with_errors",
        "saved": saved,
        "requested": num_samples,
        "errors": len(errors),
        "elapsed_secs": round(elapsed, 2),
        "samples_per_sec": round(saved / max(elapsed, 1e-6), 2),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _emit_progress(100.0, "done")
    _write_progress(output_dir, 100.0, "done", **summary)

    _register_manifest(collected_samples, output_dir, source_name, job_id)

    _log.info("simulate_done", saved=saved, requested=num_samples, errors=len(errors), elapsed=f"{elapsed:.1f}s")


if __name__ == "__main__":
    main()
