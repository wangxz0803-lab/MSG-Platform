"""Scan ``artifacts/*/`` and ``reports/*/`` to populate Run + ModelArtifact tables."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from msg_embedding.core.logging import get_logger

from ..models.model_registry import ModelArtifact
from ..models.run import Run
from ..settings import get_settings

_log = get_logger(__name__)

CKPT_BEST_NAMES = ("ckpt_best.pth", "best.pth", "model_best.pth")
CKPT_LAST_NAMES = ("ckpt_last.pth", "last.pth", "model_last.pth")
ONNX_NAMES = ("model.onnx",)
TS_NAMES = ("model.ts", "model.pt.ts")


def _find_first(run_dir: Path, names: tuple[str, ...]) -> Path | None:
    """Return the first file matching one of ``names`` in ``run_dir``."""
    for name in names:
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return None


def _load_json(path: Path | None) -> dict | None:
    """Safely load JSON or return None."""
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _log.warning("json_parse_failed", path=str(path), error=str(exc))
        return None


def _register_artifact(
    session: Session,
    run_id: str,
    fmt: str,
    path: Path,
) -> None:
    """Upsert a ModelArtifact row keyed on (run_id, format, path)."""
    existing = (
        session.query(ModelArtifact)
        .filter(
            ModelArtifact.run_id == run_id,
            ModelArtifact.format == fmt,
            ModelArtifact.path == str(path),
        )
        .first()
    )
    size = path.stat().st_size if path.exists() else 0
    if existing is None:
        session.add(
            ModelArtifact(
                run_id=run_id,
                format=fmt,
                path=str(path),
                size_bytes=size,
            )
        )
    else:
        existing.size_bytes = size


def scan_artifacts_and_reports(
    session: Session,
    artifacts_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
) -> tuple[int, int]:
    """Walk artifacts & reports directories, upserting Run + ModelArtifact rows.

    Returns ``(n_runs, n_artifacts)``.
    """
    settings = get_settings()
    art_root = Path(artifacts_dir) if artifacts_dir else settings.artifacts_path
    rep_root = Path(reports_dir) if reports_dir else settings.reports_path

    n_runs = 0
    n_art = 0

    if not art_root.exists():
        _log.info("artifacts_dir_missing", path=str(art_root))
        return (0, 0)

    for run_dir in sorted(p for p in art_root.iterdir() if p.is_dir()):
        run_id = run_dir.name
        meta_path = run_dir / "metadata.json"
        cfg_path = run_dir / "config.yaml"
        ckpt_best = _find_first(run_dir, CKPT_BEST_NAMES)
        ckpt_last = _find_first(run_dir, CKPT_LAST_NAMES)
        onnx_path = _find_first(run_dir, ONNX_NAMES)
        ts_path = _find_first(run_dir, TS_NAMES)

        metrics_path: Path | None = rep_root / run_id / "metrics.json"
        if not metrics_path.exists():
            alt = run_dir / "metrics.json"
            metrics_path = alt if alt.exists() else None

        metadata = _load_json(meta_path) or {}
        git_sha = metadata.get("git_sha") if isinstance(metadata, dict) else None
        tags = metadata.get("tags") if isinstance(metadata, dict) else None
        tag_str = ",".join(tags) if isinstance(tags, list) else None

        created_at = datetime.now(UTC)
        try:
            created_at = datetime.fromtimestamp(run_dir.stat().st_mtime)
        except OSError:
            pass

        existing = session.get(Run, run_id)
        if existing is None:
            session.add(
                Run(
                    run_id=run_id,
                    created_at=created_at,
                    ckpt_path=str(ckpt_best or ckpt_last or ""),
                    ckpt_best=str(ckpt_best) if ckpt_best else None,
                    ckpt_last=str(ckpt_last) if ckpt_last else None,
                    metrics_json_path=str(metrics_path) if metrics_path else None,
                    config_path=str(cfg_path) if cfg_path.exists() else None,
                    metadata_path=str(meta_path) if meta_path.exists() else None,
                    git_sha=git_sha,
                    tags=tag_str,
                )
            )
        else:
            existing.ckpt_best = str(ckpt_best) if ckpt_best else None
            existing.ckpt_last = str(ckpt_last) if ckpt_last else None
            existing.metrics_json_path = str(metrics_path) if metrics_path else None
            existing.config_path = str(cfg_path) if cfg_path.exists() else None
            existing.metadata_path = str(meta_path) if meta_path.exists() else None
            existing.git_sha = git_sha
            existing.tags = tag_str
        n_runs += 1

        job_id_from_meta = metadata.get("job_id") if isinstance(metadata, dict) else None
        if job_id_from_meta:
            try:
                from ..models.job import Job

                job_row = session.get(Job, job_id_from_meta)
                if (
                    job_row is not None
                    and not getattr(job_row, "run_id", None)
                    and hasattr(job_row, "run_id")
                ):
                    job_row.run_id = run_id
            except Exception:
                pass

        for fmt, path in [
            ("pt", ckpt_best),
            ("pt", ckpt_last),
            ("onnx", onnx_path),
            ("torchscript", ts_path),
        ]:
            if path is not None:
                _register_artifact(session, run_id, fmt, path)
                n_art += 1

    session.commit()
    _log.info("artifacts_scanned", runs=n_runs, artifacts=n_art)
    return (n_runs, n_art)
