"""Runs endpoints -- list, detail, metrics, compare."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models.run import Run
from ..schemas.run import (
    RunCompareItem,
    RunCompareResponse,
    RunListResponse,
    RunSchema,
)

router = APIRouter(prefix="/api/runs", tags=["runs"])


def _safe_load_json(path: str | None) -> dict[str, Any] | None:
    """Best-effort JSON loader returning None on failure."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _safe_load_yaml(path: str | None) -> dict[str, Any] | None:
    """Load a Hydra config.yaml if PyYAML is available; else None."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _split_tags(raw: str | None) -> list[str]:
    """Split comma-separated tag string into list."""
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _to_schema(run: Run, *, include_payload: bool = True) -> RunSchema:
    """Materialise a RunSchema from ORM row (optionally loading heavy JSON/YAML)."""
    config = _safe_load_yaml(run.config_path) if include_payload else None
    metadata = _safe_load_json(run.metadata_path) if include_payload else None
    metrics = _safe_load_json(run.metrics_json_path) if include_payload else None
    return RunSchema.model_validate(
        {
            "run_id": run.run_id,
            "created_at": run.created_at,
            "ckpt_path": run.ckpt_path,
            "ckpt_best": run.ckpt_best,
            "ckpt_last": run.ckpt_last,
            "config": config,
            "metadata": metadata,
            "metrics": metrics,
            "git_sha": run.git_sha,
            "tags": _split_tags(run.tags),
        }
    )


@router.get("", response_model=RunListResponse)
def list_runs(
    tag: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> RunListResponse:
    """List runs (optionally filtered by tag substring)."""
    stmt = select(Run)
    if tag:
        stmt = stmt.where(Run.tags.contains(tag))
    total = db.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()
    rows = (
        db.execute(stmt.order_by(Run.created_at.desc()).offset(offset).limit(limit))
        .scalars()
        .all()
    )
    return RunListResponse(
        total=total,
        items=[_to_schema(r, include_payload=False) for r in rows],
    )


@router.get("/compare", response_model=RunCompareResponse)
def compare_runs(
    ids: str = Query(..., description="Comma-separated run_ids"),
    db: Session = Depends(get_db),
) -> RunCompareResponse:
    """Fetch metrics.json for each id and return them side-by-side."""
    run_ids = [x.strip() for x in ids.split(",") if x.strip()]
    if not run_ids:
        raise HTTPException(status_code=400, detail="ids query cannot be empty")

    items: list[RunCompareItem] = []
    for rid in run_ids:
        run = db.get(Run, rid)
        if run is None:
            items.append(RunCompareItem(run_id=rid, metrics={}))
            continue
        metrics = _safe_load_json(run.metrics_json_path) or {}
        items.append(RunCompareItem(run_id=rid, metrics=metrics))
    return RunCompareResponse(runs=items)


@router.delete("/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_run(run_id: str, db: Session = Depends(get_db)):
    """Delete a training run record."""
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    db.delete(run)
    db.commit()


@router.get("/{run_id}", response_model=RunSchema)
def get_run(run_id: str, db: Session = Depends(get_db)) -> RunSchema:
    """Full run detail including parsed config/metadata/metrics."""
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return _to_schema(run)


@router.get("/{run_id}/metrics")
def get_run_metrics(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Raw metrics.json contents (flat dict as written by eval runner)."""
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    metrics = _safe_load_json(run.metrics_json_path)
    if metrics is None:
        raise HTTPException(status_code=404, detail="metrics.json not available")
    return metrics
