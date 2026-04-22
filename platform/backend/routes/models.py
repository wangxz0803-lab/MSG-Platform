"""Models endpoints -- list artifacts + dispatch export jobs."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models.model_registry import ModelArtifact
from ..models.run import Run
from ..schemas.model import (
    ExportRequest,
    ModelArtifactSchema,
    ModelListResponse,
)
from ..services.job_dispatch import dispatch_job

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=ModelListResponse)
def list_models(
    run_id: str | None = Query(default=None),
    format: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> ModelListResponse:
    """List registered model artifacts."""
    stmt = select(ModelArtifact)
    if run_id:
        stmt = stmt.where(ModelArtifact.run_id == run_id)
    if format:
        if format not in ("pt", "onnx", "torchscript"):
            raise HTTPException(status_code=400, detail=f"invalid format: {format}")
        stmt = stmt.where(ModelArtifact.format == format)
    total = db.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()
    rows = (
        db.execute(
            stmt.order_by(ModelArtifact.created_at.desc()).offset(offset).limit(limit)
        )
        .scalars()
        .all()
    )
    items = [ModelArtifactSchema.model_validate(r) for r in rows]
    return ModelListResponse(total=total, items=items)


@router.post("/{run_id}/export", status_code=status.HTTP_202_ACCEPTED)
def export_model(
    run_id: str,
    payload: ExportRequest,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Kick off an export job that produces ONNX / TorchScript artifacts."""
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    if payload.format not in ("onnx", "torchscript", "both"):
        raise HTTPException(status_code=400, detail=f"invalid format: {payload.format}")

    overrides = {
        "infer.run_id": run_id,
        "infer.export_format": payload.format,
    }
    job = dispatch_job(
        db,
        job_type="export",
        config_overrides=overrides,
        display_name=f"export:{run_id}:{payload.format}",
    )
    return {"job_id": job.job_id}
