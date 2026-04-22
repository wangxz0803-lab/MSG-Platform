"""Jobs endpoints -- CRUD + progress + logs + cancel."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models.job import Job
from ..schemas.job import (
    BatchJobCreateRequest,
    BatchJobCreateResponse,
    JobCancelResponse,
    JobCreateRequest,
    JobListResponse,
    JobLogsResponse,
    JobProgressResponse,
    JobSchema,
)
from ..services.job_dispatch import (
    VALID_JOB_TYPES,
    cancel_job,
    deserialize_overrides,
    dispatch_batch_jobs,
    dispatch_job,
    read_log_tail,
    read_progress,
)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _to_schema(job: Job, current_step: str | None = None) -> JobSchema:
    """Map an ORM Job into its response schema."""
    return JobSchema(
        job_id=job.job_id,
        type=job.type,
        status=job.status,  # type: ignore[arg-type]
        display_name=job.display_name,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        progress_pct=job.progress_pct or 0.0,
        current_step=current_step,
        config_overrides=deserialize_overrides(job.params_json),
        error_msg=job.error_msg,
        log_path=job.log_path,
        run_id=getattr(job, "run_id", None),
    )


@router.get("", response_model=JobListResponse)
def list_jobs(
    status_filter: str | None = Query(default=None, alias="status"),
    type_filter: str | None = Query(default=None, alias="type"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> JobListResponse:
    """List jobs matching optional filters."""
    stmt = select(Job)
    if status_filter:
        stmt = stmt.where(Job.status == status_filter)
    if type_filter:
        stmt = stmt.where(Job.type == type_filter)
    total = db.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()
    rows = (
        db.execute(stmt.order_by(Job.created_at.desc()).offset(offset).limit(limit))
        .scalars()
        .all()
    )
    return JobListResponse(total=total, items=[_to_schema(r) for r in rows])


@router.post("", response_model=JobSchema, status_code=status.HTTP_201_CREATED)
def create_job(payload: JobCreateRequest, db: Session = Depends(get_db)) -> JobSchema:
    """Create a new compute job and hand it off to the worker."""
    if payload.type not in VALID_JOB_TYPES:
        raise HTTPException(status_code=400, detail=f"unknown type: {payload.type}")
    job = dispatch_job(
        db,
        job_type=payload.type,
        config_overrides=payload.config_overrides,
        display_name=payload.display_name,
    )
    return _to_schema(job)


@router.post(
    "/batch", response_model=BatchJobCreateResponse, status_code=status.HTTP_201_CREATED
)
def create_batch_jobs(
    payload: BatchJobCreateRequest, db: Session = Depends(get_db)
) -> BatchJobCreateResponse:
    """Create multiple jobs with different config overrides."""
    if payload.type not in VALID_JOB_TYPES:
        raise HTTPException(status_code=400, detail=f"unknown type: {payload.type}")
    jobs = dispatch_batch_jobs(
        db,
        job_type=payload.type,
        configs=payload.configs,
        display_name_prefix=payload.display_name_prefix,
    )
    return BatchJobCreateResponse(jobs=[_to_schema(j) for j in jobs])


@router.get("/{job_id}", response_model=JobSchema)
def get_job(job_id: str, db: Session = Depends(get_db)) -> JobSchema:
    """Fetch a single job by id."""
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _to_schema(job)


@router.get("/{job_id}/progress", response_model=JobProgressResponse)
def get_job_progress(job_id: str, db: Session = Depends(get_db)) -> JobProgressResponse:
    """Return the worker-written progress blob (falling back to DB fields)."""
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    prog = read_progress(job_id)
    return JobProgressResponse(
        job_id=job_id,
        progress_pct=float(prog.get("progress_pct", job.progress_pct or 0.0)),
        status=str(prog.get("status", job.status)),
        current_step=prog.get("current_step"),
        eta_seconds=prog.get("eta_seconds"),
    )


@router.get("/{job_id}/logs", response_model=JobLogsResponse)
def get_job_logs(
    job_id: str,
    tail: int = Query(default=500, ge=1, le=100000),
    db: Session = Depends(get_db),
) -> JobLogsResponse:
    """Tail the worker log file for ``job_id``."""
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    lines = read_log_tail(job_id, tail=tail)
    return JobLogsResponse(job_id=job_id, lines=lines)


@router.post("/{job_id}/cancel", response_model=JobCancelResponse)
def cancel_job_route(job_id: str, db: Session = Depends(get_db)) -> JobCancelResponse:
    """Cancel a running/queued job."""
    job = cancel_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobCancelResponse(job_id=job.job_id, status=job.status)  # type: ignore[arg-type]


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job(job_id: str, db: Session = Depends(get_db)):
    """Delete a job record. Only completed/failed/cancelled jobs can be deleted."""
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status == "running":
        raise HTTPException(status_code=409, detail="Cannot delete a running job")
    if job.status == "queued":
        raise HTTPException(
            status_code=409, detail="Cannot delete a queued job -- cancel it first"
        )
    db.delete(job)
    db.commit()
