"""Datasets endpoints -- aggregates + paginated sample queries."""

from __future__ import annotations

import statistics
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models.sample import Sample
from ..schemas.job import JobSchema
from ..schemas.sample import (
    DatasetCollectRequest,
    DatasetListResponse,
    DatasetSampleListResponse,
    DatasetSummary,
    SampleSchema,
)
from ..services.job_dispatch import deserialize_overrides, dispatch_job

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def _sample_to_schema(row: Sample) -> SampleSchema:
    """Map an ORM Sample into its response schema."""
    link: Any = row.link if row.link in ("UL", "DL") else None
    return SampleSchema(
        sample_id=row.sample_id,
        uuid=row.uuid,
        source=row.source,
        link=link,
        snr_db=row.snr_db,
        sir_db=row.sir_db,
        sinr_db=row.sinr_db,
        num_cells=row.num_cells,
        ts=row.ts,
        status=row.status,
        shard_id=row.shard_id,
        run_id=row.run_id,
    )


@router.get("", response_model=DatasetListResponse)
def list_datasets(
    source: str | None = Query(default=None),
    link: str | None = Query(default=None),
    min_snr: float | None = Query(default=None),
    max_snr: float | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> DatasetListResponse:
    """Return per-source dataset summaries matching optional filters."""
    stmt = select(Sample)
    if source:
        stmt = stmt.where(Sample.source == source)
    if link:
        stmt = stmt.where(Sample.link == link)
    if min_snr is not None:
        stmt = stmt.where(Sample.snr_db >= min_snr)
    if max_snr is not None:
        stmt = stmt.where(Sample.snr_db <= max_snr)

    rows = list(db.execute(stmt).scalars())

    by_source: dict[str, list[Sample]] = {}
    for r in rows:
        by_source.setdefault(r.source or "unknown", []).append(r)

    summaries: list[DatasetSummary] = []
    for src, items in sorted(by_source.items()):
        snrs = [x.snr_db for x in items if x.snr_db is not None]
        sirs = [x.sir_db for x in items if x.sir_db is not None]
        sinrs = [x.sinr_db for x in items if x.sinr_db is not None]
        links = sorted({x.link for x in items if x.link})
        summaries.append(
            DatasetSummary(
                source=src,
                count=len(items),
                snr_mean=statistics.fmean(snrs) if snrs else None,
                snr_std=statistics.pstdev(snrs) if len(snrs) > 1 else 0.0 if snrs else None,
                sir_mean=statistics.fmean(sirs) if sirs else None,
                sinr_mean=statistics.fmean(sinrs) if sinrs else None,
                links=links,
            )
        )

    total = len(summaries)
    page = summaries[offset : offset + limit]
    return DatasetListResponse(total=total, items=page)


@router.delete("/{source}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset_source(
    source: str,
    db: Session = Depends(get_db),
) -> Response:
    """Delete all samples for a given source."""
    deleted = db.query(Sample).filter(Sample.source == source).delete()
    db.commit()
    if deleted == 0:
        raise HTTPException(
            status_code=404, detail=f"No samples found for source '{source}'"
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{source}/samples", response_model=DatasetSampleListResponse)
def list_samples_for_source(
    source: str,
    link: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> DatasetSampleListResponse:
    """Paginated sample listing for a single source."""
    stmt = select(Sample).where(Sample.source == source)
    if link:
        stmt = stmt.where(Sample.link == link)
    total = db.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()
    rows = db.execute(stmt.offset(offset).limit(limit)).scalars().all()
    items = [_sample_to_schema(r) for r in rows]
    return DatasetSampleListResponse(total=total, items=items)


@router.post(
    "/collect",
    response_model=JobSchema,
    status_code=status.HTTP_202_ACCEPTED,
)
def collect_dataset(
    payload: DatasetCollectRequest,
    db: Session = Depends(get_db),
) -> JobSchema:
    """Kick off a simulation job that produces new samples."""
    overrides = dict(payload.config_overrides)
    overrides.setdefault("data.source", payload.source)
    if payload.output_dir:
        overrides.setdefault("data.output_dir", payload.output_dir)

    try:
        job = dispatch_job(
            db,
            job_type="simulate",
            config_overrides=overrides,
            display_name=f"collect:{payload.source}",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JobSchema(
        job_id=job.job_id,
        type=job.type,
        status=job.status,  # type: ignore[arg-type]
        display_name=job.display_name,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        progress_pct=job.progress_pct,
        config_overrides=deserialize_overrides(job.params_json),
        log_path=job.log_path,
        error_msg=job.error_msg,
    )
