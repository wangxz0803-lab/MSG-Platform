"""Datasets endpoints -- aggregates, paginated sample queries, split management, export."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models.sample import Sample
from ..schemas.job import JobSchema
from ..schemas.sample import (
    DatasetCollectRequest,
    DatasetExportRequest,
    DatasetListResponse,
    DatasetSampleListResponse,
    DatasetSummary,
    SampleSchema,
    SplitComputeRequest,
    SplitInfoResponse,
)
from ..services.job_dispatch import deserialize_overrides, dispatch_job
from ..services.manifest_sync import sync_manifest_to_db
from ..settings import get_settings

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.post("/sync", status_code=status.HTTP_200_OK)
def sync_manifest(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Re-sync manifest.parquet → DB so newly collected samples appear."""
    n = sync_manifest_to_db(db)
    return {"synced": n}


def _sample_to_schema(row: Sample) -> SampleSchema:
    """Map an ORM Sample into its response schema."""
    link: Any = row.link if row.link in ("UL", "DL") else None
    pairing = getattr(row, "link_pairing", None) or "single"
    return SampleSchema(
        sample_id=row.sample_id,
        uuid=row.uuid,
        source=row.source,
        link=link,
        snr_db=row.snr_db,
        sir_db=row.sir_db,
        sinr_db=row.sinr_db,
        ul_sir_db=getattr(row, "ul_sir_db", None),
        dl_sir_db=getattr(row, "dl_sir_db", None),
        num_interfering_ues=getattr(row, "num_interfering_ues", None),
        link_pairing=pairing if pairing in ("single", "paired") else "single",
        num_cells=row.num_cells,
        ts=row.ts,
        status=row.status,
        shard_id=row.shard_id,
        run_id=row.run_id,
        stage=getattr(row, "stage", None) or "raw",
        serving_cell_id=getattr(row, "serving_cell_id", None),
        channel_est_mode=getattr(row, "channel_est_mode", None),
        bridged_path=getattr(row, "bridged_path", None),
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
    sync_manifest_to_db(db)
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
        ul_sirs = [x.ul_sir_db for x in items if getattr(x, "ul_sir_db", None) is not None]
        dl_sirs = [x.dl_sir_db for x in items if getattr(x, "dl_sir_db", None) is not None]
        links = sorted({x.link for x in items if x.link})
        has_paired = any(getattr(x, "link_pairing", None) == "paired" for x in items)
        sc: dict[str, int] = {}
        for x in items:
            s = getattr(x, "stage", None) or "raw"
            sc[s] = sc.get(s, 0) + 1
        summaries.append(
            DatasetSummary(
                source=src,
                count=len(items),
                snr_mean=statistics.fmean(snrs) if snrs else None,
                snr_std=statistics.pstdev(snrs) if len(snrs) > 1 else 0.0 if snrs else None,
                sir_mean=statistics.fmean(sirs) if sirs else None,
                sinr_mean=statistics.fmean(sinrs) if sinrs else None,
                ul_sir_mean=statistics.fmean(ul_sirs) if ul_sirs else None,
                dl_sir_mean=statistics.fmean(dl_sirs) if dl_sirs else None,
                links=links,
                has_paired=has_paired,
                stage_counts=sc,
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


# ---------------------------------------------------------------------------
# Split management
# ---------------------------------------------------------------------------

def _get_manifest():
    """Load the project manifest."""
    from msg_embedding.data.manifest import Manifest

    settings = get_settings()
    return Manifest(settings.manifest_file)


@router.get("/split/status", response_model=SplitInfoResponse)
def get_split_status() -> SplitInfoResponse:
    """Return the current split state (locked, version, counts)."""
    m = _get_manifest()
    info = m.get_split_info()
    return SplitInfoResponse(**info)


@router.post("/split", response_model=SplitInfoResponse)
def compute_and_lock_split(
    payload: SplitComputeRequest,
    db: Session = Depends(get_db),
) -> SplitInfoResponse:
    """Compute train/val/test split and optionally lock it.

    Once locked, the test and val sets are immutable. New data added later
    is automatically assigned to the train split.
    """
    m = _get_manifest()

    if m.is_split_locked:
        raise HTTPException(
            status_code=409,
            detail="Split is already locked. Unlock first to re-compute.",
        )

    ratios = tuple(payload.ratios)
    m.compute_split(strategy=payload.strategy, seed=payload.seed, ratios=ratios)
    m.save()

    if payload.lock:
        m.lock_split()

    sync_manifest_to_db(db)
    return SplitInfoResponse(**m.get_split_info())


@router.post("/split/unlock")
def unlock_split() -> dict[str, Any]:
    """Unlock the current split (admin operation). Existing labels are kept."""
    m = _get_manifest()
    if not m.is_split_locked:
        raise HTTPException(status_code=409, detail="Split is not locked.")
    m.unlock_split()
    return {"unlocked": True, **m.get_split_info()}


# ---------------------------------------------------------------------------
# Data export
# ---------------------------------------------------------------------------

@router.post(
    "/export",
    status_code=status.HTTP_202_ACCEPTED,
)
def export_dataset_endpoint(
    payload: DatasetExportRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Kick off a dataset export job."""
    overrides = {
        "export.format": payload.format,
        "export.split": payload.split,
        "export.source_filter": payload.source_filter,
        "export.link_filter": payload.link_filter,
        "export.min_snr": payload.min_snr,
        "export.max_snr": payload.max_snr,
        "export.export_name": payload.export_name,
        "export.shard_size": payload.shard_size,
        "export.include_interferers": payload.include_interferers,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    try:
        job = dispatch_job(
            db,
            job_type="dataset_export",
            config_overrides=overrides,
            display_name=f"export:{payload.format}:{payload.split or 'all'}",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"job_id": job.job_id}


@router.get("/exports")
def list_exports() -> dict[str, Any]:
    """List available export packages with download URLs."""
    settings = get_settings()
    exports_dir = settings.exports_path
    if not exports_dir.exists():
        return {"exports": []}

    result = []
    for p in sorted(exports_dir.iterdir()):
        if p.is_file() and p.suffix == ".h5":
            try:
                import h5py
                with h5py.File(p, "r") as f:
                    info = {
                        "name": p.name,
                        "format": "hdf5",
                        "num_samples": int(f.attrs.get("num_samples", 0)),
                        "split": str(f.attrs.get("split", "")),
                        "split_version": int(f.attrs.get("split_version", 0)),
                        "total_bytes": p.stat().st_size,
                        "path": str(p),
                        "download_url": f"/api/datasets/exports/{p.name}/download",
                    }
                    result.append(info)
            except Exception:
                pass
        elif p.is_dir():
            readme_path = p / "README.json"
            if readme_path.exists():
                try:
                    import json
                    meta = json.loads(readme_path.read_text(encoding="utf-8"))
                    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                    info = {
                        "name": p.name,
                        "format": meta.get("export_format", "unknown"),
                        "num_samples": meta.get("num_samples", 0),
                        "split": meta.get("export_split", ""),
                        "split_version": meta.get("split_version", 0),
                        "total_bytes": total,
                        "path": str(p),
                        "download_url": f"/api/datasets/exports/{p.name}/download",
                    }
                    result.append(info)
                except Exception:
                    pass

    return {"exports": result}


@router.get("/exports/{name}/download")
def download_export(name: str):
    """Download an export package (single file or zipped directory)."""
    settings = get_settings()
    exports_dir = settings.exports_path
    target = exports_dir / name

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"export '{name}' not found")

    safe = target.resolve()
    if not str(safe).startswith(str(exports_dir.resolve())):
        raise HTTPException(status_code=400, detail="invalid export name")

    if target.is_file():
        return FileResponse(
            path=str(target),
            filename=name,
            media_type="application/octet-stream",
        )

    import shutil
    import tempfile
    zip_dir = Path(tempfile.mkdtemp())
    zip_path = zip_dir / f"{name}.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", str(target))
    return FileResponse(
        path=str(zip_path),
        filename=f"{name}.zip",
        media_type="application/zip",
    )
