"""Models endpoints -- list artifacts, dispatch export, upload, evaluate."""

from __future__ import annotations

import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models.model_registry import ModelArtifact
from ..models.run import Run
from ..schemas.model import (
    ExportRequest,
    LeaderboardEntry,
    LeaderboardResponse,
    ModelArtifactSchema,
    ModelEvalRequest,
    ModelInferRequest,
    ModelListResponse,
    ModelUploadResponse,
)
from ..services.job_dispatch import dispatch_job
from ..settings import get_settings

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


def _validate_checkpoint_compatibility(ckpt_path: Path) -> tuple[bool, str]:
    """Check if a checkpoint is compatible with ChannelMAE."""
    try:
        import torch
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            sd = state["model"]
        elif isinstance(state, dict) and any(k.startswith("encoder") for k in state):
            sd = state
        else:
            sd = state

        if not isinstance(sd, dict):
            return False, "checkpoint does not contain a state_dict"

        has_encoder = any("encoder" in k for k in sd)
        has_decoder = any("decoder" in k for k in sd)
        has_proj = any("latent_proj" in k or "proj_shortcut" in k for k in sd)

        if has_encoder and has_proj:
            return True, f"valid ChannelMAE checkpoint ({len(sd)} parameters)"
        elif has_encoder:
            return True, f"encoder-only checkpoint ({len(sd)} parameters)"
        else:
            return False, f"no encoder keys found in checkpoint ({len(sd)} keys)"
    except Exception as exc:
        return False, f"failed to load checkpoint: {exc}"


@router.post("/upload", response_model=ModelUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_model(
    file: UploadFile = File(..., description="Model checkpoint file (.pth or .pt)"),
    run_id: str = Form(default=None, description="Run ID to associate with. Auto-generated if empty."),
    tags: str = Form(default="", description="Comma-separated tags for the run."),
    description: str = Form(default="", description="Human-readable description."),
    db: Session = Depends(get_db),
) -> ModelUploadResponse:
    """Upload an externally trained model checkpoint.

    The checkpoint is validated for ChannelMAE compatibility, registered in
    the model artifacts table, and a corresponding Run record is created.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename required")

    ext = Path(file.filename).suffix.lower()
    if ext not in (".pt", ".pth", ".ckpt"):
        raise HTTPException(status_code=400, detail=f"unsupported file type: {ext}")

    if not run_id:
        ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        run_id = f"upload-{ts}-{uuid.uuid4().hex[:6]}"

    settings = get_settings()
    run_dir = Path(settings.artifacts_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dest = run_dir / "ckpt_best.pth"

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    compatible, detail = _validate_checkpoint_compatibility(dest)

    run = db.get(Run, run_id)
    if run is None:
        run = Run(
            run_id=run_id,
            created_at=datetime.now(UTC),
            ckpt_best=str(dest),
            ckpt_path=str(dest),
            tags=tags or "uploaded",
        )
        db.add(run)

    size_bytes = dest.stat().st_size
    artifact = ModelArtifact(
        run_id=run_id,
        format="pt",
        path=str(dest),
        size_bytes=size_bytes,
    )
    db.add(artifact)
    db.commit()
    db.refresh(artifact)

    return ModelUploadResponse(
        run_id=run_id,
        artifact_id=artifact.id,
        path=str(dest),
        format="pt",
        size_bytes=size_bytes,
        compatible=compatible,
        compatibility_detail=detail,
    )


@router.post("/{run_id}/evaluate", status_code=status.HTTP_202_ACCEPTED)
def evaluate_model(
    run_id: str,
    payload: ModelEvalRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Trigger evaluation of an uploaded model on the locked test set."""
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")

    ckpt = run.ckpt_best or run.ckpt_path
    if not ckpt:
        raise HTTPException(status_code=400, detail="no checkpoint found for this run")

    overrides: dict[str, Any] = {
        "eval.run_id": run_id,
        "eval.ckpt_path": ckpt,
        "eval.split": payload.test_split,
        "eval.device": payload.device,
    }
    if payload.limit:
        overrides["eval.limit"] = payload.limit

    job = dispatch_job(
        db,
        job_type="eval",
        config_overrides=overrides,
        display_name=f"eval:{run_id}:on_{payload.test_split}",
    )
    return {"job_id": job.job_id, "run_id": run_id}


@router.post("/{run_id}/infer", status_code=status.HTTP_202_ACCEPTED)
def infer_model(
    run_id: str,
    payload: ModelInferRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Run inference with an imported model to generate embeddings."""
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")

    ckpt = run.ckpt_best or run.ckpt_path
    if not ckpt:
        raise HTTPException(status_code=400, detail="no checkpoint found for this run")

    overrides: dict[str, Any] = {
        "infer.run_id": run_id,
        "infer.ckpt_path": ckpt,
        "infer.split": payload.split,
        "infer.device": payload.device,
        "infer.batch_size": payload.batch_size,
    }
    if payload.input_path:
        overrides["infer.input_path"] = payload.input_path
    if payload.limit:
        overrides["infer.limit"] = payload.limit
    if payload.output_name:
        overrides["infer.output_name"] = payload.output_name

    job = dispatch_job(
        db,
        job_type="infer",
        config_overrides=overrides,
        display_name=f"infer:{run_id}:{payload.split}",
    )
    return {"job_id": job.job_id, "run_id": run_id}


@router.get("/{run_id}/meta")
def get_model_meta(
    run_id: str,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Return training metadata extracted from the uploaded checkpoint."""
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")

    ckpt = run.ckpt_best or run.ckpt_path
    meta: dict[str, Any] = {
        "run_id": run_id,
        "tags": run.tags,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "ckpt_path": ckpt,
    }

    if ckpt and Path(ckpt).exists():
        try:
            import torch
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            if isinstance(state, dict):
                if "epoch" in state:
                    meta["epoch"] = int(state["epoch"])
                if "best_loss" in state:
                    meta["best_loss"] = float(state["best_loss"])
                elif "loss" in state:
                    meta["best_loss"] = float(state["loss"])
                if "global_step" in state:
                    meta["global_step"] = int(state["global_step"])
                if "config" in state and isinstance(state["config"], dict):
                    meta["training_config"] = state["config"]
                if "optimizer" in state:
                    meta["has_optimizer_state"] = True
                sd = state.get("model", state)
                if isinstance(sd, dict):
                    meta["num_parameters"] = len(sd)
        except Exception:
            pass

    compatible, detail = _validate_checkpoint_compatibility(Path(ckpt)) if ckpt else (False, "no checkpoint")
    meta["compatible"] = compatible
    meta["compatibility_detail"] = detail

    if run.metrics_json_path and Path(run.metrics_json_path).exists():
        import json as _json
        try:
            meta["metrics"] = _json.loads(Path(run.metrics_json_path).read_text(encoding="utf-8"))
        except Exception:
            pass

    return meta


@router.get("/leaderboard", response_model=LeaderboardResponse)
def get_leaderboard(
    db: Session = Depends(get_db),
) -> LeaderboardResponse:
    """Return a leaderboard of all evaluated models ranked by metrics."""
    runs = db.execute(select(Run).order_by(Run.created_at.desc())).scalars().all()

    entries: list[LeaderboardEntry] = []
    for run in runs:
        artifacts = db.execute(
            select(ModelArtifact).where(ModelArtifact.run_id == run.run_id)
        ).scalars().all()
        if not artifacts:
            continue

        metrics: dict[str, float | None] = {}
        if run.metrics_json_path and Path(run.metrics_json_path).exists():
            import json as _json
            try:
                raw = _json.loads(Path(run.metrics_json_path).read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    for k, v in raw.items():
                        if isinstance(v, (int, float)):
                            metrics[k] = float(v)
            except Exception:
                pass

        ckpt = run.ckpt_best or run.ckpt_path
        compatible = False
        if ckpt and Path(ckpt).exists():
            compatible, _ = _validate_checkpoint_compatibility(Path(ckpt))

        entries.append(LeaderboardEntry(
            run_id=run.run_id,
            tags=run.tags,
            compatible=compatible,
            metrics=metrics,
            evaluated_at=run.created_at.isoformat() if run.created_at else None,
        ))

    entries.sort(key=lambda e: e.metrics.get("knn_acc") or 0, reverse=True)
    return LeaderboardResponse(entries=entries)
