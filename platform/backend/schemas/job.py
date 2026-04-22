"""Pydantic schemas for jobs endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

JobType = Literal[
    "simulate", "convert", "bridge", "train", "eval", "infer", "export", "report"
]
JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


class JobSchema(BaseModel):
    """A single job record returned over the wire."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    job_id: str
    type: str
    status: JobStatus
    display_name: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress_pct: float = 0.0
    current_step: str | None = None
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    error: str | None = Field(default=None, validation_alias="error_msg")
    log_path: str | None = None
    run_id: str | None = None


class JobCreateRequest(BaseModel):
    """Payload for ``POST /api/jobs``."""

    type: JobType
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    display_name: str | None = None


class BatchJobCreateRequest(BaseModel):
    """Payload for ``POST /api/jobs/batch``."""

    type: JobType
    configs: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of config_overrides dicts, one per job",
    )
    display_name_prefix: str | None = None


class BatchJobCreateResponse(BaseModel):
    """Response listing all created jobs."""

    jobs: list[JobSchema]


class JobListResponse(BaseModel):
    """Response for ``GET /api/jobs``."""

    total: int
    items: list[JobSchema]


class JobProgressResponse(BaseModel):
    """Response for ``GET /api/jobs/{id}/progress``."""

    job_id: str
    progress_pct: float
    status: str
    current_step: str | None = None
    eta_seconds: int | None = None


class JobLogsResponse(BaseModel):
    """Response for ``GET /api/jobs/{id}/logs``."""

    job_id: str
    lines: list[str]


class JobCancelResponse(BaseModel):
    """Response for ``DELETE /api/jobs/{id}``."""

    job_id: str
    status: JobStatus
