"""Pydantic schemas for runs endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RunSchema(BaseModel):
    """A single training run summary."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    run_id: str
    created_at: datetime
    ckpt_path: str | None = None
    ckpt_best: str | None = None
    ckpt_last: str | None = None
    config: dict[str, Any] | None = None
    run_metadata: dict[str, Any] | None = Field(default=None, validation_alias="metadata")
    metrics: dict[str, Any] | None = None
    git_sha: str | None = None
    tags: list[str] = Field(default_factory=list)


class RunListResponse(BaseModel):
    """Response for ``GET /api/runs``."""

    total: int
    items: list[RunSchema]


class RunCompareItem(BaseModel):
    """Single entry in the compare response."""

    run_id: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class RunCompareResponse(BaseModel):
    """Response for ``GET /api/runs/compare``."""

    runs: list[RunCompareItem]
