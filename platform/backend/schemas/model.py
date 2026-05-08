"""Pydantic schemas for model artifact endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ArtifactFormat = Literal["pt", "onnx", "torchscript"]
ExportFormat = Literal["onnx", "torchscript", "both"]


class ModelArtifactSchema(BaseModel):
    """A single registered model artifact."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: str
    format: ArtifactFormat
    path: str
    created_at: datetime
    size_bytes: int = 0


class ModelListResponse(BaseModel):
    """Response for ``GET /api/models``."""

    total: int
    items: list[ModelArtifactSchema]


class ExportRequest(BaseModel):
    """Payload for ``POST /api/models/{run_id}/export``."""

    format: ExportFormat


class ModelUploadResponse(BaseModel):
    """Response for ``POST /api/models/upload``."""

    run_id: str
    artifact_id: int
    path: str
    format: ArtifactFormat
    size_bytes: int
    compatible: bool
    compatibility_detail: str | None = None


class ModelEvalRequest(BaseModel):
    """Payload for ``POST /api/models/{run_id}/evaluate``."""

    test_split: str = "test"
    limit: int | None = None
    device: str = "cpu"


class ModelInferRequest(BaseModel):
    """Payload for ``POST /api/models/{run_id}/infer``."""

    input_path: str | None = None
    split: str = "test"
    limit: int | None = None
    device: str = "cpu"
    batch_size: int = 32
    output_name: str | None = None


class LeaderboardEntry(BaseModel):
    """A single row in the model leaderboard."""

    run_id: str
    tags: str | None = None
    compatible: bool = True
    test_split_version: int | None = None
    metrics: dict[str, float | None] = Field(default_factory=dict)
    evaluated_at: str | None = None


class LeaderboardResponse(BaseModel):
    """Response for ``GET /api/models/leaderboard``."""

    entries: list[LeaderboardEntry]
