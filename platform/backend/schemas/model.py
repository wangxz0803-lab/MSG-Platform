"""Pydantic schemas for model artifact endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

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
