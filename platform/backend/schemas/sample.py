"""Pydantic schemas for datasets / samples endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SampleSchema(BaseModel):
    """Single sample as returned over the wire."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    sample_id: str | None = None
    uuid: str
    source: str | None = None
    link: Literal["UL", "DL"] | None = None
    snr_dB: float | None = Field(default=None, validation_alias="snr_db")
    sir_dB: float | None = Field(default=None, validation_alias="sir_db")
    sinr_dB: float | None = Field(default=None, validation_alias="sinr_db")
    ul_sir_dB: float | None = Field(default=None, validation_alias="ul_sir_db")
    dl_sir_dB: float | None = Field(default=None, validation_alias="dl_sir_db")
    num_interfering_ues: int | None = None
    link_pairing: Literal["single", "paired"] | None = Field(default="single")
    num_cells: int | None = None
    timestamp: datetime | None = Field(default=None, validation_alias="ts")
    status: str | None = None
    shard_id: str | None = None
    run_id: str | None = None


class DatasetSummary(BaseModel):
    """Aggregated per-source summary of available samples."""

    source: str
    count: int
    snr_mean: float | None = None
    snr_std: float | None = None
    sir_mean: float | None = None
    sinr_mean: float | None = None
    ul_sir_mean: float | None = None
    dl_sir_mean: float | None = None
    links: list[str] = Field(default_factory=list)
    has_paired: bool = False


class DatasetListResponse(BaseModel):
    """Response for ``GET /api/datasets``."""

    total: int
    items: list[DatasetSummary]


class DatasetSampleListResponse(BaseModel):
    """Response for ``GET /api/datasets/{source}/samples``."""

    total: int
    items: list[SampleSchema]


class DatasetCollectRequest(BaseModel):
    """Payload for ``POST /api/datasets/collect``."""

    source: Literal[
        "quadriga_real", "sionna_rt", "internal_sim", "internal_upload"
    ]
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    output_dir: str | None = None
