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
    stage: str | None = Field(default="raw")
    serving_cell_id: int | None = None
    channel_est_mode: str | None = None
    bridged_path: str | None = None


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
    stage_counts: dict[str, int] = Field(default_factory=dict)


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


# --- Split management -------------------------------------------------------

class SplitComputeRequest(BaseModel):
    """Payload for ``POST /api/datasets/split``."""

    strategy: Literal["random", "by_position", "by_beam"] = "by_position"
    seed: int = 0
    ratios: list[float] = Field(default=[0.8, 0.1, 0.1], min_length=3, max_length=3)
    lock: bool = Field(
        default=True,
        description="If True, lock the split after computing so test/val sets are immutable.",
    )


class SplitInfoResponse(BaseModel):
    """Response for ``GET /api/datasets/split/status``."""

    locked: bool
    version: int
    strategy: str | None = None
    seed: int | None = None
    ratios: list[float] | None = None
    locked_at: str | None = None
    locked_test_uuids: int = 0
    counts: dict[str, int] = Field(default_factory=dict)


# --- Data export -------------------------------------------------------------

class DatasetExportRequest(BaseModel):
    """Payload for ``POST /api/datasets/export``."""

    format: Literal["hdf5", "webdataset", "pt_dir"] = "hdf5"
    split: str | None = Field(default="train", description="Split to export (train/val/test/None=all)")
    source_filter: str | None = None
    link_filter: Literal["UL", "DL"] | None = None
    min_snr: float | None = None
    max_snr: float | None = None
    export_name: str | None = None
    shard_size: int = Field(default=1000, ge=100, le=50000)
    include_interferers: bool = False


class DatasetExportStatusResponse(BaseModel):
    """Response for ``GET /api/datasets/exports``."""

    exports: list[dict[str, Any]]


class ExportFileInfo(BaseModel):
    """Single export package info."""

    name: str
    format: str
    num_samples: int
    total_bytes: int
    split_version: int
    created_at: str
    path: str
    download_url: str | None = None
