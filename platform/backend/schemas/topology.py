"""Pydantic schemas for topology preview."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TopologyPreviewRequest(BaseModel):
    num_sites: int = Field(default=7, ge=1, le=57, description="Number of cell sites")
    isd_m: float = Field(default=500.0, ge=50, le=5000, description="Inter-site distance (m)")
    sectors_per_site: int = Field(default=3, ge=1, le=6)
    tx_height_m: float = Field(default=25.0, ge=1, le=200)
    num_ues: int = Field(default=50, ge=1, le=10000)
    ue_distribution: Literal["uniform", "clustered", "hotspot"] = "uniform"
    ue_speed_kmh: float = Field(default=3.0, ge=0, le=500)


class SitePosition(BaseModel):
    site_id: int
    x: float
    y: float
    z: float
    sector_id: int
    azimuth_deg: float
    pci: int


class UEPosition(BaseModel):
    ue_id: int
    x: float
    y: float
    z: float = 1.5


class TopologyPreviewResponse(BaseModel):
    sites: list[SitePosition]
    ues: list[UEPosition]
    cell_radius_m: float
    bounds: dict[str, float]


class CollectConfig(BaseModel):
    """Rich collect configuration."""

    source: Literal["quadriga_multi", "sionna_rt", "internal_upload"] = "sionna_rt"
    num_sites: int = Field(default=7, ge=1, le=57)
    isd_m: float = Field(default=500.0, ge=50, le=5000)
    sectors_per_site: int = Field(default=3, ge=1, le=6)
    num_bs_antennas: int = Field(default=4, ge=1, le=256)
    carrier_freq_hz: float = Field(default=3.5e9)
    bandwidth_hz: float = Field(default=100e6)
    tx_power_dbm: float = Field(default=43.0, ge=0, le=60)
    tx_height_m: float = Field(default=25.0, ge=1, le=200)
    num_ues: int = Field(default=50, ge=1, le=10000)
    num_ue_antennas: int = Field(default=2, ge=1, le=16)
    ue_speed_kmh: float = Field(default=3.0, ge=0, le=500)
    ue_distribution: Literal["uniform", "clustered", "hotspot"] = "uniform"
    link: Literal["UL", "DL", "both"] = "DL"
    channel_est_mode: Literal["ideal", "ls_linear", "ls_mmse"] = "ls_linear"
    pilot_type: Literal["csi_rs_gold", "srs_zc"] = "csi_rs_gold"
    num_samples: int = Field(default=100, ge=1, le=100000)
    scenario: Literal["munich", "etoile", "custom_osm"] = "munich"
    custom_site_positions: list[dict[str, float]] | None = None
    custom_ue_positions: list[dict[str, float]] | None = None
