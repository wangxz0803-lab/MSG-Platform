"""Sample ORM model -- mirrors the parquet manifest schema."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from ..db import Base


class Sample(Base):
    """A single channel-data row imported from the parquet manifest."""

    __tablename__ = "samples"

    uuid: Mapped[str] = mapped_column(String, primary_key=True)
    sample_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    shard_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    source: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    link: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    snr_db: Mapped[float | None] = mapped_column(Float, nullable=True)
    sir_db: Mapped[float | None] = mapped_column(Float, nullable=True)
    sinr_db: Mapped[float | None] = mapped_column(Float, nullable=True)
    ul_sir_db: Mapped[float | None] = mapped_column(Float, nullable=True)
    dl_sir_db: Mapped[float | None] = mapped_column(Float, nullable=True)
    num_interfering_ues: Mapped[int | None] = mapped_column(Integer, nullable=True)
    link_pairing: Mapped[str | None] = mapped_column(String, nullable=True, default="single")
    num_cells: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ts: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    job_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    run_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    path: Mapped[str | None] = mapped_column(String, nullable=True)
    split: Mapped[str | None] = mapped_column(String, nullable=True)
    stage: Mapped[str | None] = mapped_column(String, nullable=True, index=True, default="raw")
    serving_cell_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    channel_est_mode: Mapped[str | None] = mapped_column(String, nullable=True)
    ue_x: Mapped[float | None] = mapped_column(Float, nullable=True)
    ue_y: Mapped[float | None] = mapped_column(Float, nullable=True)
    bridged_path: Mapped[str | None] = mapped_column(String, nullable=True)
