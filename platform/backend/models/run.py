"""Run ORM model -- one row per training run / experiment checkpoint set."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from ..db import Base


class Run(Base):
    """A single training run identified by ``run_id`` (artifacts/<run_id>)."""

    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    ckpt_path: Mapped[str | None] = mapped_column(String, nullable=True)
    ckpt_best: Mapped[str | None] = mapped_column(String, nullable=True)
    ckpt_last: Mapped[str | None] = mapped_column(String, nullable=True)
    metrics_json_path: Mapped[str | None] = mapped_column(String, nullable=True)
    config_path: Mapped[str | None] = mapped_column(String, nullable=True)
    metadata_path: Mapped[str | None] = mapped_column(String, nullable=True)
    git_sha: Mapped[str | None] = mapped_column(String, nullable=True)
    tags: Mapped[str | None] = mapped_column(String, nullable=True)
