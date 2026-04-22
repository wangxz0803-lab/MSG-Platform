"""Job ORM model -- one row per dispatched compute task."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..db import Base


class Job(Base):
    """A single compute job (simulate / convert / bridge / train / eval / infer / export / report)."""

    __tablename__ = "jobs"

    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    type: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[str] = mapped_column(String, index=True, default="queued")
    display_name: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    params_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    progress_pct: Mapped[float] = mapped_column(Float, default=0.0)
    log_path: Mapped[str | None] = mapped_column(String, nullable=True)
    error_msg: Mapped[str | None] = mapped_column(Text, nullable=True)
    run_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
