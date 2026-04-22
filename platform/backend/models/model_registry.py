"""ModelArtifact ORM model -- catalogs exported checkpoint artifacts per run."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from ..db import Base


class ModelArtifact(Base):
    """A single exported model artifact (pt / onnx / torchscript) for a run."""

    __tablename__ = "model_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("runs.run_id"), index=True)
    format: Mapped[str] = mapped_column(String, index=True)
    path: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
