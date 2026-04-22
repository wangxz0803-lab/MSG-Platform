"""Live field-measurement data source (future v3.1).

Placeholder for real over-the-air collections fed through the unified
contract. Full implementation is deferred to v3.1.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ..contract import ChannelSample
from .base import DataSource, register_source


@register_source
class FieldSource(DataSource):
    """Ingests channels captured from live field measurements."""

    name = "field"

    def iter_samples(self) -> Iterator[ChannelSample]:  # pragma: no cover - stub
        raise NotImplementedError(
            "FieldSource.iter_samples is a stub; will be implemented in v3.1 "
            "when live field-measurement ingestion is wired up."
        )

    def describe(self) -> dict[str, Any]:  # pragma: no cover - stub
        raise NotImplementedError(
            "FieldSource.describe is a stub; will be implemented in v3.1 "
            "when live field-measurement ingestion is wired up."
        )
