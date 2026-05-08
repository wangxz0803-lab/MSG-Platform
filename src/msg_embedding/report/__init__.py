"""Auto-report generation for ChannelHub."""

from __future__ import annotations

from .compare import MultiRunComparator
from .generator import ReportGenerator

__all__ = ["MultiRunComparator", "ReportGenerator"]
