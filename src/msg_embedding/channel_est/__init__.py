"""Three-tier pilot-based channel estimation (ideal / ls_linear / ls_mmse).

Public API: :func:`estimate_channel` is the canonical entry point used by
the data-collection stage (Phase 1.2).  Lower-level primitives are also
exposed for unit tests and advanced callers.
"""

from __future__ import annotations

from .interpolate import interp_2d, interp_frequency, interp_time
from .ls import ls_estimate
from .mmse import exponential_pdp_covariance, mmse_estimate
from .pipeline import EstimationMode, estimate_channel

__all__ = [
    "EstimationMode",
    "estimate_channel",
    "exponential_pdp_covariance",
    "interp_2d",
    "interp_frequency",
    "interp_time",
    "ls_estimate",
    "mmse_estimate",
]
