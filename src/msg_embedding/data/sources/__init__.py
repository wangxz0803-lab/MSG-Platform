"""Data source registry.

Every concrete :class:`DataSource` subclass registers itself via the
:func:`register_source` decorator, so the pipeline can instantiate sources by
name (``SOURCE_REGISTRY["quadriga_real"](cfg)``).

The registry itself lives in :mod:`.base` to avoid circular imports with the
stub modules below, and is re-exported here for convenience.
"""

from __future__ import annotations

# Import stubs so they register. Kept at module end to avoid circular imports.
from . import (  # noqa: E402,F401
    field,
    internal_sim,
    quadriga_real,
    sionna_rt,
)
from .base import SOURCE_REGISTRY, DataSource, register_source

__all__ = ["DataSource", "SOURCE_REGISTRY", "register_source"]
