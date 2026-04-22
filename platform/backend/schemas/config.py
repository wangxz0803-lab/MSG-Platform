"""Pydantic schemas for config endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ConfigSchemaResponse(BaseModel):
    """Raw JSON-Schema dump (``configs/_schema.json``)."""

    schema_: dict[str, Any]


class ConfigDefaultsResponse(BaseModel):
    """Resolved Hydra defaults (optionally scoped to a section)."""

    section: str
    config: dict[str, Any]
