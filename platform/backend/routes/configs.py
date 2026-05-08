"""Configs endpoints -- JSON schema + resolved Hydra defaults."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ..settings import get_settings

router = APIRouter(prefix="/api/configs", tags=["configs"])

_VALID_SECTIONS = {"data", "model", "eval", "infer", "all"}


def _read_json_schema() -> dict[str, Any]:
    """Return the contents of ``configs/_schema.json`` as a dict."""
    settings = get_settings()
    schema_file = settings.schema_file
    if not schema_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"schema file not found at {schema_file}",
        )
    try:
        data = json.loads(schema_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"invalid schema JSON: {exc}") from exc
    return data


def _read_hydra_defaults(section: str) -> dict[str, Any]:
    """Return resolved Hydra defaults for ``section`` (or the whole config)."""
    settings = get_settings()
    cfg_dir = settings.configs_path

    try:
        from msg_embedding.utils.config import get_config_as_dict, load_config

        cfg = load_config()
        data = get_config_as_dict(cfg)
    except Exception:
        data = _load_yaml_defaults(cfg_dir)

    if section == "all":
        return data
    return data.get(section, {}) if isinstance(data, dict) else {}


def _load_yaml_defaults(cfg_dir: Path) -> dict[str, Any]:
    """Read each ``configs/<group>/default.yaml`` and stitch into one dict."""
    out: dict[str, Any] = {}
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return out
    for group in ("data", "model", "eval", "infer"):
        p = cfg_dir / group / "default.yaml"
        if p.exists():
            try:
                out[group] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except Exception:
                out[group] = {}
    return out


@router.get("/schema")
def get_schema() -> dict[str, Any]:
    """Raw JSON Schema describing the full Hydra config surface."""
    return _read_json_schema()


@router.get("/defaults")
def get_defaults(
    section: str = Query(default="all", description="Config section to return"),
) -> dict[str, Any]:
    """Resolved Hydra defaults for the given section."""
    if section not in _VALID_SECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"section must be one of {sorted(_VALID_SECTIONS)}",
        )
    return _read_hydra_defaults(section)
