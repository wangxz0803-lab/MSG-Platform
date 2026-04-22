"""Sync the parquet manifest into the SQLite Sample table.

Idempotent -- existing rows are updated in place, new rows inserted.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from msg_embedding.core.logging import get_logger

from ..models.sample import Sample
from ..settings import get_settings

_log = get_logger(__name__)


def _row_to_kwargs(row: dict[str, Any]) -> dict[str, Any]:
    """Translate a manifest row dict into Sample ORM kwargs."""
    ts = row.get("created_at") or row.get("updated_at")
    if ts is not None and not isinstance(ts, datetime):
        try:
            ts = datetime.fromisoformat(str(ts))
        except (TypeError, ValueError):
            ts = None

    def _as_str(v: Any) -> str | None:
        if v is None:
            return None
        try:
            import pandas as pd

            if pd.isna(v):
                return None
        except Exception:
            pass
        return str(v)

    def _as_float(v: Any) -> float | None:
        if v is None:
            return None
        try:
            import pandas as pd

            if pd.isna(v):
                return None
        except Exception:
            pass
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _as_int(v: Any) -> int | None:
        f = _as_float(v)
        return None if f is None else int(f)

    return {
        "uuid": _as_str(row.get("uuid")) or "",
        "sample_id": _as_str(row.get("sample_id")),
        "shard_id": _as_str(row.get("shard_id")),
        "source": _as_str(row.get("source")),
        "link": _as_str(row.get("link")),
        "snr_db": _as_float(row.get("snr_dB") if "snr_dB" in row else row.get("snr_db")),
        "sir_db": _as_float(row.get("sir_dB") if "sir_dB" in row else row.get("sir_db")),
        "sinr_db": _as_float(
            row.get("sinr_dB") if "sinr_dB" in row else row.get("sinr_db")
        ),
        "num_cells": _as_int(row.get("num_cells")),
        "ts": ts,
        "status": _as_str(row.get("status")),
        "job_id": _as_str(row.get("job_id")),
        "run_id": _as_str(row.get("run_id")),
        "path": _as_str(row.get("path")),
        "split": _as_str(row.get("split")),
    }


def sync_manifest_to_db(session: Session, manifest_path: Path | str | None = None) -> int:
    """Load manifest.parquet and upsert every row into the ``samples`` table.

    Returns the number of rows synced. Missing manifest or missing pyarrow is
    logged and treated as zero rows.
    """
    settings = get_settings()
    path = Path(manifest_path) if manifest_path is not None else settings.manifest_file
    if not path.exists():
        _log.info("manifest_not_found", path=str(path))
        return 0

    try:
        import pandas as pd  # noqa: F401
        import pyarrow.parquet as pq
    except ImportError as exc:
        _log.warning("pyarrow_unavailable", error=str(exc))
        return 0

    try:
        table = pq.read_table(path)
    except Exception as exc:
        _log.warning("manifest_read_failed", path=str(path), error=str(exc))
        return 0

    df = table.to_pandas()
    n = 0
    for record in df.to_dict(orient="records"):
        kwargs = _row_to_kwargs(record)
        if not kwargs["uuid"]:
            continue
        existing = session.get(Sample, kwargs["uuid"])
        if existing is None:
            session.add(Sample(**kwargs))
        else:
            for k, v in kwargs.items():
                if k == "uuid":
                    continue
                setattr(existing, k, v)
        n += 1
    session.commit()
    _log.info("manifest_synced", rows=n)
    return n
