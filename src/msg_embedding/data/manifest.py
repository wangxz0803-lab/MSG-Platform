"""Parquet-backed manifest for the ChannelHub data pipeline.

The manifest is the single source of truth about which samples have been
materialised, where they live on disk, what stage of the pipeline they are in
and how they should be split for training. It is designed to be:

* **DB-ready** — the schema mirrors the eventual platform SQLite / Postgres
  table, so :meth:`Manifest.to_sqlite` is a lossless dump.
* **Idempotent** — :meth:`Manifest.append` silently skips rows whose ``uuid``
  is already tracked so re-runs of ingestion jobs are safe.
* **Lazy** — ``load`` reads the parquet file on demand and ``save`` re-writes
  the whole file atomically via a tmp-file + rename.

The schema is intentionally wider than what Phase 1.6 needs (``job_id``,
``run_id``, ``stage``, ``status``, ``error_msg``) because Phase 6 platform
integration consumes these columns without requiring a schema migration.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

MANIFEST_SCHEMA = pa.schema(
    [
        ("uuid", pa.string()),
        ("job_id", pa.string()),
        ("run_id", pa.string()),
        ("source", pa.string()),
        ("shard_id", pa.int32()),
        ("sample_id", pa.int32()),
        ("stage", pa.string()),
        ("status", pa.string()),
        ("link", pa.string()),
        ("snr_dB", pa.float32()),
        ("sir_dB", pa.float32()),
        ("sinr_dB", pa.float32()),
        ("num_cells", pa.int32()),
        ("serving_cell_id", pa.int32()),
        ("ue_x", pa.float32()),
        ("ue_y", pa.float32()),
        ("ue_z", pa.float32()),
        ("channel_est_mode", pa.string()),
        ("split", pa.string()),
        ("hash", pa.string()),
        ("path", pa.string()),
        ("error_msg", pa.string()),
        ("created_at", pa.timestamp("us")),
        ("updated_at", pa.timestamp("us")),
    ]
)

#: Column order as plain Python list — used when building empty DataFrames so
#: round-tripping through parquet preserves column order.
COLUMNS: list[str] = [f.name for f in MANIFEST_SCHEMA]

#: Mapping of column -> (pandas dtype, default value for missing rows).
_COLUMN_DEFAULTS: dict[str, tuple[str, Any]] = {
    "uuid": ("string", None),
    "job_id": ("string", None),
    "run_id": ("string", None),
    "source": ("string", None),
    "shard_id": ("Int32", pd.NA),
    "sample_id": ("Int32", pd.NA),
    "stage": ("string", "raw"),
    "status": ("string", "pending"),
    "link": ("string", None),
    "snr_dB": ("Float32", pd.NA),
    "sir_dB": ("Float32", pd.NA),
    "sinr_dB": ("Float32", pd.NA),
    "num_cells": ("Int32", pd.NA),
    "serving_cell_id": ("Int32", pd.NA),
    "ue_x": ("Float32", pd.NA),
    "ue_y": ("Float32", pd.NA),
    "ue_z": ("Float32", pd.NA),
    "channel_est_mode": ("string", None),
    "split": ("string", "unassigned"),
    "hash": ("string", None),
    "path": ("string", None),
    "error_msg": ("string", None),
    "created_at": ("datetime64[us]", None),
    "updated_at": ("datetime64[us]", None),
}

SplitStrategy = Literal["random", "by_position", "by_beam"]


def _empty_frame() -> pd.DataFrame:
    """Return a schema-compliant empty DataFrame."""
    cols: dict[str, pd.Series] = {}
    for name, (dtype, _default) in _COLUMN_DEFAULTS.items():
        cols[name] = pd.Series([], dtype=dtype)
    return pd.DataFrame(cols, columns=COLUMNS)


def _strip_tz(ts: Any) -> Any:
    """Return a tz-naive representation of ``ts`` (UTC-aligned)."""
    if ts is None:
        return ts
    if isinstance(ts, datetime) and ts.tzinfo is not None:
        return ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts


def _coerce_row(row: dict[str, Any], now: datetime) -> dict[str, Any]:
    """Fill missing defaults for an inbound manifest row.

    Mutates nothing — returns a new dict.
    """
    out: dict[str, Any] = {}
    for name, (_dtype, default) in _COLUMN_DEFAULTS.items():
        out[name] = row.get(name, default)
    if out.get("created_at") is None:
        out["created_at"] = now
    if out.get("updated_at") is None:
        out["updated_at"] = now
    out["created_at"] = _strip_tz(out["created_at"])
    out["updated_at"] = _strip_tz(out["updated_at"])
    return out


def _coerce_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Cast each column of ``df`` to the manifest dtypes in-place-copy style."""
    df = df.reindex(columns=COLUMNS)
    for name, (dtype, _default) in _COLUMN_DEFAULTS.items():
        series = df[name]
        try:
            df[name] = series.astype(dtype)
        except (TypeError, ValueError):
            df[name] = series
    return df


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class Manifest:
    """Append-only manifest of :class:`ChannelSample` rows.

    Parameters
    ----------
    path :
        Target parquet file. If it exists on construction the contents are
        loaded eagerly; otherwise a fresh in-memory DataFrame is created and
        only written when :meth:`save` is called.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._df: pd.DataFrame = _empty_frame()
        if self.path.exists():
            self.load()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def df(self) -> pd.DataFrame:
        """Return the underlying DataFrame (not a copy)."""
        return self._df

    def __len__(self) -> int:
        return len(self._df)

    # ------------------------------------------------------------------
    # Mutations (append is defined below, after split locking)
    # ------------------------------------------------------------------
    def update(self, uuid: str, **fields: Any) -> None:
        """Patch fields on the row identified by ``uuid``.

        Raises ``KeyError`` if no such row exists. ``updated_at`` is always
        bumped to ``now``.
        """
        mask = self._df["uuid"].astype(str) == str(uuid)
        if not mask.any():
            raise KeyError(f"uuid {uuid!r} not present in manifest")
        unknown = set(fields) - set(COLUMNS)
        if unknown:
            raise ValueError(f"unknown manifest column(s): {sorted(unknown)}")
        for col, val in fields.items():
            if col in ("created_at", "updated_at"):
                val = _strip_tz(val)
            self._df.loc[mask, col] = val
        self._df.loc[mask, "updated_at"] = _strip_tz(datetime.now(timezone.utc))
        self._df = _coerce_frame(self._df)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def query(self, **filters: Any) -> pd.DataFrame:
        """Return rows matching all ``col == value`` filters.

        ``value`` may be a scalar or a list/tuple/set of accepted values. The
        returned DataFrame is a copy (safe to mutate).
        """
        df = self._df
        for col, val in filters.items():
            if col not in COLUMNS:
                raise ValueError(f"unknown manifest column: {col!r}")
            if isinstance(val, (list, tuple, set)):
                df = df[df[col].isin(list(val))]
            else:
                df = df[df[col] == val]
        return df.copy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        """Atomically write the manifest to :attr:`path` as parquet."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        df = _coerce_frame(self._df.copy())
        table = pa.Table.from_pandas(df, schema=MANIFEST_SCHEMA, preserve_index=False)
        pq.write_table(table, tmp)
        tmp.replace(self.path)

    def load(self) -> None:
        """Reload the manifest from :attr:`path`. Overwrites any in-memory rows."""
        if not self.path.exists():
            self._df = _empty_frame()
            return
        table = pq.read_table(self.path)
        df = table.to_pandas()
        self._df = _coerce_frame(df)

    # ------------------------------------------------------------------
    # Platform hook
    # ------------------------------------------------------------------
    def to_sqlite(self, db_path: Path | str, table_name: str = "manifest") -> None:
        """Dump the manifest as a SQLite table (DB-ready handoff for Phase 6)."""
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        df = _coerce_frame(self._df.copy())
        with sqlite3.connect(db_path) as conn:
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_uuid " f"ON {table_name}(uuid)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_status " f"ON {table_name}(status)"
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Split locking (for fixed test sets)
    # ------------------------------------------------------------------
    @property
    def _split_meta_path(self) -> Path:
        return self.path.with_name(self.path.stem + "_split_meta.json")

    def _load_split_meta(self) -> dict[str, Any]:
        if self._split_meta_path.exists():
            return json.loads(self._split_meta_path.read_text(encoding="utf-8"))
        return {}

    def _save_split_meta(self, meta: dict[str, Any]) -> None:
        self._split_meta_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._split_meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._split_meta_path)

    @property
    def is_split_locked(self) -> bool:
        return bool(self._load_split_meta().get("locked", False))

    @property
    def split_version(self) -> int:
        return int(self._load_split_meta().get("version", 0))

    def get_split_info(self) -> dict[str, Any]:
        """Return full split metadata including lock status, version, counts."""
        meta = self._load_split_meta()
        counts = {}
        if len(self._df) > 0 and "split" in self._df.columns:
            for s in ("train", "val", "test", "unassigned"):
                counts[s] = int((self._df["split"] == s).sum())
        return {
            "locked": meta.get("locked", False),
            "version": meta.get("version", 0),
            "strategy": meta.get("strategy"),
            "seed": meta.get("seed"),
            "ratios": meta.get("ratios"),
            "locked_at": meta.get("locked_at"),
            "locked_test_uuids": meta.get("locked_test_uuids_count", 0),
            "counts": counts,
        }

    def lock_split(self) -> dict[str, Any]:
        """Lock the current split — test/val sets become immutable.

        After locking:
        - ``compute_split`` raises if called again.
        - ``append`` auto-assigns new rows to ``"train"``.
        - The test/val UUID sets are recorded in the sidecar file.
        """
        if self.is_split_locked:
            raise RuntimeError("split is already locked")

        n_test = int((self._df["split"] == "test").sum())
        n_val = int((self._df["split"] == "val").sum())
        if n_test == 0 and n_val == 0:
            raise ValueError("no test/val rows to lock — call compute_split first")

        test_uuids = self._df.loc[self._df["split"] == "test", "uuid"].dropna().tolist()
        val_uuids = self._df.loc[self._df["split"] == "val", "uuid"].dropna().tolist()

        meta = self._load_split_meta()
        meta["locked"] = True
        meta["version"] = meta.get("version", 0) + 1
        meta["locked_at"] = datetime.now(timezone.utc).isoformat()
        meta["locked_test_uuids"] = test_uuids
        meta["locked_val_uuids"] = val_uuids
        meta["locked_test_uuids_count"] = len(test_uuids)
        meta["locked_val_uuids_count"] = len(val_uuids)
        self._save_split_meta(meta)
        return self.get_split_info()

    def unlock_split(self) -> None:
        """Unlock the split (admin operation). All rows remain as-is."""
        meta = self._load_split_meta()
        meta["locked"] = False
        self._save_split_meta(meta)

    def compute_split(
        self,
        strategy: SplitStrategy = "random",
        seed: int = 0,
        ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> None:
        """Assign train/val/test labels to every row according to ``strategy``.

        If the split is locked, only ``"unassigned"`` rows are assigned to
        ``"train"`` — test/val sets are never changed.
        """
        if self.is_split_locked:
            self._assign_unassigned_to_train()
            return

        if len(self._df) == 0:
            return
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"ratios must sum to 1, got {sum(ratios)}")
        train_r, val_r, _test_r = ratios

        rng = np.random.default_rng(seed)

        if strategy == "random":
            keys = np.arange(len(self._df))
            group_labels = keys
        elif strategy == "by_position":
            xs = self._df["ue_x"].fillna(0.0).to_numpy(dtype=float)
            ys = self._df["ue_y"].fillna(0.0).to_numpy(dtype=float)
            group_labels = np.array(
                [f"{int(round(x))}:{int(round(y))}" for x, y in zip(xs, ys, strict=False)]
            )
        elif strategy == "by_beam":
            group_labels = self._df["serving_cell_id"].fillna(-1).astype(int).to_numpy()
        else:
            raise ValueError(f"unknown split strategy: {strategy!r}")

        unique_groups = pd.unique(group_labels)
        perm = rng.permutation(len(unique_groups))
        shuffled = unique_groups[perm]
        n = len(shuffled)
        n_train = int(round(n * train_r))
        n_val = int(round(n * val_r))
        train_groups = set(shuffled[:n_train].tolist())
        val_groups = set(shuffled[n_train : n_train + n_val].tolist())

        splits = []
        for g in group_labels:
            if g in train_groups:
                splits.append("train")
            elif g in val_groups:
                splits.append("val")
            else:
                splits.append("test")

        self._df["split"] = pd.Series(splits, dtype="string")
        self._df["updated_at"] = _strip_tz(datetime.now(timezone.utc))
        self._df = _coerce_frame(self._df)

        meta = self._load_split_meta()
        meta["version"] = meta.get("version", 0) + 1
        meta["strategy"] = strategy
        meta["seed"] = seed
        meta["ratios"] = list(ratios)
        meta["computed_at"] = datetime.now(timezone.utc).isoformat()
        self._save_split_meta(meta)

    def _assign_unassigned_to_train(self) -> None:
        """Auto-assign unassigned rows to train (used when split is locked)."""
        mask = self._df["split"] == "unassigned"
        if mask.any():
            self._df.loc[mask, "split"] = "train"
            self._df.loc[mask, "updated_at"] = _strip_tz(datetime.now(timezone.utc))
            self._df = _coerce_frame(self._df)

    def append(self, rows: Iterable[dict[str, Any]]) -> None:
        """Append ``rows`` to the manifest, silently skipping known uuids.

        If the split is locked, new rows are automatically assigned to train.
        """
        rows = list(rows)
        if not rows:
            return
        now = datetime.now(timezone.utc)
        coerced = [_coerce_row(r, now) for r in rows]
        existing = set(self._df["uuid"].dropna().astype(str).tolist())
        fresh = [r for r in coerced if r["uuid"] not in existing]
        if not fresh:
            return

        if self.is_split_locked:
            for r in fresh:
                if r.get("split") in (None, "unassigned"):
                    r["split"] = "train"

        new_df = pd.DataFrame(fresh, columns=COLUMNS)
        new_df = _coerce_frame(new_df)
        if len(self._df) == 0:
            self._df = new_df
        else:
            self._df = pd.concat([self._df, new_df], ignore_index=True)
        self._df = _coerce_frame(self._df)

    def get_split_uuids(self, split: str) -> list[str]:
        """Return all UUIDs belonging to the given split."""
        return self._df.loc[self._df["split"] == split, "uuid"].dropna().tolist()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_content_hash(path: Path | str, chunk: int = 1 << 20) -> str:
    """Compute a sha256 hash of a file's contents for idempotency checks."""
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


__all__ = [
    "MANIFEST_SCHEMA",
    "COLUMNS",
    "Manifest",
    "SplitStrategy",
    "compute_content_hash",
    "SPLIT_META_VERSION",
]

SPLIT_META_VERSION = 1
