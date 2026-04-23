"""PyTorch Dataset that reads :class:`ChannelSample` records via a manifest.

The dataset is deliberately thin: it filters the manifest at ``__init__`` time
and lazily loads sample artefacts in ``__getitem__`` by calling
:meth:`ChannelSample.from_pt_file`. Tensor conversion is the caller's
responsibility via ``transform``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # pragma: no cover - torch guaranteed in production env
    torch = None  # type: ignore[assignment]

    class _TorchDataset:  # type: ignore[no-redef]
        """Tiny stand-in so the module still imports without torch installed."""


from .contract import ChannelSample
from .manifest import Manifest

LinkFilter = Literal["UL", "DL", "both"]


class ChannelDataset(_TorchDataset):
    """Index into a :class:`Manifest`, loading artefacts on demand.

    Parameters
    ----------
    manifest :
        Fully-populated manifest. Only rows with ``status == 'succeeded'`` and
        a non-null ``path`` are considered.
    split :
        Which split to serve (``train`` / ``val`` / ``test`` / ``unassigned`` /
        ``"all"`` to disable filtering).
    source_filter :
        Optional list of source names (e.g. ``['quadriga_real', 'sionna_rt']``)
        to restrict to. ``None`` = all sources.
    link_filter :
        ``'UL'``, ``'DL'``, or ``'both'``.
    transform :
        Optional callable invoked on the record dict produced by
        :meth:`__getitem__`. Must return the final sample.
    """

    def __init__(
        self,
        manifest: Manifest,
        split: str = "train",
        source_filter: list[str] | None = None,
        link_filter: LinkFilter = "both",
        transform: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        super().__init__()
        self.manifest = manifest
        self.split = split
        self.source_filter = list(source_filter) if source_filter else None
        self.link_filter: LinkFilter = link_filter
        self.transform = transform
        self._index: pd.DataFrame = self._build_index()

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------
    def _build_index(self) -> pd.DataFrame:
        df = self.manifest.df
        if len(df) == 0:
            return df.copy()

        mask = df["path"].notna()
        # Keep only rows that have actually been materialised successfully.
        status_col = df["status"].astype(str)
        mask &= status_col.isin(["succeeded", "ready"])

        if self.split != "all":
            mask &= df["split"].astype(str) == self.split
        if self.source_filter is not None:
            mask &= df["source"].astype(str).isin(self.source_filter)
        if self.link_filter != "both":
            mask &= df["link"].astype(str) == self.link_filter

        return df.loc[mask].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Any:
        if idx < 0:
            idx += len(self._index)
        if not 0 <= idx < len(self._index):
            raise IndexError(f"index {idx} out of range for dataset of size {len(self._index)}")
        row = self._index.iloc[idx]
        path = Path(str(row["path"]))
        sample = ChannelSample.from_pt_file(path)
        record: dict[str, Any] = {
            "uuid": str(row["uuid"]) if pd.notna(row["uuid"]) else None,
            "h_true": _as_complex_tensor(sample.h_serving_true),
            "h_est": _as_complex_tensor(sample.h_serving_est),
            "h_interferers": (
                _as_complex_tensor(sample.h_interferers)
                if sample.h_interferers is not None
                else None
            ),
            "sinr_dB": float(sample.sinr_dB),
            "snr_dB": float(sample.snr_dB),
            "sir_dB": None if sample.sir_dB is None else float(sample.sir_dB),
            "link": sample.link,
            "source": sample.source,
            "sample_id": sample.sample_id,
            "channel_est_mode": sample.channel_est_mode,
            "meta": dict(sample.meta),
        }
        if self.transform is not None:
            return self.transform(record)
        return record

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    @property
    def index_frame(self) -> pd.DataFrame:
        """Expose the filtered manifest rows backing this dataset."""
        return self._index


def _as_complex_tensor(arr: np.ndarray) -> Any:
    """Convert a complex64 ndarray to a torch tensor when torch is available."""
    if torch is None:  # pragma: no cover - torch is required in prod envs
        return arr
    if arr.dtype != np.complex64:
        arr = arr.astype(np.complex64)
    return torch.from_numpy(arr)


__all__ = ["ChannelDataset", "LinkFilter"]
