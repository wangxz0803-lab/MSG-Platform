"""Unified ChannelSample data contract for the multi-cell PHY simulation pipeline.

This module defines the single canonical schema that every DataSource (QuaDRiGa
legacy, QuaDRiGa multi-cell, Sionna RT, internal simulator, live field data)
must emit. Downstream feature extraction, training and evaluation all consume
``ChannelSample`` instances, so this contract is the narrow-waist of the
MSG-Embedding data pipeline.

Design notes
------------
* Uses **pydantic v2** ``BaseModel`` for runtime validation. Numpy arrays are
  carried as regular fields with ``arbitrary_types_allowed=True`` plus explicit
  field validators — this avoids pydantic trying to coerce ndarray into a dict.
* Complex arrays are stored as ``np.complex64`` on disk and in memory. The
  serialisation scheme in :meth:`ChannelSample.to_dict` splits each complex
  array into a ``(real, imag)`` pair of ``float32`` arrays so the result is
  JSON-friendly when arrays are later converted to lists.
* ``to_parquet_row`` flattens only scalar metadata fields. Array payloads are
  assumed to live in a separate artefact (``.pt`` / ``.npz``) referenced by
  ``sample_id`` — the manifest row carries an ``array_path`` slot that writers
  are expected to fill in.
* No ``to_sqlite`` method: that lives in Phase 1.6.
"""

from __future__ import annotations

import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema, field_validator, model_validator

# Annotation that replaces the opaque numpy ndarray schema with a descriptive
# placeholder so :func:`ChannelSample.model_json_schema` succeeds. The real
# shape/dtype constraints are enforced by the field validators below.
_NdarraySchema = WithJsonSchema(
    {
        "type": "object",
        "title": "NdArray",
        "description": "numpy.ndarray; exact shape/dtype enforced by validator",
        "properties": {
            "dtype": {"type": "string"},
            "shape": {"type": "array", "items": {"type": "integer"}},
        },
    }
)
NdArray = Annotated[np.ndarray, _NdarraySchema]

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

LinkType = Literal["UL", "DL"]
LinkPairing = Literal["single", "paired"]
ChannelEstMode = Literal["ideal", "ls_linear", "ls_mmse"]
SourceType = Literal[
    "quadriga_real",
    "sionna_rt",
    "internal_sim",
    "field",
]

_SNR_BOUNDS = (-50.0, 50.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _complex_to_pair(arr: np.ndarray) -> dict[str, Any]:
    """Split a complex64 ndarray into float32 (real, imag) lists + shape."""
    arr = np.ascontiguousarray(arr)
    return {
        "dtype": "complex64",
        "shape": list(arr.shape),
        "real": arr.real.astype(np.float32).tolist(),
        "imag": arr.imag.astype(np.float32).tolist(),
    }


def _pair_to_complex(payload: dict[str, Any]) -> np.ndarray:
    """Inverse of :func:`_complex_to_pair`."""
    shape = tuple(payload["shape"])
    real = np.asarray(payload["real"], dtype=np.float32).reshape(shape)
    imag = np.asarray(payload["imag"], dtype=np.float32).reshape(shape)
    return (real + 1j * imag).astype(np.complex64)


def _float_array_to_list(arr: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "values": arr.astype(np.float64).tolist(),
    }


def _list_to_float_array(payload: dict[str, Any], dtype: np.dtype) -> np.ndarray:
    shape = tuple(payload["shape"])
    return np.asarray(payload["values"], dtype=dtype).reshape(shape)


# ---------------------------------------------------------------------------
# ChannelSample
# ---------------------------------------------------------------------------


class ChannelSample(BaseModel):
    """Canonical channel-measurement record used across the pipeline.

    Shapes
    ------
    * ``h_serving_true``, ``h_serving_est``: ``[T, RB, BS_ant, UE_ant]``
    * ``h_interferers``: ``[K-1, T, RB, BS_ant, UE_ant]`` (``None`` if K == 1)
    * ``interference_signal``: ``[T, N_RE_obs, ...]`` — PHY-level interference
      as observed in time-frequency domain. Optional.
    * ``ue_position``: ``[3]`` float64 meters.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
    )

    # --- Channel payloads -------------------------------------------------
    h_serving_true: NdArray = Field(
        ...,
        description="[T, RB, BS_ant, UE_ant] complex64 ideal serving-cell channel.",
    )
    h_serving_est: NdArray = Field(
        ...,
        description="[T, RB, BS_ant, UE_ant] complex64 estimated serving-cell channel.",
    )
    h_interferers: NdArray | None = Field(
        default=None,
        description="[K-1, T, RB, BS_ant, UE_ant] complex64 interferer channels; None if K=1.",
    )
    interference_signal: NdArray | None = Field(
        default=None,
        description="[T, N_RE_obs, ...] observed PHY-layer interference.",
    )

    # --- Scalar link-level stats -----------------------------------------
    noise_power_dBm: float
    snr_dB: float
    sir_dB: float | None = None
    sinr_dB: float

    # --- SSB measurements (multi-cell) -----------------------------------
    ssb_rsrp_dBm: list[float] | None = Field(
        default=None,
        description="Per-cell best-beam SS-RSRP in dBm, length K.",
    )
    ssb_rsrq_dB: list[float] | None = Field(
        default=None,
        description="Per-cell RSRQ in dB, length K.",
    )
    ssb_sinr_dB: list[float] | None = Field(
        default=None,
        description="Per-cell SS-SINR in dB, length K.",
    )
    ssb_best_beam_idx: list[int] | None = Field(
        default=None,
        description="Per-cell best SSB beam index, length K.",
    )
    ssb_pcis: list[int] | None = Field(
        default=None,
        description="Physical Cell IDs for SSB measurements, length K.",
    )

    # --- Link metadata ---------------------------------------------------
    link: LinkType
    channel_est_mode: ChannelEstMode
    serving_cell_id: int
    ue_position: NdArray | None = None
    channel_model: str | None = Field(
        default=None,
        description="Channel model profile, e.g., 'TDL-C', 'TDL-A'.",
    )
    tdd_pattern: str | None = Field(
        default=None,
        description="TDD slot pattern, e.g., 'DDDSU'.",
    )

    # --- Paired UL/DL channels (interference-aware) -----------------------
    link_pairing: LinkPairing = Field(
        default="single",
        description="'single': legacy mode (h_serving_est only). "
        "'paired': UL+DL ideal/estimated channels populated.",
    )
    h_ul_true: NdArray | None = Field(
        default=None,
        description="[T, RB, BS_ant, UE_ant] complex64 ideal UL channel (BS perspective).",
    )
    h_ul_est: NdArray | None = Field(
        default=None,
        description="[T, RB, BS_ant, UE_ant] complex64 UL channel estimated with interference.",
    )
    h_dl_true: NdArray | None = Field(
        default=None,
        description="[T, RB, BS_ant, UE_ant] complex64 ideal DL channel.",
    )
    h_dl_est: NdArray | None = Field(
        default=None,
        description="[T, RB, BS_ant, UE_ant] complex64 DL channel estimated with interference.",
    )
    ul_sir_dB: float | None = Field(default=None, description="UL signal-to-interference ratio.")
    dl_sir_dB: float | None = Field(default=None, description="DL signal-to-interference ratio.")
    num_interfering_ues: int | None = Field(
        default=None,
        description="Number of interfering UEs in UL estimation.",
    )
    ssb_rsrp_true_dBm: list[float] | None = Field(
        default=None,
        description="Per-cell SS-RSRP under ideal conditions (no inter-cell interference).",
    )
    ssb_sinr_true_dB: list[float] | None = Field(
        default=None,
        description="Per-cell SS-SINR under ideal conditions.",
    )

    # --- Provenance ------------------------------------------------------
    source: SourceType
    sample_id: str = Field(..., description="UUID4 string primary key.")
    created_at: datetime = Field(..., description="ISO8601 UTC timestamp.")
    meta: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator(
        "h_serving_true",
        "h_serving_est",
        "h_interferers",
        "h_ul_true",
        "h_ul_est",
        "h_dl_true",
        "h_dl_est",
        mode="before",
    )
    @classmethod
    def _ensure_complex64(cls, v: Any) -> Any:
        if v is None:
            return v
        if not isinstance(v, np.ndarray):
            raise TypeError(f"expected numpy.ndarray, got {type(v).__name__}")
        if v.dtype != np.complex64:
            raise ValueError(f"expected dtype complex64, got {v.dtype}")
        return v

    @field_validator("interference_signal", mode="before")
    @classmethod
    def _ensure_complex64_interference(cls, v: Any) -> Any:
        if v is None:
            return v
        if not isinstance(v, np.ndarray):
            raise TypeError(f"expected numpy.ndarray, got {type(v).__name__}")
        if v.dtype != np.complex64:
            raise ValueError(f"expected dtype complex64, got {v.dtype}")
        return v

    @field_validator("ue_position", mode="before")
    @classmethod
    def _ensure_position(cls, v: Any) -> Any:
        if v is None:
            return v
        if not isinstance(v, np.ndarray):
            raise TypeError(f"ue_position must be np.ndarray, got {type(v).__name__}")
        if v.shape != (3,):
            raise ValueError(f"ue_position must have shape [3], got {v.shape}")
        if v.dtype != np.float64:
            # coerce silently: positions are small and must be float64 for precision
            v = v.astype(np.float64)
        return v

    @field_validator("sample_id")
    @classmethod
    def _ensure_uuid4(cls, v: str) -> str:
        try:
            parsed = uuid.UUID(v)
        except (ValueError, AttributeError, TypeError) as exc:
            raise ValueError(f"sample_id must be a UUID4 string, got {v!r}") from exc
        if parsed.version != 4:
            raise ValueError(f"sample_id must be UUID version 4, got version {parsed.version}")
        return str(parsed)

    @field_validator("snr_dB", "sinr_dB")
    @classmethod
    def _check_db_required(cls, v: float) -> float:
        lo, hi = _SNR_BOUNDS
        if not (lo <= v <= hi):
            raise ValueError(f"value {v} dB out of range [{lo}, {hi}]")
        return v

    @field_validator("sir_dB")
    @classmethod
    def _check_sir(cls, v: float | None) -> float | None:
        if v is None:
            return v
        lo, hi = _SNR_BOUNDS
        if not (lo <= v <= hi):
            raise ValueError(f"sir_dB {v} out of range [{lo}, {hi}]")
        return v

    @model_validator(mode="after")
    def _check_shape_consistency(self) -> ChannelSample:
        true_shape = self.h_serving_true.shape
        est_shape = self.h_serving_est.shape
        if len(true_shape) != 4:
            raise ValueError(f"h_serving_true must be 4-D [T,RB,BS,UE], got shape {true_shape}")
        if true_shape != est_shape:
            raise ValueError(
                "h_serving_true and h_serving_est must have identical shape, "
                f"got {true_shape} vs {est_shape}"
            )
        T, RB, BS, UE = true_shape

        if self.h_interferers is not None:
            ih = self.h_interferers.shape
            if len(ih) != 5:
                raise ValueError(f"h_interferers must be 5-D [K-1,T,RB,BS,UE], got {ih}")
            if ih[1:] != (T, RB, BS, UE):
                raise ValueError(
                    "h_interferers trailing dims must match h_serving_*: "
                    f"got {ih[1:]} vs {(T, RB, BS, UE)}"
                )
            if ih[0] < 1:
                raise ValueError("h_interferers leading dim (K-1) must be >= 1")

        if self.interference_signal is not None:
            if self.interference_signal.ndim < 2:
                raise ValueError("interference_signal must have >=2 dims [T, N_RE_obs, ...]")
            if self.interference_signal.shape[0] != T:
                raise ValueError(
                    "interference_signal leading dim must match T "
                    f"({self.interference_signal.shape[0]} vs {T})"
                )

        if self.link_pairing == "paired":
            for name in ("h_ul_true", "h_ul_est", "h_dl_true", "h_dl_est"):
                arr = getattr(self, name)
                if arr is None:
                    raise ValueError(
                        f"link_pairing='paired' requires {name} to be set"
                    )
                if arr.ndim != 4:
                    raise ValueError(f"{name} must be 4-D [T,RB,BS,UE], got {arr.shape}")
            ul_t_shape = self.h_ul_true.shape  # type: ignore[union-attr]
            if self.h_ul_est.shape != ul_t_shape:  # type: ignore[union-attr]
                raise ValueError(
                    f"h_ul_true and h_ul_est shape mismatch: "
                    f"{ul_t_shape} vs {self.h_ul_est.shape}"  # type: ignore[union-attr]
                )
            dl_t_shape = self.h_dl_true.shape  # type: ignore[union-attr]
            if self.h_dl_est.shape != dl_t_shape:  # type: ignore[union-attr]
                raise ValueError(
                    f"h_dl_true and h_dl_est shape mismatch: "
                    f"{dl_t_shape} vs {self.h_dl_est.shape}"  # type: ignore[union-attr]
                )

        return self

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict.

        Complex arrays are split into ``(real, imag)`` float32 payloads,
        ``datetime`` is ISO-8601 UTC, and ``ue_position`` is a plain list.
        """
        out: dict[str, Any] = {
            "h_serving_true": _complex_to_pair(self.h_serving_true),
            "h_serving_est": _complex_to_pair(self.h_serving_est),
            "h_interferers": (
                _complex_to_pair(self.h_interferers) if self.h_interferers is not None else None
            ),
            "interference_signal": (
                _complex_to_pair(self.interference_signal)
                if self.interference_signal is not None
                else None
            ),
            "link_pairing": self.link_pairing,
            "h_ul_true": (
                _complex_to_pair(self.h_ul_true) if self.h_ul_true is not None else None
            ),
            "h_ul_est": (
                _complex_to_pair(self.h_ul_est) if self.h_ul_est is not None else None
            ),
            "h_dl_true": (
                _complex_to_pair(self.h_dl_true) if self.h_dl_true is not None else None
            ),
            "h_dl_est": (
                _complex_to_pair(self.h_dl_est) if self.h_dl_est is not None else None
            ),
            "ul_sir_dB": self.ul_sir_dB,
            "dl_sir_dB": self.dl_sir_dB,
            "num_interfering_ues": self.num_interfering_ues,
            "ssb_rsrp_true_dBm": self.ssb_rsrp_true_dBm,
            "ssb_sinr_true_dB": self.ssb_sinr_true_dB,
            "noise_power_dBm": float(self.noise_power_dBm),
            "snr_dB": float(self.snr_dB),
            "sir_dB": None if self.sir_dB is None else float(self.sir_dB),
            "sinr_dB": float(self.sinr_dB),
            "ssb_rsrp_dBm": self.ssb_rsrp_dBm,
            "ssb_rsrq_dB": self.ssb_rsrq_dB,
            "ssb_sinr_dB": self.ssb_sinr_dB,
            "ssb_best_beam_idx": self.ssb_best_beam_idx,
            "ssb_pcis": self.ssb_pcis,
            "link": self.link,
            "channel_est_mode": self.channel_est_mode,
            "serving_cell_id": int(self.serving_cell_id),
            "ue_position": (
                _float_array_to_list(self.ue_position) if self.ue_position is not None else None
            ),
            "channel_model": self.channel_model,
            "tdd_pattern": self.tdd_pattern,
            "source": self.source,
            "sample_id": self.sample_id,
            "created_at": self.created_at.astimezone(timezone.utc).isoformat(),
            "meta": dict(self.meta),
        }
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChannelSample:
        """Inverse of :meth:`to_dict`."""

        def _maybe_complex(key: str) -> np.ndarray | None:
            payload = d.get(key)
            return None if payload is None else _pair_to_complex(payload)

        ue_pos_payload = d.get("ue_position")
        ue_pos = (
            None if ue_pos_payload is None else _list_to_float_array(ue_pos_payload, np.float64)
        )

        created = d["created_at"]
        if isinstance(created, str):
            created = datetime.fromisoformat(created)

        return cls(
            h_serving_true=_maybe_complex("h_serving_true"),
            h_serving_est=_maybe_complex("h_serving_est"),
            h_interferers=_maybe_complex("h_interferers"),
            interference_signal=_maybe_complex("interference_signal"),
            link_pairing=d.get("link_pairing", "single"),
            h_ul_true=_maybe_complex("h_ul_true"),
            h_ul_est=_maybe_complex("h_ul_est"),
            h_dl_true=_maybe_complex("h_dl_true"),
            h_dl_est=_maybe_complex("h_dl_est"),
            ul_sir_dB=d.get("ul_sir_dB"),
            dl_sir_dB=d.get("dl_sir_dB"),
            num_interfering_ues=d.get("num_interfering_ues"),
            ssb_rsrp_true_dBm=d.get("ssb_rsrp_true_dBm"),
            ssb_sinr_true_dB=d.get("ssb_sinr_true_dB"),
            noise_power_dBm=d["noise_power_dBm"],
            snr_dB=d["snr_dB"],
            sir_dB=d.get("sir_dB"),
            sinr_dB=d["sinr_dB"],
            ssb_rsrp_dBm=d.get("ssb_rsrp_dBm"),
            ssb_rsrq_dB=d.get("ssb_rsrq_dB"),
            ssb_sinr_dB=d.get("ssb_sinr_dB"),
            ssb_best_beam_idx=d.get("ssb_best_beam_idx"),
            ssb_pcis=d.get("ssb_pcis"),
            link=d["link"],
            channel_est_mode=d["channel_est_mode"],
            serving_cell_id=d["serving_cell_id"],
            ue_position=ue_pos,
            channel_model=d.get("channel_model"),
            tdd_pattern=d.get("tdd_pattern"),
            source=d["source"],
            sample_id=d["sample_id"],
            created_at=created,
            meta=d.get("meta", {}),
        )

    def to_parquet_row(self) -> dict[str, Any]:
        """Flatten scalar/meta fields for insertion into a parquet manifest.

        Array payloads are NOT inlined — writers are expected to dump them to
        a side artefact (``.pt``/``.npz``) and fill in ``array_path``.
        """
        T, RB, BS, UE = self.h_serving_true.shape
        k_minus_1 = 0 if self.h_interferers is None else self.h_interferers.shape[0]
        return {
            "sample_id": self.sample_id,
            "source": self.source,
            "link": self.link,
            "channel_est_mode": self.channel_est_mode,
            "serving_cell_id": int(self.serving_cell_id),
            "T": int(T),
            "RB": int(RB),
            "BS_ant": int(BS),
            "UE_ant": int(UE),
            "num_interferers": int(k_minus_1),
            "noise_power_dBm": float(self.noise_power_dBm),
            "snr_dB": float(self.snr_dB),
            "sir_dB": None if self.sir_dB is None else float(self.sir_dB),
            "sinr_dB": float(self.sinr_dB),
            "has_ssb_measurements": self.ssb_rsrp_dBm is not None,
            "has_interference_signal": self.interference_signal is not None,
            "link_pairing": self.link_pairing,
            "ul_sir_dB": self.ul_sir_dB,
            "dl_sir_dB": self.dl_sir_dB,
            "num_interfering_ues": self.num_interfering_ues,
            "channel_model": self.channel_model,
            "tdd_pattern": self.tdd_pattern,
            "ue_x": None if self.ue_position is None else float(self.ue_position[0]),
            "ue_y": None if self.ue_position is None else float(self.ue_position[1]),
            "ue_z": None if self.ue_position is None else float(self.ue_position[2]),
            "created_at": self.created_at.astimezone(timezone.utc).isoformat(),
            "array_path": self.meta.get("array_path"),
            "meta_json": self.meta,
        }

    @classmethod
    def from_pt_file(cls, path: Path) -> ChannelSample:
        """Load a pickled ``.pt`` file that stores a :meth:`to_dict` payload.

        The file may contain either:

        * the dict produced by :meth:`to_dict` directly, or
        * a raw ``ChannelSample`` instance (pickled), in which case it is
          re-validated.
        """
        path = Path(path)
        # Prefer torch.load so that tensor-containing legacy payloads still work,
        # but fall back to pickle if torch is not installed in the consumer env.
        try:
            import torch  # type: ignore

            payload = torch.load(path, map_location="cpu", weights_only=False)
        except ImportError:  # pragma: no cover - torch optional in lean envs
            with open(path, "rb") as f:
                payload = pickle.load(f)

        if isinstance(payload, ChannelSample):
            # Re-run validation by round-tripping through dict.
            return cls.from_dict(payload.to_dict())
        if isinstance(payload, dict):
            return cls.from_dict(payload)
        raise TypeError(f"{path}: expected ChannelSample or dict, got {type(payload).__name__}")


__all__ = ["ChannelSample", "LinkType", "LinkPairing", "ChannelEstMode", "SourceType"]
