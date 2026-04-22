"""Multi-cell QuaDRiGa MATLAB data source.

Consumes the output of ``matlab/main_multi.m`` which produces .mat files
containing ``Hf_multi`` of shape ``[no_ue, K, BsAnt, ue_ant, N_RB, no_ss]``
and a ``meta`` struct with topology information.

For each UE the serving cell is selected as the one with the strongest
average channel power. Interferer channels are the remaining K-1 cells.
Channel estimation is applied via
:func:`msg_embedding.channel_est.estimate_channel`.
"""

from __future__ import annotations

import contextlib
import glob
import os
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ..contract import ChannelSample
from .base import DataSource, register_source

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------

try:
    import scipy.io as _sio

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _sio = None  # type: ignore[assignment]
    _SCIPY_AVAILABLE = False

try:
    import hdf5storage as _hdf5

    _HDF5_AVAILABLE = True
except ImportError:  # pragma: no cover
    _hdf5 = None  # type: ignore[assignment]
    _HDF5_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dict_get(cfg: Any, key: str, default: Any) -> Any:
    """Small helper that works for DictConfig / dict / SimpleNamespace / None."""
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except Exception:  # pragma: no cover
            pass
    return getattr(cfg, key, default)


def _load_mat(path: str) -> dict[str, Any]:
    """Load a .mat file, trying v7 first then v7.3 (HDF5)."""
    try:
        return _sio.loadmat(path, simplify_cells=True)  # type: ignore[union-attr]
    except NotImplementedError as exc:
        if not _HDF5_AVAILABLE:
            raise ImportError(
                f"MAT file {path} is v7.3 (HDF5) but hdf5storage is not installed."
            ) from exc
        return _hdf5.loadmat(path)  # type: ignore[union-attr]


def _compute_power_db(h: np.ndarray) -> float:
    """Average power of a complex array in dB (relative to 1)."""
    p = float(np.mean(np.abs(h.ravel()) ** 2))
    if p <= 0 or not np.isfinite(p):
        return -50.0
    return float(10.0 * np.log10(p))


def _clamp(val: float, lo: float = -50.0, hi: float = 50.0) -> float:
    return float(np.clip(val, lo, hi))


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


@register_source
class QuadrigaMultiSource(DataSource):
    """Ingests multi-cell QuaDRiGa .mat files with serving + interferer cells.

    Supported config keys:

    ``mat_dir``
        Directory containing ``.mat`` files produced by ``matlab/main_multi.m``.
    ``num_samples``
        Maximum number of :class:`ChannelSample` to yield (0 = all available).
    ``channel_est_mode``
        ``'ideal'`` / ``'ls_linear'`` / ``'ls_mmse'``. Default ``'ideal'``.
    ``link``
        ``'UL'`` or ``'DL'``. Default ``'DL'``.
    ``noise_power_dBm``
        Noise power in dBm. Default -94.0.
    ``tx_power_dBm``
        Transmit power in dBm. Default 23.0.
    ``snr_dB``
        Optional override; if not set, estimated from channel + noise power.
    ``mat_pattern``
        Glob pattern for .mat files within ``mat_dir``. Default ``'*.mat'``.
    ``seed``
        RNG seed for reproducible noise injection. Default 42.
    """

    name = "quadriga_multi"

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def validate_config(self) -> None:
        cfg = self.config
        self.mat_dir: str = str(_dict_get(cfg, "mat_dir", ""))
        if not self.mat_dir:
            raise ValueError("QuadrigaMultiSource requires `mat_dir` in config.")
        if not os.path.isdir(self.mat_dir):
            raise FileNotFoundError(f"mat_dir does not exist: {self.mat_dir!r}")
        self.num_samples: int = int(_dict_get(cfg, "num_samples", 0))
        self.channel_est_mode: str = str(_dict_get(cfg, "channel_est_mode", "ideal"))
        if self.channel_est_mode not in {"ideal", "ls_linear", "ls_mmse"}:
            raise ValueError(f"Unknown channel_est_mode {self.channel_est_mode!r}.")
        self.link: str = str(_dict_get(cfg, "link", "DL")).upper()
        if self.link not in {"UL", "DL"}:
            raise ValueError(f"link must be 'UL' or 'DL', got {self.link!r}.")
        self.noise_power_dBm: float = float(_dict_get(cfg, "noise_power_dBm", -94.0))
        self.tx_power_dBm: float = float(_dict_get(cfg, "tx_power_dBm", 23.0))
        self._snr_override: float | None = _dict_get(cfg, "snr_dB", None)
        if self._snr_override is not None:
            self._snr_override = float(self._snr_override)
        self.mat_pattern: str = str(_dict_get(cfg, "mat_pattern", "*.mat"))
        self._seed: int = int(_dict_get(cfg, "seed", 42))

        # -- Channel model (informational for QuaDRiGa — model is baked in .mat) --
        self.channel_model_name: str = str(_dict_get(cfg, "channel_model", ""))
        self.tdd_pattern_name: str = str(_dict_get(cfg, "tdd_pattern", "DDDSU"))

        # -- SSB measurement --
        self.num_ssb_beams: int = int(_dict_get(cfg, "num_ssb_beams", 8))
        self.enable_ssb: bool = bool(_dict_get(cfg, "enable_ssb", True))

        # -- SRS hopping --
        self.srs_group_hopping: bool = bool(_dict_get(cfg, "srs_group_hopping", False))
        self.srs_sequence_hopping: bool = bool(_dict_get(cfg, "srs_sequence_hopping", False))
        self.srs_periodicity: int = int(_dict_get(cfg, "srs_periodicity", 10))
        self.srs_b_hop: int = int(_dict_get(cfg, "srs_b_hop", 3))

        if not _SCIPY_AVAILABLE:
            raise ImportError("scipy is required for QuadrigaMultiSource (scipy.io.loadmat).")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _list_mat_files(self) -> list[str]:
        pattern = os.path.join(self.mat_dir, self.mat_pattern)
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No .mat files matching {pattern!r} in {self.mat_dir!r}")
        return files

    def _estimate_snr(self, h_serving: np.ndarray) -> float:
        """Rough SNR estimate from channel power and configured noise floor."""
        if self._snr_override is not None:
            return _clamp(self._snr_override)
        p_h = float(np.mean(np.abs(h_serving.ravel()) ** 2))
        if p_h <= 0 or not np.isfinite(p_h):
            return 0.0
        sig_dBm = self.tx_power_dBm + 10.0 * np.log10(p_h)
        return _clamp(float(sig_dBm - self.noise_power_dBm))

    def _estimate_sir(self, h_serving: np.ndarray, h_interferers: np.ndarray) -> float:
        """SIR = serving power / total interferer power (dB)."""
        p_s = float(np.mean(np.abs(h_serving.ravel()) ** 2))
        p_i = float(np.mean(np.abs(h_interferers.ravel()) ** 2))
        if p_i <= 0 or not np.isfinite(p_i):
            return 50.0
        if p_s <= 0 or not np.isfinite(p_s):
            return -50.0
        return _clamp(float(10.0 * np.log10(p_s / p_i)))

    @staticmethod
    def _sinr_from_snr_sir(snr_dB: float, sir_dB: float | None) -> float:
        """SINR = 1 / (1/SNR_lin + 1/SIR_lin) in dB."""
        snr_lin = 10.0 ** (snr_dB / 10.0)
        if sir_dB is None:
            return _clamp(snr_dB)
        sir_lin = 10.0 ** (sir_dB / 10.0)
        sinr_lin = 1.0 / (1.0 / snr_lin + 1.0 / sir_lin)
        return _clamp(float(10.0 * np.log10(sinr_lin)))

    def _run_channel_est(self, h_true: np.ndarray, snr_dB: float) -> np.ndarray:
        """Apply channel estimation to produce h_est.

        ``h_true`` has shape ``[T, RB, BS_ant, UE_ant]`` complex64.
        """
        if self.channel_est_mode == "ideal":
            return h_true.copy()

        from msg_embedding.channel_est.pipeline import estimate_channel

        T, RB, BS, UE = h_true.shape
        rs_freq = np.arange(RB, dtype=np.int64)
        rs_time = np.arange(T, dtype=np.int64)
        pilots = np.ones(RB * T, dtype=np.complex64)

        # Flatten spatial dims for the estimator.
        h_grid = h_true.transpose(1, 0, 2, 3).reshape(RB, T, BS * UE)
        Y = h_grid.reshape(-1, BS * UE)

        est = estimate_channel(
            Y_rs=Y,
            X_rs=pilots,
            rs_positions_freq=rs_freq,
            rs_positions_time=rs_time,
            N_sc=RB,
            N_sym=T,
            mode=self.channel_est_mode,
            h_true=h_grid,
            pdp_prior={"tau_rms": 100e-9, "delta_f": 3e3},
            snr_db=snr_dB,
            dtype="complex64",
        )
        return est.reshape(RB, T, BS, UE).transpose(1, 0, 2, 3).astype(np.complex64)

    def _pick_serving_cell(self, hf_ue: np.ndarray) -> int:
        """Select serving cell as the one with strongest average power.

        ``hf_ue`` has shape ``[K, BsAnt, ue_ant, N_RB, no_ss]``.
        """
        K = hf_ue.shape[0]
        if K == 1:
            return 0
        powers = np.array([float(np.mean(np.abs(hf_ue[k]) ** 2)) for k in range(K)])
        return int(np.argmax(powers))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def iter_samples(self) -> Iterator[ChannelSample]:
        mat_files = self._list_mat_files()
        np.random.default_rng(self._seed)
        yielded = 0

        for mat_path in mat_files:
            data = _load_mat(mat_path)

            # Hf_multi: [no_ue, K, BsAnt, ue_ant, N_RB, no_ss]
            Hf_multi = data["Hf_multi"]
            if Hf_multi.dtype != np.complex64:
                Hf_multi = Hf_multi.astype(np.complex64)

            # scipy.io with simplify_cells=True squeezes singleton dims.
            # Restore expected 6D shape if needed.
            if Hf_multi.ndim == 5:
                # K=1 was squeezed out — restore it
                Hf_multi = Hf_multi[:, np.newaxis, ...]
            elif Hf_multi.ndim == 4:
                # Both no_ue=1 and K=1 squeezed
                Hf_multi = Hf_multi[np.newaxis, np.newaxis, ...]

            # Metadata
            meta_raw = data.get("meta", {})
            if not isinstance(meta_raw, dict):
                # scipy.io may return struct as nested dict
                meta_raw = {}

            no_ue_file = Hf_multi.shape[0]
            K = Hf_multi.shape[1]

            # Extract positions if available
            bs_positions = meta_raw.get("bs_positions")
            ue_positions = meta_raw.get("ue_positions")
            meta_serving_ids = meta_raw.get("serving_cell_ids")
            # scipy may squeeze [N] to scalar when N=1
            if meta_serving_ids is not None and np.ndim(meta_serving_ids) == 0:
                meta_serving_ids = np.atleast_1d(meta_serving_ids)

            for i_ue in range(no_ue_file):
                if 0 < self.num_samples <= yielded:
                    return

                hf_ue = Hf_multi[i_ue]  # [K, BsAnt, ue_ant, N_RB, no_ss]

                # Pick serving cell (use metadata if available, else compute).
                if meta_serving_ids is not None:
                    serving_id = int(meta_serving_ids[i_ue]) - 1  # MATLAB 1-based
                    if serving_id < 0 or serving_id >= K:
                        serving_id = self._pick_serving_cell(hf_ue)
                else:
                    serving_id = self._pick_serving_cell(hf_ue)

                # Extract serving channel: [BsAnt, ue_ant, N_RB, no_ss]
                h_serving_raw = hf_ue[serving_id]

                # Transpose to contract shape: [T, RB, BS_ant, UE_ant]
                # From [BsAnt, ue_ant, N_RB, no_ss] -> [no_ss, N_RB, BsAnt, ue_ant]
                h_serving_true = np.ascontiguousarray(
                    h_serving_raw.transpose(3, 2, 0, 1).astype(np.complex64)
                )

                # Extract interferers: [K-1, BsAnt, ue_ant, N_RB, no_ss]
                if K > 1:
                    other_idx = [k for k in range(K) if k != serving_id]
                    h_int_raw = hf_ue[other_idx]  # [K-1, BsAnt, ue_ant, N_RB, no_ss]
                    # -> [K-1, no_ss, N_RB, BsAnt, ue_ant]
                    h_interferers: np.ndarray | None = np.ascontiguousarray(
                        h_int_raw.transpose(0, 4, 3, 1, 2).astype(np.complex64)
                    )
                else:
                    h_interferers = None

                # SNR / SIR / SINR
                snr_dB = self._estimate_snr(h_serving_true)
                sir_dB: float | None = None
                if h_interferers is not None:
                    sir_dB = self._estimate_sir(h_serving_true, h_interferers)
                sinr_dB = self._sinr_from_snr_sir(snr_dB, sir_dB)

                # Channel estimation
                h_serving_est = self._run_channel_est(h_serving_true, snr_dB)

                # UE position
                ue_pos: np.ndarray | None = None
                if ue_positions is not None:
                    try:
                        # ue_positions from MATLAB is [3 x no_ue]
                        raw_pos = np.asarray(ue_positions)
                        if raw_pos.ndim == 2:
                            if raw_pos.shape[0] == 3:
                                ue_pos = raw_pos[:, i_ue].astype(np.float64)
                            elif raw_pos.shape[1] == 3:
                                ue_pos = raw_pos[i_ue, :].astype(np.float64)
                        if ue_pos is not None and ue_pos.shape != (3,):
                            ue_pos = None
                    except (IndexError, TypeError):
                        ue_pos = None

                # SSB measurement
                ssb_rsrp = None
                ssb_rsrq = None
                ssb_sinr = None
                ssb_best_beam = None
                ssb_pcis_list = None
                if self.enable_ssb:
                    try:
                        from msg_embedding.phy_sim.ssb_measurement import SSBMeasurement

                        pci_list = list(range(K))
                        # Build h_per_cell list: [h_cell0, h_cell1, ...] each [T, RB, BS, UE]
                        h_cells = []
                        for k_idx in range(K):
                            if k_idx == serving_id:
                                h_cells.append(h_serving_true)
                            else:
                                # Find this cell in the interferer array
                                other_idx_list = [kk for kk in range(K) if kk != serving_id]
                                int_pos = other_idx_list.index(k_idx)
                                h_cells.append(
                                    h_interferers[int_pos]
                                    if h_interferers is not None
                                    else h_serving_true * 0.1
                                )

                        noise_lin = 10.0 ** (self.noise_power_dBm / 10.0) * 1e-3
                        meas = SSBMeasurement(
                            num_beams=self.num_ssb_beams,
                            num_bs_ant=h_serving_true.shape[2],
                        )
                        result = meas.measure(
                            h_per_cell=h_cells, pcis=pci_list, noise_power_lin=noise_lin
                        )
                        ssb_rsrp = result.rsrp_dBm.tolist()
                        ssb_rsrq = result.rsrq_dB.tolist()
                        ssb_sinr = result.ss_sinr_dB.tolist()
                        ssb_best_beam = result.best_beam_idx.tolist()
                        ssb_pcis_list = pci_list
                    except Exception:
                        pass

                # Build metadata dict
                sample_meta: dict[str, Any] = {
                    "num_cells": int(K),
                    "isd_m": meta_raw.get("isd_m", 500),
                    "scenario": meta_raw.get("scenario", "unknown"),
                    "carrier_freq_hz": meta_raw.get("carrier_freq_hz", 3e9),
                    "ue_speed_kmh": meta_raw.get("ue_speed_kmh", 0),
                    "mat_file": os.path.basename(mat_path),
                    "ue_index_in_file": i_ue,
                    "tx_power_dBm": self.tx_power_dBm,
                    "channel_model": self.channel_model_name,
                    "tdd_pattern": self.tdd_pattern_name,
                    "srs_group_hopping": self.srs_group_hopping,
                    "srs_sequence_hopping": self.srs_sequence_hopping,
                    "srs_periodicity": self.srs_periodicity,
                    "srs_b_hop": self.srs_b_hop,
                    "num_ssb_beams": self.num_ssb_beams,
                    "mock": False,
                }
                if bs_positions is not None:
                    with contextlib.suppress(Exception):
                        sample_meta["bs_positions"] = np.asarray(bs_positions).tolist()

                yield ChannelSample(
                    h_serving_true=h_serving_true,
                    h_serving_est=h_serving_est,
                    h_interferers=h_interferers,
                    noise_power_dBm=self.noise_power_dBm,
                    snr_dB=snr_dB,
                    sir_dB=sir_dB,
                    sinr_dB=sinr_dB,
                    link=self.link,  # type: ignore[arg-type]
                    channel_est_mode=self.channel_est_mode,  # type: ignore[arg-type]
                    serving_cell_id=serving_id,
                    ue_position=ue_pos,
                    source="quadriga_multi",
                    sample_id=str(uuid.uuid4()),
                    created_at=datetime.now(timezone.utc),
                    meta=sample_meta,
                    channel_model=self.channel_model_name or None,
                    tdd_pattern=self.tdd_pattern_name or None,
                    ssb_rsrp_dBm=ssb_rsrp,
                    ssb_rsrq_dB=ssb_rsrq,
                    ssb_sinr_dB=ssb_sinr,
                    ssb_best_beam_idx=ssb_best_beam,
                    ssb_pcis=ssb_pcis_list,
                )
                yielded += 1

    def describe(self) -> dict[str, Any]:
        mat_files = []
        with contextlib.suppress(FileNotFoundError):
            mat_files = self._list_mat_files()

        return {
            "source": "quadriga_multi",
            "mat_dir": self.mat_dir,
            "mat_files_found": len(mat_files),
            "num_samples": self.num_samples,
            "channel_est_mode": self.channel_est_mode,
            "link": self.link,
            "noise_power_dBm": self.noise_power_dBm,
            "tx_power_dBm": self.tx_power_dBm,
            "channel_model": self.channel_model_name,
            "tdd_pattern": self.tdd_pattern_name,
            "num_ssb_beams": self.num_ssb_beams,
            "enable_ssb": self.enable_ssb,
            "srs_group_hopping": self.srs_group_hopping,
            "srs_sequence_hopping": self.srs_sequence_hopping,
            "srs_periodicity": self.srs_periodicity,
            "srs_b_hop": self.srs_b_hop,
        }


__all__ = ["QuadrigaMultiSource"]
