"""QuaDRiGa real MATLAB data source — generates .mat via MATLAB subprocess, then yields ChannelSample.

Combines MATLAB channel generation (run_quadriga_real.py logic) with
.mat → ChannelSample conversion in a single DataSource, so the platform
can trigger end-to-end QuaDRiGa data collection.

Requires: MATLAB installed locally. Path configurable via ``matlab_exe``
config key or ``MSG_MATLAB_EXE`` environment variable.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..contract import ChannelEstMode, ChannelSample
from .base import DataSource, register_source

try:
    import scipy.io as _sio

    _SCIPY_AVAILABLE = True
except ImportError:
    _sio = None  # type: ignore[assignment]
    _SCIPY_AVAILABLE = False

try:
    import h5py as _h5py

    _H5PY_AVAILABLE = True
except ImportError:
    _h5py = None  # type: ignore[assignment]
    _H5PY_AVAILABLE = False


def _find_matlab() -> str:
    """Locate MATLAB executable from config or environment."""
    env_path = os.environ.get("MSG_MATLAB_EXE", "")
    if env_path and os.path.isfile(env_path):
        return env_path
    defaults = [
        r"D:\matlab\bin\matlab.exe",
        r"C:\Program Files\MATLAB\R2024b\bin\matlab.exe",
        r"C:\Program Files\MATLAB\R2024a\bin\matlab.exe",
        r"C:\Program Files\MATLAB\R2023b\bin\matlab.exe",
        r"C:\Program Files\MATLAB\R2023a\bin\matlab.exe",
        "/usr/local/MATLAB/R2024b/bin/matlab",
        "/usr/local/MATLAB/R2023b/bin/matlab",
        "/Applications/MATLAB_R2024b.app/bin/matlab",
    ]
    for p in defaults:
        if os.path.isfile(p):
            return p
    return "matlab"


def _load_mat(path: str) -> dict[str, Any]:
    try:
        return _sio.loadmat(path)  # type: ignore[union-attr]
    except Exception:
        if not _H5PY_AVAILABLE:
            raise
        d: dict[str, Any] = {}
        with _h5py.File(path, "r") as f:  # type: ignore[union-attr]
            for key in [
                "Hf_serving_est",
                "Hf_serving_ideal",
                "rsrp_per_cell",
                "snr_dB",
                "sir_dB",
                "sinr_dB",
            ]:
                if key in f:
                    d[key] = np.array(f[key]).T
            if "meta" in f:
                meta_grp = f["meta"]
                d["meta"] = {}
                for k in meta_grp:
                    d["meta"][k] = np.array(meta_grp[k]).T
        return d


def _meta_get(m: Any, key: str, default: Any = None) -> Any:
    if isinstance(m, dict):
        val = m.get(key, default)
    else:
        try:
            val = m[key]
        except (KeyError, ValueError, IndexError):
            return default
    if isinstance(val, np.ndarray) and val.size == 1:
        return val.item()
    return val


@register_source
class QuadrigaRealSource(DataSource):
    """End-to-end QuaDRiGa MATLAB source: generate .mat → yield ChannelSample.

    Config keys:
        ``matlab_exe``  MATLAB executable path (or set MSG_MATLAB_EXE env var)
        ``num_samples``  Total samples to generate (default 1000)
        ``ues_per_shard``  UEs per MATLAB shard (default 50)
        ``num_cells``  Number of cells (default 7)
        ``num_snapshots``  Time snapshots (default 14)
        ``carrier_freq_hz``  Carrier frequency (default 3.5e9)
        ``scenario``  3GPP scenario (default 3GPP_38.901_UMa_NLOS)
        ``isd_m``  Inter-site distance (default 500)
        ``seed``  Base seed (default 42)
        ``mat_dir``  Directory for .mat output (auto-created)
        ``skip_generation``  If true, only read existing .mat files
    """

    name = "quadriga_real"

    def validate_config(self) -> None:
        cfg = self.config or {}
        if not _SCIPY_AVAILABLE:
            raise ImportError("scipy is required for QuadrigaRealSource.")

        self.matlab_exe = str(cfg.get("matlab_exe", "")) or _find_matlab()
        self.num_samples = int(cfg.get("num_samples", 1000))
        self.ues_per_shard = int(cfg.get("ues_per_shard", 50))
        self.num_shards = (self.num_samples + self.ues_per_shard - 1) // self.ues_per_shard
        self.num_cells = int(cfg.get("num_cells", 7))
        self.num_snapshots = int(cfg.get("num_snapshots", 14))
        self.carrier_freq_hz = float(cfg.get("carrier_freq_hz", 3.5e9))
        self.scenario = str(cfg.get("scenario", "3GPP_38.901_UMa_NLOS"))
        self.isd_m = int(cfg.get("isd_m", 500))
        self.seed = int(cfg.get("seed", 42))
        self.skip_generation = bool(cfg.get("skip_generation", False))

        # -- Link direction & estimation mode (F-001: was previously ignored) --
        raw_link = str(cfg.get("link", "UL")).upper()
        if raw_link == "BOTH":
            self.link = "BOTH"
        elif raw_link == "DL":
            self.link = "DL"
        else:
            self.link = "UL"
        raw_est = str(cfg.get("channel_est_mode", "ls_linear")).lower()
        self.channel_est_mode: str = raw_est if raw_est in ("ideal", "ls_linear", "ls_mmse") else "ls_linear"
        self.pilot_type_ul: str = str(cfg.get("pilot_type_ul", cfg.get("pilot_type", "srs_zc")))
        self.pilot_type_dl: str = str(cfg.get("pilot_type_dl", cfg.get("pilot_type", "csi_rs_gold")))
        self.tdd_pattern: str = str(cfg.get("tdd_pattern", "DDDSU"))
        self.num_interfering_ues: int = int(cfg.get("num_interfering_ues", 3))

        _project_root = Path(__file__).resolve().parents[4]
        repo = Path(cfg.get("repo_root", os.environ.get("MSG_REPO_ROOT", str(_project_root))))
        _local_matlab = _project_root / "matlab"
        if _local_matlab.is_dir() and (_local_matlab / "main_multi.m").is_file():
            self.matlab_dir = _local_matlab
        else:
            self.matlab_dir = repo / "matlab"
        self.mat_dir = Path(str(cfg.get("mat_dir", str(_project_root / "artifacts" / "quadriga_real_mat"))))

        self._matlab_config = {
            "num_cells": self.num_cells,
            "num_ues": self.ues_per_shard,
            "num_snapshots": self.num_snapshots,
            "carrier_freq_hz": self.carrier_freq_hz,
            "scenario": self.scenario,
            "isd_m": self.isd_m,
            "tx_height_m": int(cfg.get("tx_height_m", 25)),
            "rx_height_m": float(cfg.get("rx_height_m", 1.5)),
            "cell_radius_m": int(cfg.get("cell_radius_m", 250)),
            "ue_speed_kmh": int(cfg.get("ue_speed_kmh", 3)),
            "mobility_mode": str(cfg.get("mobility_mode", "linear")),
            "bs_ant_v": int(cfg.get("bs_ant_v", 4)),
            "bs_ant_h": int(cfg.get("bs_ant_h", 8)),
            "ue_ant_v": int(cfg.get("ue_ant_v", 1)),
            "ue_ant_h": int(cfg.get("ue_ant_h", 2)),
            "n_rb": int(cfg.get("n_rb", 273)),
            "sc_inter": int(cfg.get("sc_inter", 30000)),
            "output_dir": str(self.mat_dir),
        }

    def _generate_shard(self, shard_id: int) -> bool:
        out_file = self.mat_dir / f"QDRG_real_shard{shard_id:04d}.mat"
        if out_file.exists():
            return True

        cfg = dict(self._matlab_config)
        cfg["seed"] = self.seed + shard_id
        cfg["shard_id"] = shard_id

        config_path = self.matlab_dir / f"_config_shard{shard_id:04d}.json"
        config_path.write_text(json.dumps(cfg, indent=2))

        cmd = [
            self.matlab_exe,
            "-nosplash",
            "-nodesktop",
            "-batch",
            f"main_multi('{config_path.as_posix()}')",
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.matlab_dir),
                capture_output=True,
                text=True,
                timeout=7200,
            )
            config_path.unlink(missing_ok=True)
            return result.returncode == 0
        except Exception:
            config_path.unlink(missing_ok=True)
            return False

    def _generate_all(self) -> None:
        self.mat_dir.mkdir(parents=True, exist_ok=True)
        for i in range(self.num_shards):
            pct = i / self.num_shards * 50
            print(f"[progress] pct={pct:.1f} step=MATLAB shard {i+1}/{self.num_shards}", flush=True)
            ok = self._generate_shard(i)
            if not ok:
                print(f"WARNING: shard {i} failed", flush=True)

    def _iter_from_mat(self) -> Iterator[ChannelSample]:
        mat_files = sorted(self.mat_dir.glob("QDRG_real_shard*.mat"))
        if not mat_files:
            raise FileNotFoundError(f"No .mat files in {self.mat_dir}")

        n_re = 273 * 12
        yielded = 0

        for mat_path in mat_files:
            if yielded >= self.num_samples:
                return

            d = _load_mat(str(mat_path))
            shard_num = int(mat_path.stem.split("shard")[-1])

            Hf_est = d["Hf_serving_est"]
            Hf_ideal = d["Hf_serving_ideal"]
            mat_snr = np.asarray(d["snr_dB"]).flatten()
            mat_sir = np.asarray(d["sir_dB"]).flatten()
            mat_sinr = np.asarray(d["sinr_dB"]).flatten()
            mat_rsrp = np.asarray(d["rsrp_per_cell"])

            meta = d.get("meta", {})
            if isinstance(meta, np.ndarray) and meta.dtype.names:
                meta = meta[0, 0]

            no_ue = Hf_est.shape[0]

            ue_pos_all = _meta_get(meta, "ue_positions", np.zeros((3, no_ue)))
            K = int(_meta_get(meta, "num_cells", 7))
            noise_dBm = float(_meta_get(meta, "noise_dBm", -89.1))
            ptx_dBm = float(_meta_get(meta, "Ptx_BS_dBm", _meta_get(meta, "Ptx_dBm", 46.0)))
            ptx_per_re = ptx_dBm - 10.0 * np.log10(n_re)

            for i_ue in range(no_ue):
                if yielded >= self.num_samples:
                    return

                # MATLAB outputs UL channels: [BsAnt, ue_ant, N_RB, no_ss]
                # Transpose to platform shape: [T, RB, BS_ant, UE_ant]
                h_ul_ideal = np.transpose(Hf_ideal[i_ue], (3, 2, 0, 1))
                h_ul_est = np.transpose(Hf_est[i_ue], (3, 2, 0, 1))

                snr_dB = float(np.nan_to_num(np.clip(mat_snr[i_ue], -50, 50), nan=50.0))
                sir_dB = float(np.nan_to_num(np.clip(mat_sir[i_ue], -50, 50), nan=50.0))
                sinr_dB = float(np.nan_to_num(np.clip(mat_sinr[i_ue], -50, 50), nan=50.0))

                raw_gain = np.mean(np.abs(h_ul_ideal) ** 2)
                if raw_gain > 0:
                    scale = np.sqrt(raw_gain)
                    h_ul_ideal = h_ul_ideal / scale
                    h_ul_est = h_ul_est / scale

                try:
                    ue_p = np.asarray(ue_pos_all)[:, i_ue].flatten()
                except (IndexError, TypeError):
                    ue_p = np.array([0.0, 0.0, 1.5])

                rsrp_row = mat_rsrp[i_ue] if i_ue < mat_rsrp.shape[0] else np.zeros(K)
                ssb_rsrp_list = [
                    float(ptx_per_re + 10 * np.log10(max(float(g), 1e-30))) for g in rsrp_row
                ]

                sample_meta = {
                    "num_cells": K,
                    "isd_m": self.isd_m,
                    "scenario": self.scenario,
                    "carrier_freq_hz": self.carrier_freq_hz,
                    "ue_speed_kmh": self._matlab_config.get("ue_speed_kmh", 0),
                    "mobility_mode": self._matlab_config.get("mobility_mode", "linear"),
                    "shard": shard_num,
                    "ue_idx": i_ue,
                }

                # -- Respect channel_est_mode: ideal means no estimation error --
                if self.channel_est_mode == "ideal":
                    h_ul_est = h_ul_ideal.copy()

                # -- Direction handling via TDD reciprocity --
                # MATLAB only produces UL (SRS) channels. For DL/BOTH we derive
                # via H_DL = conj(H_UL^T), consistent with internal_sim/sionna_rt.
                est_mode_out: ChannelEstMode = "ls_linear"
                if self.channel_est_mode in ("ideal", "ls_linear", "ls_mmse"):
                    est_mode_out = self.channel_est_mode  # type: ignore[assignment]

                if self.link == "BOTH":
                    # Paired mode: DL in h_serving + paired UL/DL fields
                    h_dl_ideal = np.conj(h_ul_ideal.transpose(0, 1, 3, 2))
                    h_dl_est = np.conj(h_ul_est.transpose(0, 1, 3, 2))
                    yield ChannelSample(
                        h_serving_true=h_dl_ideal.astype(np.complex64),
                        h_serving_est=h_dl_est.astype(np.complex64),
                        h_interferers=None,
                        noise_power_dBm=noise_dBm,
                        snr_dB=snr_dB,
                        sir_dB=sir_dB,
                        sinr_dB=sinr_dB,
                        ssb_rsrp_dBm=ssb_rsrp_list,
                        link="DL",
                        channel_est_mode=est_mode_out,
                        serving_cell_id=0,
                        ue_position=ue_p,
                        tdd_pattern=self.tdd_pattern,
                        link_pairing="paired",
                        h_ul_true=h_ul_ideal.astype(np.complex64),
                        h_ul_est=h_ul_est.astype(np.complex64),
                        h_dl_true=h_dl_ideal.astype(np.complex64),
                        h_dl_est=h_dl_est.astype(np.complex64),
                        source="quadriga_real",
                        sample_id=str(uuid.uuid4()),
                        created_at=datetime.now(timezone.utc),
                        meta=sample_meta,
                    )
                elif self.link == "DL":
                    # TDD reciprocity: H_DL = conj(H_UL^T)
                    h_dl_ideal = np.conj(h_ul_ideal.transpose(0, 1, 3, 2))
                    h_dl_est = np.conj(h_ul_est.transpose(0, 1, 3, 2))
                    yield ChannelSample(
                        h_serving_true=h_dl_ideal.astype(np.complex64),
                        h_serving_est=h_dl_est.astype(np.complex64),
                        h_interferers=None,
                        noise_power_dBm=noise_dBm,
                        snr_dB=snr_dB,
                        sir_dB=sir_dB,
                        sinr_dB=sinr_dB,
                        ssb_rsrp_dBm=ssb_rsrp_list,
                        link="DL",
                        channel_est_mode=est_mode_out,
                        serving_cell_id=0,
                        ue_position=ue_p,
                        tdd_pattern=self.tdd_pattern,
                        source="quadriga_real",
                        sample_id=str(uuid.uuid4()),
                        created_at=datetime.now(timezone.utc),
                        meta=sample_meta,
                    )
                else:
                    yield ChannelSample(
                        h_serving_true=h_ul_ideal.astype(np.complex64),
                        h_serving_est=h_ul_est.astype(np.complex64),
                        h_interferers=None,
                        noise_power_dBm=noise_dBm,
                        snr_dB=snr_dB,
                        sir_dB=sir_dB,
                        sinr_dB=sinr_dB,
                        ssb_rsrp_dBm=ssb_rsrp_list,
                        link="UL",
                        channel_est_mode=est_mode_out,
                        serving_cell_id=0,
                        ue_position=ue_p,
                        tdd_pattern=self.tdd_pattern,
                        source="quadriga_real",
                        sample_id=str(uuid.uuid4()),
                        created_at=datetime.now(timezone.utc),
                        meta=sample_meta,
                    )
                yielded += 1

            del Hf_est, Hf_ideal, d
            gc.collect()

    def iter_samples(self) -> Iterator[ChannelSample]:
        if not self.skip_generation:
            existing = (
                list(self.mat_dir.glob("QDRG_real_shard*.mat")) if self.mat_dir.exists() else []
            )
            if len(existing) < self.num_shards:
                if not os.path.isfile(self.matlab_exe):
                    raise RuntimeError(
                        f"MATLAB not found at '{self.matlab_exe}'. "
                        f"Set MSG_MATLAB_EXE environment variable or pass matlab_exe= in config. "
                        f"On Windows: set MSG_MATLAB_EXE=C:\\Program Files\\MATLAB\\R2024b\\bin\\matlab.exe"
                    )
                self._generate_all()

        yield from self._iter_from_mat()

    def describe(self) -> dict[str, Any]:
        existing = list(self.mat_dir.glob("QDRG_real_shard*.mat")) if self.mat_dir.exists() else []
        return {
            "source": "quadriga_real",
            "matlab_exe": self.matlab_exe,
            "matlab_found": os.path.isfile(self.matlab_exe),
            "num_samples": self.num_samples,
            "num_shards": self.num_shards,
            "existing_shards": len(existing),
            "mat_dir": str(self.mat_dir),
            "scenario": self.scenario,
            "num_cells": self.num_cells,
            "carrier_freq_hz": self.carrier_freq_hz,
            "link": self.link,
            "channel_est_mode": self.channel_est_mode,
            "pilot_type_ul": self.pilot_type_ul,
            "pilot_type_dl": self.pilot_type_dl,
            "tdd_pattern": self.tdd_pattern,
            "num_interfering_ues": self.num_interfering_ues,
        }


__all__ = ["QuadrigaRealSource"]
