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
import math
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

        # -- Bandwidth / SCS / RB (3GPP TS 38.101 standard table) -------------
        self.bandwidth_hz: float = float(cfg.get("bandwidth_hz", 100e6))
        self.subcarrier_spacing: float = float(cfg.get("subcarrier_spacing", 30e3))

        from msg_embedding.phy_sim.nr_rb_table import nr_rb_lookup

        _cfg_num_rb = cfg.get("n_rb", cfg.get("num_rb", None))
        if _cfg_num_rb is not None:
            self._num_rb: int = int(_cfg_num_rb)
        else:
            self._num_rb = nr_rb_lookup(self.bandwidth_hz, self.subcarrier_spacing)

        # -- Link direction & estimation mode ----------------------------------
        raw_link = str(cfg.get("link", "UL")).upper()
        if raw_link == "BOTH":
            self.link = "BOTH"
        elif raw_link == "DL":
            self.link = "DL"
        else:
            self.link = "UL"
        raw_est = str(cfg.get("channel_est_mode", "ls_linear")).lower()
        self.channel_est_mode: str = raw_est if raw_est in ("ideal", "ls_linear", "ls_mmse", "ls_hop_concat") else "ls_linear"
        self.pilot_type_ul: str = str(cfg.get("pilot_type_ul", cfg.get("pilot_type", "srs_zc")))
        self.pilot_type_dl: str = str(cfg.get("pilot_type_dl", cfg.get("pilot_type", "csi_rs_gold")))
        self.num_interfering_ues: int = int(cfg.get("num_interfering_ues", 3))

        # -- TDD slot pattern --------------------------------------------------
        from msg_embedding.phy_sim.tdd_config import get_tdd_pattern

        self.tdd_pattern_name: str = str(cfg.get("tdd_pattern", "DDDSU"))
        self._tdd_pattern = get_tdd_pattern(self.tdd_pattern_name)

        # -- SRS configuration (3GPP TS 38.211 §6.4.1.4) ----------------------
        self.srs_group_hopping: bool = bool(cfg.get("srs_group_hopping", False))
        self.srs_sequence_hopping: bool = bool(cfg.get("srs_sequence_hopping", False))
        if self.srs_group_hopping and self.srs_sequence_hopping:
            raise ValueError("srs_group_hopping and srs_sequence_hopping are mutually exclusive")
        self.srs_periodicity: int = int(cfg.get("srs_periodicity", 10))
        from msg_embedding.ref_signals.srs import SRS_PERIODICITY_TABLE

        if self.srs_periodicity not in SRS_PERIODICITY_TABLE:
            raise ValueError(
                f"srs_periodicity={self.srs_periodicity} not in "
                f"3GPP TS 38.211 Table 6.4.1.4.4-1: {SRS_PERIODICITY_TABLE}"
            )
        self.srs_comb: int = int(cfg.get("srs_comb", 2))
        if self.srs_comb not in (2, 4, 8):
            raise ValueError(f"srs_comb (K_TC) must be in {{2, 4, 8}}, got {self.srs_comb}")
        self.srs_b_hop: int = int(cfg.get("srs_b_hop", 0))
        self.srs_c_srs: int = int(cfg.get("srs_c_srs", 3))
        self.srs_b_srs: int = int(cfg.get("srs_b_srs", 1))
        self.srs_n_rrc: int = int(cfg.get("srs_n_rrc", 0))
        from msg_embedding.ref_signals.srs import SRSResourceConfig

        self._srs_resource_cfg = SRSResourceConfig(
            C_SRS=self.srs_c_srs,
            B_SRS=self.srs_b_srs,
            K_TC=self.srs_comb,
            n_RRC=self.srs_n_rrc,
            b_hop=self.srs_b_hop,
            n_SRS_ID=0,
            T_SRS=self.srs_periodicity,
            T_offset=0,
            group_hopping=self.srs_group_hopping,
            sequence_hopping=self.srs_sequence_hopping,
        )

        _project_root = Path(__file__).resolve().parents[4]
        repo = Path(cfg.get("repo_root", os.environ.get("MSG_REPO_ROOT", str(_project_root))))
        _local_matlab = _project_root / "matlab"
        if _local_matlab.is_dir() and (_local_matlab / "main_multi.m").is_file():
            self.matlab_dir = _local_matlab
        else:
            self.matlab_dir = repo / "matlab"
        self.mat_dir = Path(str(cfg.get("mat_dir", str(_project_root / "artifacts" / "quadriga_real_mat"))))

        # -- Panel array (dual-polarization) -----------------------------------
        _bs_panel_raw = cfg.get("bs_panel", None)
        _ue_panel_raw = cfg.get("ue_panel", None)
        if _bs_panel_raw is not None:
            _bp = [int(x) for x in _bs_panel_raw]
            self.bs_panel_h: int = _bp[0]
            self.bs_panel_v: int = _bp[1]
            self.bs_panel_p: int = _bp[2]
        else:
            self.bs_panel_h = int(cfg.get("bs_ant_h", 8))
            self.bs_panel_v = int(cfg.get("bs_ant_v", 4))
            self.bs_panel_p = 1
        if _ue_panel_raw is not None:
            _up = [int(x) for x in _ue_panel_raw]
            self.ue_panel_h: int = _up[0]
            self.ue_panel_v: int = _up[1]
            self.ue_panel_p: int = _up[2]
        else:
            self.ue_panel_h = int(cfg.get("ue_ant_h", 2))
            self.ue_panel_v = int(cfg.get("ue_ant_v", 1))
            self.ue_panel_p = 1
        self.xpd_db: float = float(cfg.get("xpd_db", 8.0))
        self.ue_tx_power_dbm: float = float(cfg.get("ue_tx_power_dbm", 23.0))

        # -- Additional parameters for full parity with internal_sim --------
        self.topology_layout: str = str(cfg.get("topology_layout", "hexagonal"))
        self.sectors_per_site: int = int(cfg.get("sectors_per_site", 1))
        self.track_offset_m: float = float(cfg.get("track_offset_m", 80.0))
        self.hypercell_size: int = int(cfg.get("hypercell_size", 1))
        self.ue_distribution: str = str(cfg.get("ue_distribution", "uniform"))
        self.ue_height_m: float = float(cfg.get("ue_height_m", cfg.get("rx_height_m", 1.5)))
        self.mobility_mode: str = str(cfg.get("mobility_mode", "linear"))
        self.ue_speed_kmh: float = float(cfg.get("ue_speed_kmh", 3.0))
        self.sample_interval_s: float = float(cfg.get("sample_interval_s", 0.5e-3))
        self.train_penetration_loss_db: float = float(cfg.get("train_penetration_loss_db", 0.0))
        self.train_length_m: float = float(cfg.get("train_length_m", 400.0))
        self.train_width_m: float = float(cfg.get("train_width_m", 3.4))
        self.channel_model: str = str(cfg.get("channel_model", "TDL-C"))
        self.noise_figure_db: float = float(cfg.get("noise_figure_db", 7.0))
        self.num_ssb_beams: int = int(cfg.get("num_ssb_beams", 8))
        self.max_rank: int = int(cfg.get("max_rank", 4))
        self.rank_threshold: float = float(cfg.get("rank_threshold", 0.1))
        self.apply_interferer_precoding: bool = bool(cfg.get("apply_interferer_precoding", True))
        self.store_interferer_channels: bool = bool(cfg.get("store_interferer_channels", False))
        self.tx_height_m: float = float(cfg.get("tx_height_m", 25.0))
        self.rx_height_m: float = float(cfg.get("rx_height_m", 1.5))
        self.cell_radius_m: float = float(cfg.get("cell_radius_m", 250.0))

        self._matlab_config = {
            # Basic
            "num_cells": self.num_cells,
            "num_ues": self.ues_per_shard,
            "num_snapshots": self.num_snapshots,
            "carrier_freq_hz": self.carrier_freq_hz,
            "scenario": self.scenario,
            "isd_m": self.isd_m,
            "tx_height_m": self.tx_height_m,
            "rx_height_m": self.rx_height_m,
            "cell_radius_m": self.cell_radius_m,
            # Antenna
            "bs_ant_v": self.bs_panel_v,
            "bs_ant_h": self.bs_panel_h,
            "bs_ant_p": self.bs_panel_p,
            "ue_ant_v": self.ue_panel_v,
            "ue_ant_h": self.ue_panel_h,
            "ue_ant_p": self.ue_panel_p,
            "xpd_db": self.xpd_db,
            # Bandwidth
            "n_rb": self._num_rb,
            "sc_inter": int(self.subcarrier_spacing),
            "bandwidth_hz": self.bandwidth_hz,
            # Topology (all scenarios)
            "topology_layout": self.topology_layout,
            "sectors_per_site": self.sectors_per_site,
            "track_offset_m": self.track_offset_m,
            "hypercell_size": self.hypercell_size,
            # UE distribution
            "ue_distribution": self.ue_distribution,
            "ue_height_m": self.ue_height_m,
            # Mobility
            "mobility_mode": self.mobility_mode,
            "ue_speed_kmh": self.ue_speed_kmh,
            "sample_interval_s": self.sample_interval_s,
            # HSR
            "train_penetration_loss_db": self.train_penetration_loss_db,
            "train_length_m": self.train_length_m,
            "train_width_m": self.train_width_m,
            # Channel model
            "channel_model": self.channel_model,
            # Estimation
            "link": self.link,
            "est_mode": self.channel_est_mode,
            "tdd_pattern": self.tdd_pattern_name,
            "pilot_type_ul": self.pilot_type_ul,
            "pilot_type_dl": self.pilot_type_dl,
            # SRS
            "srs_c_srs": self.srs_c_srs,
            "srs_b_srs": self.srs_b_srs,
            "srs_b_hop": self.srs_b_hop,
            "srs_n_rrc": self.srs_n_rrc,
            "srs_comb": self.srs_comb,
            "srs_periodicity": self.srs_periodicity,
            "srs_group_hopping": self.srs_group_hopping,
            "srs_sequence_hopping": self.srs_sequence_hopping,
            # SSB & precoding
            "num_ssb_beams": self.num_ssb_beams,
            "max_rank": self.max_rank,
            "rank_threshold": self.rank_threshold,
            # Interference
            "num_interfering_ues": self.num_interfering_ues,
            # Power
            "ue_tx_power_dbm": self.ue_tx_power_dbm,
            "noise_figure_db": self.noise_figure_db,
            # Output
            "output_dir": str(self.mat_dir),
            # Custom positions
            "custom_site_positions": cfg.get("custom_site_positions", None),
            "custom_ue_positions": cfg.get("custom_ue_positions", None),
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

        n_re = self._num_rb * 12
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

            has_dl_data = "Hf_dl_est" in d
            Hf_dl_est_mat = d.get("Hf_dl_est", None)
            mat_dl_sinr = np.asarray(d["dl_sinr_dB"]).flatten() if "dl_sinr_dB" in d else None
            mat_dl_sir = np.asarray(d["dl_sir_dB"]).flatten() if "dl_sir_dB" in d else None

            meta = d.get("meta", {})
            if isinstance(meta, np.ndarray) and meta.dtype.names:
                meta = meta[0, 0]

            no_ue = Hf_est.shape[0]

            ue_pos_all = _meta_get(meta, "ue_positions", np.zeros((3, no_ue)))
            K = int(_meta_get(meta, "num_cells", 7))
            noise_dBm = float(_meta_get(meta, "noise_dBm", -89.1))
            ptx_dBm = float(_meta_get(meta, "Ptx_BS_dBm", _meta_get(meta, "Ptx_dBm", 46.0)))
            ptx_per_re = ptx_dBm - 10.0 * np.log10(n_re)

            # New MATLAB outputs (Phase 3-4)
            Hf_all = d.get("Hf_all_cells")  # [no_ue, K, BsAnt, ue_ant, N_RB, no_ss]
            mat_ssb_rsrp = d.get("ssb_rsrp")        # [no_ue, K] dBm
            mat_ssb_rsrq = d.get("ssb_rsrq")        # [no_ue, K] dB
            mat_ssb_sinr = d.get("ssb_sinr")        # [no_ue, K] dB
            mat_ssb_beam = d.get("ssb_best_beam")   # [no_ue, K] 1-based
            mat_w_dl = d.get("w_dl_all")             # [no_ue, BsAnt, rank, N_RB]
            mat_dl_rank = d.get("dl_rank_all")       # [1, no_ue]
            mat_pre_sinr = d.get("ul_pre_sinr_dB")  # [1, no_ue]
            mat_pre_sinr_rb = d.get("ul_pre_sinr_per_rb")  # [no_ue, N_RB]
            mat_bs_pcis = _meta_get(meta, "bs_pcis", np.arange(K))

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

                # --- SSB measurements from MATLAB ---
                if mat_ssb_rsrp is not None and i_ue < mat_ssb_rsrp.shape[0]:
                    ssb_rsrp_list = [float(v) for v in mat_ssb_rsrp[i_ue]]
                else:
                    rsrp_row = mat_rsrp[i_ue] if i_ue < mat_rsrp.shape[0] else np.zeros(K)
                    ssb_rsrp_list = [
                        float(ptx_per_re + 10 * np.log10(max(float(g), 1e-30))) for g in rsrp_row
                    ]
                ssb_rsrq_list: list[float] | None = None
                ssb_sinr_list: list[float] | None = None
                ssb_beam_list: list[int] | None = None
                ssb_pcis_list: list[int] | None = None
                if mat_ssb_rsrq is not None and i_ue < mat_ssb_rsrq.shape[0]:
                    ssb_rsrq_list = [float(v) for v in mat_ssb_rsrq[i_ue]]
                if mat_ssb_sinr is not None and i_ue < mat_ssb_sinr.shape[0]:
                    ssb_sinr_list = [float(v) for v in mat_ssb_sinr[i_ue]]
                if mat_ssb_beam is not None and i_ue < mat_ssb_beam.shape[0]:
                    ssb_beam_list = [int(v) for v in mat_ssb_beam[i_ue]]
                if mat_bs_pcis is not None:
                    ssb_pcis_list = [int(v) for v in np.asarray(mat_bs_pcis).flatten()[:K]]

                # --- h_interferers from Hf_all_cells ---
                h_interferers_arr: np.ndarray | None = None
                _scids_raw = _meta_get(meta, "serving_cell_ids", None)
                _scids_arr = np.asarray(_scids_raw).flatten() if _scids_raw is not None else np.zeros(no_ue)
                sid_0 = int(_scids_arr[min(i_ue, len(_scids_arr) - 1)]) - 1  # 0-based
                if sid_0 < 0:
                    sid_0 = 0
                if Hf_all is not None and K > 1:
                    intf_mask = [k for k in range(K) if k != sid_0]
                    # Hf_all[i_ue]: [K, BsAnt, ue_ant, N_RB, no_ss]
                    h_intf_mat = Hf_all[i_ue][intf_mask]  # [K-1, BsAnt, ue_ant, N_RB, no_ss]
                    # Transpose to contract: [K-1, T, RB, BS_ant, UE_ant]
                    h_interferers_arr = np.transpose(h_intf_mat, (0, 4, 3, 1, 2))
                    if raw_gain > 0:
                        h_interferers_arr = h_interferers_arr / scale

                # --- Per-interferer DL precoding projection ---
                _intf_ranks: list[int] | None = None
                if h_interferers_arr is not None and self.apply_interferer_precoding:
                    from msg_embedding.phy_sim.precoding import project_interference_channels
                    _h_own_proxies = [
                        h_interferers_arr[ki] for ki in range(h_interferers_arr.shape[0])
                    ]
                    h_interferers_arr, _intf_ranks = project_interference_channels(
                        h_interferers_arr,
                        _h_own_proxies,
                        max_rank=self.max_rank,
                        rank_threshold=self.rank_threshold,
                    )

                # --- Precoding from MATLAB ---
                if mat_w_dl is not None and mat_dl_rank is not None:
                    _rank = int(np.asarray(mat_dl_rank).flatten()[i_ue])
                    _rank = max(_rank, 1)
                    # mat_w_dl[i_ue]: [BsAnt, max_rank, N_RB] → [N_RB, BsAnt, rank]
                    _w_raw = mat_w_dl[i_ue][:, :_rank, :]  # [BsAnt, rank, N_RB]
                    _w_dl = np.transpose(_w_raw, (2, 0, 1)).astype(np.complex64)  # [N_RB, BsAnt, rank]
                else:
                    from msg_embedding.phy_sim.precoding import compute_dl_precoding
                    _bs_ant = h_ul_est.shape[2]
                    _ue_ant = h_ul_est.shape[3]
                    _max_r = min(4, _ue_ant, _bs_ant)
                    _prec = compute_dl_precoding(h_ul_est, max_rank=_max_r)
                    _w_dl = _prec.w_dl
                    _rank = _prec.rank

                # --- Pre-SINR from MATLAB (preferred) or Python fallback ---
                _qr_pre_sinr_db: float | None = None
                _qr_pre_sinr_per_rb: np.ndarray | None = None
                if mat_pre_sinr is not None and mat_pre_sinr_rb is not None:
                    _qr_pre_sinr_db = float(np.asarray(mat_pre_sinr).flatten()[i_ue])
                    _qr_pre_sinr_per_rb = np.asarray(mat_pre_sinr_rb[i_ue]).astype(np.float32)
                elif h_ul_ideal is not None and h_ul_est is not None:
                    _sig_prb = np.mean(np.abs(h_ul_ideal) ** 2, axis=(0, 2, 3))
                    _err_prb = np.mean(np.abs(h_ul_est - h_ul_ideal) ** 2, axis=(0, 2, 3))
                    _ps_lin = _sig_prb / (_err_prb + 1e-30)
                    _qr_pre_sinr_per_rb = np.clip(
                        10.0 * np.log10(_ps_lin + 1e-30), -50.0, 50.0
                    ).astype(np.float32)
                    _wb_s = float(np.mean(_sig_prb))
                    _wb_e = float(np.mean(_err_prb))
                    _qr_pre_sinr_db = float(np.clip(
                        10.0 * math.log10(_wb_s / (_wb_e + 1e-30) + 1e-30), -50.0, 50.0
                    ))

                _serving_cell_id = sid_0

                # SRS frequency — accumulate full hopping cycle
                from msg_embedding.ref_signals.srs import srs_accumulated_rb_indices as _srs_rb_fn

                tdd = self._tdd_pattern
                _slot_idx = yielded % tdd.period_slots
                _sym_map = tdd.symbol_map(_slot_idx)
                _ul_syms = [j for j, m in enumerate(_sym_map) if m == "U"]
                _srs_sym = _ul_syms[-1] if _ul_syms else 0
                _srs_rb_idx = _srs_rb_fn(self._srs_resource_cfg, yielded, _srs_sym, self._num_rb)

                _est_pathloss_dB = ptx_dBm - noise_dBm - snr_dB

                sample_meta = {
                    "num_cells": K,
                    "isd_m": self.isd_m,
                    "scenario": self.scenario,
                    "pathloss_dB": _est_pathloss_dB,
                    "carrier_freq_hz": self.carrier_freq_hz,
                    "bandwidth_hz": self.bandwidth_hz,
                    "subcarrier_spacing": self.subcarrier_spacing,
                    "num_rb": self._num_rb,
                    "ue_speed_kmh": self._matlab_config.get("ue_speed_kmh", 0),
                    "mobility_mode": self._matlab_config.get("mobility_mode", "linear"),
                    "tdd_pattern": self.tdd_pattern_name,
                    "srs_periodicity": self.srs_periodicity,
                    "srs_group_hopping": self.srs_group_hopping,
                    "srs_sequence_hopping": self.srs_sequence_hopping,
                    "srs_comb": self.srs_comb,
                    "srs_c_srs": self.srs_c_srs,
                    "srs_b_srs": self.srs_b_srs,
                    "srs_n_rrc": self.srs_n_rrc,
                    "srs_b_hop": self.srs_b_hop,
                    "srs_rb_indices": _srs_rb_idx.tolist(),
                    "shard": shard_num,
                    "ue_idx": i_ue,
                    "bs_panel": [self.bs_panel_h, self.bs_panel_v, self.bs_panel_p],
                    "ue_panel": [self.ue_panel_h, self.ue_panel_v, self.ue_panel_p],
                    "xpd_db": self.xpd_db,
                    "ue_tx_power_dbm": self.ue_tx_power_dbm,
                    "tx_power_dbm": ptx_dBm,
                    "dl_rank": _rank,
                    "w_dl_shape": list(_w_dl.shape),
                    "w_dl": _w_dl,
                    "interferer_precoding_applied": self.apply_interferer_precoding,
                    "store_interferer_channels": self.store_interferer_channels,
                }
                if _intf_ranks is not None:
                    _sample_meta["interferer_ranks"] = _intf_ranks

                if self.channel_est_mode == "ideal":
                    h_ul_est = h_ul_ideal.copy()

                est_mode_out: ChannelEstMode = "ls_linear"
                if self.channel_est_mode in ("ideal", "ls_linear", "ls_mmse", "ls_hop_concat"):
                    est_mode_out = self.channel_est_mode  # type: ignore[assignment]

                h_dl_ideal = np.conj(h_ul_ideal)
                if has_dl_data and Hf_dl_est_mat is not None:
                    h_dl_est_raw = np.transpose(Hf_dl_est_mat[i_ue], (3, 2, 0, 1))
                    if raw_gain > 0:
                        h_dl_est_raw = h_dl_est_raw / scale
                    h_dl_est = h_dl_est_raw
                else:
                    h_dl_est = np.conj(h_ul_est)

                # -- UL SNR/SINR from UE TX power --
                _tx_pwr_offset = ptx_dBm - self.ue_tx_power_dbm
                _ul_snr = float(np.clip(snr_dB - _tx_pwr_offset, -50, 50))
                _ul_sinr = float(np.clip(
                    -10.0 * math.log10(
                        10.0 ** (-_ul_snr / 10.0) + 10.0 ** (-sir_dB / 10.0)
                    ), -50, 50
                ))

                # Common fields for all link modes
                # Interference signal (derived from h_interferers)
                _interference_signal: np.ndarray | None = None
                if h_interferers_arr is not None:
                    _intf_sum = np.sum(h_interferers_arr, axis=0)  # [T, RB, BS, UE]
                    _interference_signal = np.sum(_intf_sum, axis=2).astype(np.complex64)  # [T, RB, UE]

                # UL/DL SIR from MATLAB
                _ul_sir = float(np.nan_to_num(np.clip(sir_dB, -50, 50), nan=50.0)) if K > 1 else None
                _dl_sir_val = float(mat_dl_sir[i_ue]) if mat_dl_sir is not None and i_ue < len(mat_dl_sir) else _ul_sir

                _common = dict(
                    h_interferers=(
                        h_interferers_arr.astype(np.complex64)
                        if h_interferers_arr is not None and self.store_interferer_channels
                        else None
                    ),
                    interference_signal=_interference_signal if self.store_interferer_channels else None,
                    noise_power_dBm=noise_dBm,
                    ssb_rsrp_dBm=ssb_rsrp_list,
                    ssb_rsrq_dB=ssb_rsrq_list,
                    ssb_sinr_dB=ssb_sinr_list,
                    ssb_best_beam_idx=ssb_beam_list,
                    ssb_pcis=ssb_pcis_list,
                    channel_est_mode=est_mode_out,
                    serving_cell_id=_serving_cell_id,
                    ue_position=ue_p,
                    channel_model=self.channel_model,
                    tdd_pattern=self.tdd_pattern_name,
                    num_interfering_ues=self.num_interfering_ues,
                    h_ul_true=h_ul_ideal.astype(np.complex64),
                    h_ul_est=h_ul_est.astype(np.complex64),
                    h_dl_true=h_dl_ideal.astype(np.complex64),
                    h_dl_est=h_dl_est.astype(np.complex64),
                    ul_sir_dB=_ul_sir,
                    dl_sir_dB=_dl_sir_val,
                    ssb_rsrp_true_dBm=ssb_rsrp_list,
                    ssb_sinr_true_dB=ssb_sinr_list,
                    ul_pre_sinr_dB=_qr_pre_sinr_db,
                    ul_pre_sinr_per_rb=_qr_pre_sinr_per_rb,
                    ul_snr_dB=_ul_snr,
                    ul_sinr_dB=_ul_sinr,
                    w_dl=_w_dl,
                    dl_rank=_rank,
                    source="quadriga_real",
                    sample_id=str(uuid.uuid4()),
                    created_at=datetime.now(timezone.utc),
                    meta=sample_meta,
                )

                if self.link == "BOTH":
                    _dl_sinr = float(mat_dl_sinr[i_ue]) if mat_dl_sinr is not None else sinr_dB
                    _dl_sir = float(mat_dl_sir[i_ue]) if mat_dl_sir is not None else sir_dB
                    yield ChannelSample(
                        h_serving_true=h_dl_ideal.astype(np.complex64),
                        h_serving_est=h_dl_est.astype(np.complex64),
                        snr_dB=snr_dB,
                        sir_dB=_dl_sir,
                        sinr_dB=_dl_sinr,
                        link="DL",
                        link_pairing="paired",
                        **_common,
                    )
                elif self.link == "DL":
                    yield ChannelSample(
                        h_serving_true=h_dl_ideal.astype(np.complex64),
                        h_serving_est=h_dl_est.astype(np.complex64),
                        snr_dB=snr_dB,
                        sir_dB=sir_dB,
                        sinr_dB=sinr_dB,
                        link="DL",
                        **_common,
                    )
                else:
                    yield ChannelSample(
                        h_serving_true=h_ul_ideal.astype(np.complex64),
                        h_serving_est=h_ul_est.astype(np.complex64),
                        snr_dB=snr_dB,
                        sir_dB=sir_dB,
                        sinr_dB=sinr_dB,
                        link="UL",
                        **_common,
                    )
                yielded += 1

            del Hf_est, Hf_ideal, Hf_all, d
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
            "bandwidth_hz": self.bandwidth_hz,
            "subcarrier_spacing": self.subcarrier_spacing,
            "RB": self._num_rb,
            "link": self.link,
            "channel_est_mode": self.channel_est_mode,
            "pilot_type_ul": self.pilot_type_ul,
            "pilot_type_dl": self.pilot_type_dl,
            "tdd_pattern": self.tdd_pattern_name,
            "srs_periodicity": self.srs_periodicity,
            "srs_group_hopping": self.srs_group_hopping,
            "srs_sequence_hopping": self.srs_sequence_hopping,
            "srs_comb": self.srs_comb,
            "srs_c_srs": self.srs_c_srs,
            "srs_b_srs": self.srs_b_srs,
            "srs_n_rrc": self.srs_n_rrc,
            "srs_b_hop": self.srs_b_hop,
            "num_interfering_ues": self.num_interfering_ues,
            "bs_panel": [self.bs_panel_h, self.bs_panel_v, self.bs_panel_p],
            "ue_panel": [self.ue_panel_h, self.ue_panel_v, self.ue_panel_p],
            "xpd_db": self.xpd_db,
            "ue_tx_power_dbm": self.ue_tx_power_dbm,
            "topology_layout": self.topology_layout,
            "sectors_per_site": self.sectors_per_site,
            "track_offset_m": self.track_offset_m,
            "hypercell_size": self.hypercell_size,
            "ue_distribution": self.ue_distribution,
            "channel_model": self.channel_model,
            "noise_figure_db": self.noise_figure_db,
            "mobility_mode": self.mobility_mode,
            "ue_speed_kmh": self.ue_speed_kmh,
            "sample_interval_s": self.sample_interval_s,
            "train_penetration_loss_db": self.train_penetration_loss_db,
            "num_ssb_beams": self.num_ssb_beams,
        }


__all__ = ["QuadrigaRealSource"]
