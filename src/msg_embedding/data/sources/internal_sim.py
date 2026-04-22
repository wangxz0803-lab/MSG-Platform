"""Internal first-party multi-cell channel simulator data source.

Pure-Python / NumPy implementation of a 3GPP 38.901-inspired multi-cell
channel simulator.  No external dependencies (no MATLAB, no Sionna).

Physics modelled
----------------
* **Large-scale fading** — 3GPP 38.901 UMa/UMi/InF NLOS pathloss with
  lognormal shadowing (σ_SF configurable per scenario).
* **Small-scale fading** — simplified TDL-C tapped-delay-line model with
  an exponential power-delay profile.  Per-tap gains are complex Gaussian
  with optional Rician K-factor on the first tap.  Doppler shift is
  applied per-tap via a time-domain phase ramp.
* **Multi-cell interference** — channels from all K BSs are computed for
  each UE; the strongest received power selects the serving cell and the
  remaining K-1 are interferers.
* **Reference signals** — Zadoff-Chu (UL SRS) and Gold-PRBS CSI-RS (DL)
  from the repo's own ``ref_signals`` package.
* **Channel estimation** — delegated to
  :func:`msg_embedding.channel_est.estimate_channel` with configurable
  mode (ideal / ls_linear / ls_mmse).
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ..contract import ChannelSample
from .base import DataSource, register_source

# ---------------------------------------------------------------------------
# Re-usable helpers (same pattern as sionna_rt.py)
# ---------------------------------------------------------------------------


def _dict_get(cfg: Any, key: str, default: Any) -> Any:
    """Small helper that works for DictConfig / dict / SimpleNamespace / None."""
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except Exception:
            pass
    return getattr(cfg, key, default)


def _sites_to_rings(num_sites: int) -> int:
    """Convert a site count to the minimum number of hex rings required.

    Ring 0 = 1 site, ring 1 = 7, ring 2 = 19, ring 3 = 37, ...
    Cumulative: ``1 + 3*r*(r+1)`` sites for ``r`` rings.
    """
    if num_sites <= 1:
        return 0
    r = 0
    while True:
        total = 1 + 3 * r * (r + 1)
        if total >= num_sites:
            return r
        r += 1
        if r > 20:
            return r


def _clamp_db(val: float, lo: float = -49.9, hi: float = 49.9) -> float:
    """Clamp a dB value to the ChannelSample contract range."""
    return float(np.clip(val, lo, hi))


# ---------------------------------------------------------------------------
# Pathloss models (3GPP TR 38.901 Table 7.4.1-1)
# ---------------------------------------------------------------------------

# Each model returns (PL_dB, shadow_std_dB).  ``d_3d`` in metres, ``fc``
# in GHz, ``h_bs`` and ``h_ut`` in metres.


def _pathloss_uma_nlos(d_3d: float, fc_ghz: float, h_bs: float, h_ut: float) -> tuple[float, float]:
    """38.901 UMa NLOS: Table 7.4.1-1, eq (7.4.1-1)."""
    d_3d = max(d_3d, 1.0)  # guard
    pl = 13.54 + 39.08 * math.log10(d_3d) + 20.0 * math.log10(fc_ghz) - 0.6 * (h_ut - 1.5)
    return pl, 6.0  # σ_SF = 6 dB


def _pathloss_umi_nlos(d_3d: float, fc_ghz: float, h_bs: float, h_ut: float) -> tuple[float, float]:
    """38.901 UMi Street Canyon NLOS (simplified)."""
    d_3d = max(d_3d, 1.0)
    pl = 22.4 + 35.3 * math.log10(d_3d) + 21.3 * math.log10(fc_ghz) - 0.3 * (h_ut - 1.5)
    return pl, 7.82


def _pathloss_inf(d_3d: float, fc_ghz: float, h_bs: float, h_ut: float) -> tuple[float, float]:
    """38.901 InF-SH NLOS (simplified)."""
    d_3d = max(d_3d, 1.0)
    pl = 31.84 + 21.50 * math.log10(d_3d) + 19.0 * math.log10(fc_ghz)
    return pl, 7.56


_PATHLOSS_MODELS = {
    "UMa_NLOS": _pathloss_uma_nlos,
    "UMi_NLOS": _pathloss_umi_nlos,
    "InF": _pathloss_inf,
}


# 38.901 Table 7.5-6: Large-scale parameter statistics
# Format: {scenario: {param: (mu, sigma, decorr_dist_m)}}
_LSP_TABLE_38901 = {
    "UMa_NLOS": {
        "lgDS": (-6.44, 0.39, 40.0),
        "lgASA": (1.81, 0.20, 50.0),
        "lgASD": (1.20, 0.43, 50.0),
        "lgZSA": (0.95, 0.16, 50.0),
        "lgZSD": (0.47, 0.40, 50.0),
        "lgSF": (0.0, 6.0, 50.0),
        "lgK": (0.0, 0.0, 0.0),  # NLOS: no K-factor
    },
    "UMi_NLOS": {
        "lgDS": (-6.89, 0.54, 10.0),
        "lgASA": (1.76, 0.16, 25.0),
        "lgASD": (1.25, 0.42, 25.0),
        "lgZSA": (0.93, 0.22, 25.0),
        "lgZSD": (0.39, 0.27, 25.0),
        "lgSF": (0.0, 7.82, 13.0),
        "lgK": (0.0, 0.0, 0.0),
    },
    "InF": {
        "lgDS": (-7.20, 0.30, 10.0),
        "lgASA": (1.56, 0.15, 10.0),
        "lgASD": (1.08, 0.36, 10.0),
        "lgZSA": (0.74, 0.21, 10.0),
        "lgZSD": (0.28, 0.30, 10.0),
        "lgSF": (0.0, 7.56, 10.0),
        "lgK": (0.0, 0.0, 0.0),
    },
    # LOS scenarios
    "UMa_LOS": {
        "lgDS": (-7.03, 0.66, 30.0),
        "lgASA": (1.81, 0.20, 15.0),
        "lgASD": (1.06, 0.28, 8.0),
        "lgZSA": (0.95, 0.16, 15.0),
        "lgZSD": (0.41, 0.39, 15.0),
        "lgSF": (0.0, 4.0, 37.0),
        "lgK": (9.0, 3.5, 12.0),
    },
    "UMi_LOS": {
        "lgDS": (-7.19, 0.40, 7.0),
        "lgASA": (1.75, 0.19, 8.0),
        "lgASD": (1.21, 0.41, 8.0),
        "lgZSA": (0.73, 0.20, 8.0),
        "lgZSD": (0.41, 0.31, 12.0),
        "lgSF": (0.0, 3.0, 10.0),
        "lgK": (9.0, 5.0, 15.0),
    },
}


# Per-scenario spatial correlation coefficients (Fix 4)
_SCENARIO_SPATIAL_RHO = {
    "UMa_NLOS": 0.7,
    "UMa_LOS": 0.8,
    "UMi_NLOS": 0.5,
    "UMi_LOS": 0.6,
    "InF": 0.3,
}


# ---------------------------------------------------------------------------
# TDL channel generation
# ---------------------------------------------------------------------------


def _generate_tdl_channel(
    rng: np.random.Generator,
    num_taps: int,
    num_ofdm_sym: int,
    num_rb: int,
    num_tx_ant: int,
    num_rx_ant: int,
    tau_rms_ns: float,
    subcarrier_spacing_hz: float,
    doppler_hz: float,
    rician_k_linear: float = 0.0,
    tdl_profile: Any = None,
    los_aod_rad: float = 0.0,
    los_aoa_rad: float = 0.0,
    spatial_corr_rho: float = 0.7,
) -> np.ndarray:
    """Generate a frequency-domain channel matrix via a tapped-delay-line model.

    Returns shape ``[T, RB, tx_ant, rx_ant]`` complex128 (cast to c64 by caller).

    When ``tdl_profile`` is provided (a :class:`TDLProfile` from
    ``channel_models.tdl``), uses the profile's tap delays and powers per
    3GPP 38.901.  Otherwise falls back to an exponential PDP.

    The frequency-domain channel is obtained by DFT across taps::

        H[k, t, tx, rx] = sum_l  h_tap[l, t, tx, rx] * exp(-j 2pi k tau_l f_k)

    Doppler is modelled as a per-tap time-domain phase ramp with a random
    Doppler shift uniformly distributed in ``[-doppler_hz, +doppler_hz]``.
    """
    T = num_ofdm_sym
    N_tx = num_tx_ant
    N_rx = num_rx_ant
    N_fft = num_rb

    tau_rms_s = tau_rms_ns * 1e-9

    if tdl_profile is not None:
        # Use 3GPP profile tap delays and powers
        L = tdl_profile.num_taps
        pdp = tdl_profile.powers_normalized()
        delays_s = tdl_profile.delays_seconds(tau_rms_s)
        # Override Rician K for LOS profiles
        if tdl_profile.is_los and tdl_profile.k_factor_dB is not None:
            rician_k_linear = 10.0 ** (tdl_profile.k_factor_dB / 10.0)
            los_tap = tdl_profile.los_tap_index
        else:
            los_tap = 0
    else:
        # Fallback: exponential PDP
        L = max(num_taps, 1)
        tap_spacing_s = 1.0 / (subcarrier_spacing_hz * num_rb) if num_rb > 0 else 1e-6
        decay = tap_spacing_s / tau_rms_s if tau_rms_s > 0 and tap_spacing_s > 0 else 1.0
        pdp = np.exp(-np.arange(L, dtype=np.float64) * decay)
        pdp /= pdp.sum()
        delays_s = np.arange(L, dtype=np.float64) * tap_spacing_s
        los_tap = 0

    # --- Spatial correlation (exponential model: R[i,j] = rho^|i-j|) -----
    rho = spatial_corr_rho  # per-scenario ULA spatial correlation coefficient
    R_tx = np.zeros((N_tx, N_tx), dtype=np.float64)
    for i in range(N_tx):
        for j in range(N_tx):
            R_tx[i, j] = rho ** abs(i - j)
    R_rx = np.zeros((N_rx, N_rx), dtype=np.float64)
    for i in range(N_rx):
        for j in range(N_rx):
            R_rx[i, j] = rho ** abs(i - j)
    # Cholesky decomposition for coloring: H_corr = R_rx^(1/2) * H_iid * R_tx^(T/2)
    L_tx = np.linalg.cholesky(R_tx)  # [N_tx, N_tx]
    L_rx = np.linalg.cholesky(R_rx)  # [N_rx, N_rx]

    # --- Per-tap complex gains -------------------------------------------
    h_taps_real = rng.standard_normal((L, 1, N_tx, N_rx))
    h_taps_imag = rng.standard_normal((L, 1, N_tx, N_rx))
    h_taps_iid = (h_taps_real + 1j * h_taps_imag) / math.sqrt(2.0)

    # Apply spatial correlation: H_corr = L_rx @ H_iid @ L_tx^T per tap
    h_taps_corr = np.zeros_like(h_taps_iid)
    for l_idx in range(L):
        # h_taps_iid[l_idx, 0] has shape [N_tx, N_rx]
        h_taps_corr[l_idx, 0] = L_tx @ h_taps_iid[l_idx, 0] @ L_rx.T

    h_taps = np.broadcast_to(h_taps_corr, (L, T, N_tx, N_rx)).copy()

    # Apply PDP scaling
    pdp_sqrt = np.sqrt(pdp)
    h_taps *= pdp_sqrt[:, None, None, None]

    # Rician LOS component on the designated tap
    if rician_k_linear > 0:
        K = rician_k_linear
        nlos_scale = math.sqrt(1.0 / (1.0 + K))
        los_scale = math.sqrt(K / (1.0 + K))
        # LOS component with steering vectors for spatial structure
        d_spacing = 0.5  # antenna spacing in wavelengths
        a_tx = np.exp(1j * 2 * np.pi * d_spacing * np.arange(N_tx) * np.sin(los_aod_rad)) / np.sqrt(
            N_tx
        )
        a_rx = np.exp(1j * 2 * np.pi * d_spacing * np.arange(N_rx) * np.sin(los_aoa_rad)) / np.sqrt(
            N_rx
        )
        los_matrix = np.outer(a_tx, a_rx.conj())  # [N_tx, N_rx]
        h_taps[los_tap] = (
            h_taps[los_tap] * nlos_scale + los_scale * pdp_sqrt[los_tap] * los_matrix[None, :, :]
        )

    # --- Doppler via improved Jakes sum-of-sinusoids model ----------------
    if doppler_hz > 0 and T > 1:
        N_sinusoids = 16  # number of sinusoids per tap for Jakes spectrum
        T_sym = 1.0 / subcarrier_spacing_hz
        t_axis = np.arange(T, dtype=np.float64) * T_sym  # [T]

        for l_idx in range(L):
            # Sum-of-sinusoids: sum_n cos(2*pi*f_d*cos(alpha_n)*t + phi_n)
            alpha_n = rng.uniform(0, 2 * np.pi, size=N_sinusoids)  # arrival angles
            phi_n = rng.uniform(0, 2 * np.pi, size=N_sinusoids)  # random phases
            # Real and imaginary parts for complex Doppler process
            doppler_real = np.zeros(T, dtype=np.float64)
            doppler_imag = np.zeros(T, dtype=np.float64)
            for n in range(N_sinusoids):
                f_shift = doppler_hz * np.cos(alpha_n[n])
                doppler_real += np.cos(2.0 * np.pi * f_shift * t_axis + phi_n[n])
                doppler_imag += np.sin(2.0 * np.pi * f_shift * t_axis + phi_n[n])
            # Normalize by sqrt(N_sinusoids) for unit power
            doppler_process = (doppler_real + 1j * doppler_imag) / math.sqrt(N_sinusoids)
            h_taps[l_idx] *= doppler_process[:, None, None]

    # --- DFT: time-delay → frequency-domain ------------------------------
    # Use actual tap delays for phase computation (not integer tap indices)
    k_axis = np.arange(N_fft, dtype=np.float64)
    delta_f = 12 * subcarrier_spacing_hz  # frequency spacing per RB (12 subcarriers per RB)
    # DFT kernel: [N_fft, L] with physical delays
    dft_matrix = np.exp(-1j * 2.0 * np.pi * np.outer(k_axis * delta_f, delays_s))

    h_flat = h_taps.transpose(1, 2, 3, 0).reshape(-1, L)
    H_flat = h_flat @ dft_matrix.T
    H = H_flat.reshape(T, N_tx, N_rx, N_fft).transpose(0, 3, 1, 2)

    return H


# ---------------------------------------------------------------------------
# UE placement helpers
# ---------------------------------------------------------------------------


def _place_ues_uniform(
    rng: np.random.Generator,
    num_ues: int,
    cell_radius_m: float,
    h_ut: float,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """Uniformly distributed UEs within a circle of ``cell_radius_m``."""
    if center is None:
        center = np.zeros(2, dtype=np.float64)
    # Polar coordinates with sqrt(r) for uniform density
    r = cell_radius_m * np.sqrt(rng.uniform(0, 1, size=num_ues))
    theta = rng.uniform(0, 2 * np.pi, size=num_ues)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    z = np.full(num_ues, h_ut, dtype=np.float64)
    return np.stack([x, y, z], axis=-1)  # [num_ues, 3]


def _place_ues_clustered(
    rng: np.random.Generator,
    num_ues: int,
    cell_radius_m: float,
    h_ut: float,
    num_clusters: int = 3,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """Clustered UE placement: ``num_clusters`` hotspots within the cell."""
    if center is None:
        center = np.zeros(2, dtype=np.float64)
    # Cluster centres uniformly within the cell
    cluster_r = cell_radius_m * 0.6 * np.sqrt(rng.uniform(0, 1, size=num_clusters))
    cluster_theta = rng.uniform(0, 2 * np.pi, size=num_clusters)
    cx = center[0] + cluster_r * np.cos(cluster_theta)
    cy = center[1] + cluster_r * np.sin(cluster_theta)

    # Assign UEs to clusters
    assignments = rng.integers(0, num_clusters, size=num_ues)
    cluster_spread = cell_radius_m * 0.15
    dx = rng.normal(0, cluster_spread, size=num_ues)
    dy = rng.normal(0, cluster_spread, size=num_ues)
    x = cx[assignments] + dx
    y = cy[assignments] + dy
    z = np.full(num_ues, h_ut, dtype=np.float64)
    return np.stack([x, y, z], axis=-1)


def _place_ues_hotspot(
    rng: np.random.Generator,
    num_ues: int,
    cell_radius_m: float,
    h_ut: float,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """Hotspot placement: 70% near cell centre, 30% at cell edge."""
    if center is None:
        center = np.zeros(2, dtype=np.float64)
    n_centre = int(num_ues * 0.7)
    n_edge = num_ues - n_centre
    # Centre UEs: within 30% of cell radius
    ues_c = _place_ues_uniform(rng, max(n_centre, 1), cell_radius_m * 0.3, h_ut, center)
    # Edge UEs: between 70%-100% of cell radius
    r_edge = cell_radius_m * (0.7 + 0.3 * np.sqrt(rng.uniform(0, 1, size=max(n_edge, 1))))
    theta_edge = rng.uniform(0, 2 * np.pi, size=max(n_edge, 1))
    x_e = center[0] + r_edge * np.cos(theta_edge)
    y_e = center[1] + r_edge * np.sin(theta_edge)
    z_e = np.full(max(n_edge, 1), h_ut, dtype=np.float64)
    ues_e = np.stack([x_e, y_e, z_e], axis=-1)
    all_ues = np.concatenate([ues_c, ues_e], axis=0)
    return all_ues[:num_ues]


_UE_PLACERS = {
    "uniform": _place_ues_uniform,
    "clustered": _place_ues_clustered,
    "hotspot": _place_ues_hotspot,
}


# ---------------------------------------------------------------------------
# Pilot generation helpers
# ---------------------------------------------------------------------------


def _generate_pilots_srs(
    num_rb: int,
    cell_id: int,
    slot: int = 0,
) -> np.ndarray:
    """Generate Zadoff-Chu SRS pilot symbols for UL.

    Returns shape ``[num_rb]`` complex128.
    """
    from msg_embedding.ref_signals.zc import zadoff_chu

    Nzc = num_rb
    # Find the largest prime <= Nzc for the ZC length
    if Nzc < 2:
        return np.ones(max(num_rb, 1), dtype=np.complex128)

    # Pick a prime Nzc
    nzc_prime = Nzc
    while nzc_prime >= 2:
        if _is_prime_simple(nzc_prime):
            break
        nzc_prime -= 1
    if nzc_prime < 2:
        nzc_prime = 2

    # Root index derived from cell_id (avoid 0)
    u = max(1, (cell_id * 7 + 1) % nzc_prime)
    if u == 0:
        u = 1

    seq = zadoff_chu(u, nzc_prime)
    # Pad or truncate to num_rb
    if len(seq) < num_rb:
        pad = np.ones(num_rb - len(seq), dtype=np.complex128)
        seq = np.concatenate([seq, pad])
    else:
        seq = seq[:num_rb]
    return seq


def _generate_pilots_csirs(
    num_rb: int,
    cell_id: int,
    slot: int = 0,
    symbol: int = 0,
) -> np.ndarray:
    """Generate CSI-RS pilot symbols for DL.

    Returns shape ``[num_rb]`` complex128.
    """
    from msg_embedding.ref_signals.csi_rs import csi_rs_sequence

    n_rb_csi = max(1, num_rb)
    seq = csi_rs_sequence(
        n_ID=cell_id % 1024,
        slot=slot,
        symbol=symbol,
        density=1.0,
        n_rb=n_rb_csi,
    )
    if len(seq) < num_rb:
        pad = np.ones(num_rb - len(seq), dtype=np.complex128)
        seq = np.concatenate([seq, pad])
    else:
        seq = seq[:num_rb]
    return seq


def _is_prime_simple(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    r = int(math.isqrt(n))
    return all(n % i != 0 for i in range(3, r + 1, 2))


# ---------------------------------------------------------------------------
# LOS probability (38.901 Table 7.4.2-1)
# ---------------------------------------------------------------------------


def _los_probability(scenario: str, d_2d: float) -> float:
    """38.901 Table 7.4.2-1 LOS probability."""
    d_2d = max(d_2d, 1.0)
    if scenario in ("UMa_NLOS", "UMa_LOS"):
        return min(18.0 / d_2d, 1.0) * (1.0 - math.exp(-d_2d / 63.0)) + math.exp(-d_2d / 63.0)
    elif scenario in ("UMi_NLOS", "UMi_LOS"):
        return min(18.0 / d_2d, 1.0) * (1.0 - math.exp(-d_2d / 36.0)) + math.exp(-d_2d / 36.0)
    elif scenario.startswith("InF"):
        d_clutter = 10.0  # typical for InF-SH
        return math.exp(-d_2d / d_clutter)
    return 0.0  # default NLOS


# ---------------------------------------------------------------------------
# InternalSimSource
# ---------------------------------------------------------------------------


@register_source
class InternalSimSource(DataSource):
    """3GPP 38.901-inspired multi-cell channel simulator (pure Python/NumPy).

    Generates physically meaningful :class:`ChannelSample` instances by
    modelling pathloss, shadowing, TDL fading, multi-cell interference,
    reference-signal insertion, and pilot-based channel estimation.

    Supported config keys mirror :class:`SionnaRTSource` for consistency.
    See the module docstring and ``validate_config`` for the full list.
    """

    name = "internal_sim"

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------
    def validate_config(self) -> None:
        cfg = self.config

        # -- Topology --------------------------------------------------------
        _ns_raw = _dict_get(cfg, "num_sites", None)
        _nc_raw = _dict_get(cfg, "num_cells", None)
        if _ns_raw is not None:
            self.num_sites: int = int(_ns_raw)
        elif _nc_raw is not None:
            self.num_sites = int(_nc_raw)
        else:
            self.num_sites = 7

        self.num_ues: int = int(_dict_get(cfg, "num_ues", 5))
        self.num_samples: int = int(_dict_get(cfg, "num_samples", 10))
        self.isd_m: float = float(_dict_get(cfg, "isd_m", 500.0))
        self.sectors_per_site: int = int(_dict_get(cfg, "sectors_per_site", 3))
        if self.sectors_per_site not in (1, 3):
            raise ValueError(f"sectors_per_site must be 1 or 3, got {self.sectors_per_site}.")
        self.tx_height_m: float = float(_dict_get(cfg, "tx_height_m", 25.0))

        # -- OFDM parameters -------------------------------------------------
        self.carrier_freq_hz: float = float(_dict_get(cfg, "carrier_freq_hz", 3.5e9))
        self.bandwidth_hz: float = float(_dict_get(cfg, "bandwidth_hz", 100e6))
        self.subcarrier_spacing: float = float(_dict_get(cfg, "subcarrier_spacing", 30e3))
        self.num_ofdm_symbols: int = int(_dict_get(cfg, "num_ofdm_symbols", 14))
        # Number of RBs: derived from bandwidth / (12 * SCS)
        self._num_rb: int = int(
            _dict_get(
                cfg,
                "num_rb",
                max(1, int(self.bandwidth_hz / (12 * self.subcarrier_spacing))),
            )
        )

        # -- Antenna arrays ---------------------------------------------------
        _legacy_bs = int(_dict_get(cfg, "num_bs_ant", 4))
        _legacy_ue = int(_dict_get(cfg, "num_ue_ant", 2))
        self.num_bs_tx_ant: int = int(_dict_get(cfg, "num_bs_tx_ant", _legacy_bs))
        self.num_bs_rx_ant: int = int(_dict_get(cfg, "num_bs_rx_ant", _legacy_bs))
        self.num_ue_tx_ant: int = int(_dict_get(cfg, "num_ue_tx_ant", _legacy_ue))
        self.num_ue_rx_ant: int = int(_dict_get(cfg, "num_ue_rx_ant", _legacy_ue))
        self.num_bs_ant: int = max(self.num_bs_tx_ant, self.num_bs_rx_ant)
        self.num_ue_ant: int = max(self.num_ue_tx_ant, self.num_ue_rx_ant)

        # -- Power / mobility -------------------------------------------------
        self.tx_power_dbm: float = float(_dict_get(cfg, "tx_power_dbm", 43.0))
        self._tx_power_explicitly_set: bool = _dict_get(cfg, "tx_power_dbm", None) is not None
        self.ue_speed_kmh: float = float(_dict_get(cfg, "ue_speed_kmh", 3.0))
        self.ue_distribution: str = str(_dict_get(cfg, "ue_distribution", "uniform"))
        if self.ue_distribution not in {"uniform", "clustered", "hotspot"}:
            raise ValueError(
                f"Unknown ue_distribution {self.ue_distribution!r}; expected "
                "'uniform' / 'clustered' / 'hotspot'."
            )

        # -- Pilot / estimation / link ----------------------------------------
        self.pilot_type: str = str(_dict_get(cfg, "pilot_type", "csi_rs_gold"))
        if self.pilot_type not in {"srs_zc", "csi_rs_gold"}:
            raise ValueError(
                f"Unknown pilot_type {self.pilot_type!r}; expected " "'srs_zc' / 'csi_rs_gold'."
            )
        self.pilot_type_dl: str = str(_dict_get(cfg, "pilot_type_dl", "csi_rs_gold"))
        self.pilot_type_ul: str = str(_dict_get(cfg, "pilot_type_ul", "srs_zc"))
        self.channel_est_mode: str = str(_dict_get(cfg, "channel_est_mode", "ls_linear"))
        if self.channel_est_mode not in {"ideal", "ls_linear", "ls_mmse"}:
            raise ValueError(f"Unknown channel_est_mode {self.channel_est_mode!r}.")
        self.link: str = str(_dict_get(cfg, "link", "DL")).upper()
        if self.link not in {"UL", "DL", "BOTH"}:
            raise ValueError(f"link must be 'UL', 'DL', or 'both', got {self.link!r}.")

        # -- Custom positions -------------------------------------------------
        self.custom_site_positions: list[dict[str, float]] | None = _dict_get(
            cfg, "custom_site_positions", None
        )
        self.custom_ue_positions: list[dict[str, float]] | None = _dict_get(
            cfg, "custom_ue_positions", None
        )

        # -- Scenario / TDL / seed --------------------------------------------
        _scenario_aliases = {
            "munich": "UMa_NLOS",
            "etoile": "UMa_NLOS",
            "custom_osm": "UMa_NLOS",
            "3GPP_38.901_UMa_NLOS": "UMa_NLOS",
            "3GPP_38.901_UMi_NLOS": "UMi_NLOS",
        }
        raw_scenario = str(_dict_get(cfg, "scenario", "UMa_NLOS"))
        self.scenario: str = _scenario_aliases.get(raw_scenario, raw_scenario)
        if self.scenario not in _PATHLOSS_MODELS:
            raise ValueError(
                f"Unknown scenario {self.scenario!r}; expected one of "
                f"{sorted(_PATHLOSS_MODELS)}."
            )
        self._seed: int = int(_dict_get(cfg, "seed", 42))

        # Scenario-specific TX power defaults (Fix 8: InF uses small cell power)
        if not self._tx_power_explicitly_set:
            _default_tx_power = {
                "UMa_NLOS": 43.0,
                "UMa_LOS": 43.0,
                "UMi_NLOS": 33.0,
                "UMi_LOS": 33.0,
                "InF": 24.0,
            }
            self.tx_power_dbm = _default_tx_power.get(self.scenario, 43.0)

        # -- Channel model profile (TDL-A through TDL-E) --------------------
        from msg_embedding.channel_models.tdl import get_tdl_profile, list_tdl_profiles

        self.channel_model_name: str = (
            str(_dict_get(cfg, "channel_model", "TDL-C")).upper().replace("_", "-")
        )
        try:
            self._tdl_profile = get_tdl_profile(self.channel_model_name)
        except ValueError as exc:
            raise ValueError(
                f"Unknown channel_model {self.channel_model_name!r}; "
                f"available: {list_tdl_profiles()}"
            ) from exc
        self.num_taps: int = self._tdl_profile.num_taps

        # -- TDL model params -------------------------------------------------
        # RMS delay spread in nanoseconds (38.901 Table 7.5-6)
        _default_tau_rms = {"UMa_NLOS": 363.0, "UMi_NLOS": 129.0, "InF": 56.0}
        self.tau_rms_ns: float = float(
            _dict_get(cfg, "tau_rms_ns", _default_tau_rms.get(self.scenario, 200.0))
        )
        # Rician K-factor: use profile default for LOS models, override if set
        if self._tdl_profile.is_los and self._tdl_profile.k_factor_dB is not None:
            _default_k = self._tdl_profile.k_factor_dB
        else:
            _default_k = -100.0
        self.rician_k_db: float = float(_dict_get(cfg, "rician_k_db", _default_k))

        # -- TDD slot pattern -------------------------------------------------
        from msg_embedding.phy_sim.tdd_config import get_tdd_pattern

        self.tdd_pattern_name: str = str(_dict_get(cfg, "tdd_pattern", "DDDSU"))
        self._tdd_pattern = get_tdd_pattern(self.tdd_pattern_name)

        # -- SRS frequency hopping --------------------------------------------
        self.srs_group_hopping: bool = bool(_dict_get(cfg, "srs_group_hopping", False))
        self.srs_sequence_hopping: bool = bool(_dict_get(cfg, "srs_sequence_hopping", False))
        self.srs_periodicity: int = int(_dict_get(cfg, "srs_periodicity", 10))
        self.srs_b_hop: int = int(_dict_get(cfg, "srs_b_hop", 3))

        # -- SSB measurement --------------------------------------------------
        self.num_ssb_beams: int = int(_dict_get(cfg, "num_ssb_beams", 8))
        self.enable_ssb: bool = bool(_dict_get(cfg, "enable_ssb", True))

        # -- UE height --------------------------------------------------------
        self.ue_height_m: float = float(_dict_get(cfg, "ue_height_m", 1.5))

        # -- Noise figure and thermal noise -----------------------------------
        self.noise_figure_db: float = float(_dict_get(cfg, "noise_figure_db", 7.0))

    # ------------------------------------------------------------------
    # Topology building
    # ------------------------------------------------------------------
    def _build_sites(self) -> list:
        """Build cell-site topology from config."""
        from msg_embedding.topology.hex_grid import CellSite, make_hex_grid
        from msg_embedding.topology.pci_planner import assign_pci_mod3

        if self.custom_site_positions:
            sites = [
                CellSite(
                    site_id=i,
                    position=np.array(
                        [
                            float(p.get("x", 0)),
                            float(p.get("y", 0)),
                            float(p.get("z", self.tx_height_m)),
                        ],
                        dtype=np.float64,
                    ),
                    sector_id=0,
                    azimuth_deg=0.0,
                    tx_height_m=self.tx_height_m,
                    scenario=self.scenario,
                )
                for i, p in enumerate(self.custom_site_positions)
            ]
        else:
            num_rings = _sites_to_rings(self.num_sites)
            sites = make_hex_grid(
                num_rings=num_rings,
                isd_m=self.isd_m,
                sectors=self.sectors_per_site,
                tx_height_m=self.tx_height_m,
                scenario=self.scenario,
            )
        sites = assign_pci_mod3(sites)
        return sites

    # ------------------------------------------------------------------
    # UE placement
    # ------------------------------------------------------------------
    def _place_ues(
        self,
        rng: np.random.Generator,
        sites: list,
        num_ues: int,
    ) -> np.ndarray:
        """Place UEs. Returns shape ``[num_ues, 3]`` float64."""
        if self.custom_ue_positions:
            ue_pos = np.array(
                [
                    [
                        float(p.get("x", 0)),
                        float(p.get("y", 0)),
                        float(p.get("z", self.ue_height_m)),
                    ]
                    for p in self.custom_ue_positions
                ],
                dtype=np.float64,
            )
            # Tile or truncate to match num_ues
            if len(ue_pos) < num_ues:
                repeats = (num_ues // len(ue_pos)) + 1
                ue_pos = np.tile(ue_pos, (repeats, 1))[:num_ues]
            else:
                ue_pos = ue_pos[:num_ues]
            return ue_pos

        # Compute cell radius from ISD
        cell_radius = self.isd_m / math.sqrt(3.0)

        # Centre of the network (origin for hex grid)
        centre = np.zeros(2, dtype=np.float64)

        # Network-wide radius: include all rings
        num_rings = _sites_to_rings(self.num_sites)
        network_radius = max(cell_radius, self.isd_m * num_rings + cell_radius)

        _UE_PLACERS.get(self.ue_distribution, _place_ues_uniform)
        if self.ue_distribution == "clustered":
            return _place_ues_clustered(
                rng,
                num_ues,
                network_radius,
                self.ue_height_m,
                center=centre,
            )
        elif self.ue_distribution == "hotspot":
            return _place_ues_hotspot(
                rng,
                num_ues,
                network_radius,
                self.ue_height_m,
                center=centre,
            )
        else:
            return _place_ues_uniform(
                rng,
                num_ues,
                network_radius,
                self.ue_height_m,
                center=centre,
            )

    # ------------------------------------------------------------------
    # Pathloss computation
    # ------------------------------------------------------------------
    def _compute_pathloss(
        self,
        rng: np.random.Generator,
        bs_pos: np.ndarray,
        ue_pos: np.ndarray,
    ) -> tuple[float, float]:
        """Return ``(pathloss_dB, shadow_dB)`` for one BS→UE link.

        ``bs_pos`` and ``ue_pos`` are 3D position vectors [x, y, z].
        """
        d_3d = float(np.linalg.norm(bs_pos - ue_pos))
        d_3d = max(d_3d, 1.0)  # Minimum distance guard
        fc_ghz = self.carrier_freq_hz / 1e9
        h_bs = float(bs_pos[2])
        h_ut = float(ue_pos[2])

        pl_func = _PATHLOSS_MODELS[self.scenario]
        pl_db, sigma_sf = pl_func(d_3d, fc_ghz, h_bs, h_ut)
        shadow_db = float(rng.normal(0, sigma_sf))
        return pl_db + shadow_db, d_3d

    # ------------------------------------------------------------------
    # Noise floor
    # ------------------------------------------------------------------
    def _noise_power_dbm(self) -> float:
        """Thermal noise power in dBm: kTB + NF."""
        # kT at 290 K in dBm/Hz = -174 dBm/Hz
        k_t_dbm_hz = -174.0
        bw_db = 10.0 * math.log10(self.bandwidth_hz) if self.bandwidth_hz > 0 else 0.0
        return k_t_dbm_hz + bw_db + self.noise_figure_db

    # ------------------------------------------------------------------
    # Channel estimation via the repo pipeline
    # ------------------------------------------------------------------
    def _estimate_channel(
        self,
        h_true: np.ndarray,
        pilots: np.ndarray,
        mode: str,
        snr_db: float,
        rng: np.random.Generator,
        tau_rms_ns_override: float | None = None,
    ) -> np.ndarray:
        """Run channel estimation on h_true using the given pilots.

        ``h_true``: [T, RB, BS_ant, UE_ant] complex64
        ``pilots``: [RB] complex128  (reference signal sequence)
        Returns: [T, RB, BS_ant, UE_ant] complex64
        """
        from msg_embedding.channel_est import estimate_channel

        T, RB, BS, UE = h_true.shape

        if mode == "ideal":
            # Use the pipeline's ideal mode: just return h_true
            # We still go through the pipeline for consistency
            h_true_freq_time = h_true.transpose(1, 0, 2, 3).reshape(RB, T, BS * UE)
            est = estimate_channel(
                Y_rs=np.zeros((RB * T, BS * UE), dtype=np.complex64),
                X_rs=np.ones(RB * T, dtype=np.complex64),
                rs_positions_freq=np.arange(RB, dtype=np.int64),
                rs_positions_time=np.arange(T, dtype=np.int64),
                N_sc=RB,
                N_sym=T,
                mode="ideal",
                h_true=h_true_freq_time,
                dtype="complex64",
            )
            return est.reshape(RB, T, BS, UE).transpose(1, 0, 2, 3).astype(np.complex64)

        # Pilot density matches NR DMRS: every RB in frequency (needed to
        # satisfy Nyquist for typical delay spreads), every 4th OFDM symbol
        # in time (adequate for low-to-medium Doppler).
        pilot_rb_spacing = 1
        pilot_sym_spacing = max(1, min(4, T))
        rs_freq = np.arange(0, RB, pilot_rb_spacing, dtype=np.int64)
        rs_time = np.arange(0, T, pilot_sym_spacing, dtype=np.int64)

        if len(rs_freq) == 0:
            rs_freq = np.array([0], dtype=np.int64)
        if len(rs_time) == 0:
            rs_time = np.array([0], dtype=np.int64)

        n_freq = len(rs_freq)
        n_time = len(rs_time)

        # Extract the true channel at pilot positions
        # h_true: [T, RB, BS, UE]
        h_at_pilots = h_true[np.ix_(rs_time, rs_freq)]  # [n_time, n_freq, BS, UE]
        h_at_pilots = h_at_pilots.transpose(1, 0, 2, 3)  # [n_freq, n_time, BS, UE]

        # Pilot symbols at the RS positions
        pilot_seq = pilots[rs_freq]  # [n_freq]
        # Repeat to match the (freq-slow, time-fast) flattening order used
        # by _gather_rs_grid in the estimation pipeline: position i maps to
        # (f = i // n_time, t = i % n_time), so each freq pilot repeats
        # n_time times consecutively.
        pilot_tiled = np.repeat(pilot_seq, n_time)  # [n_freq * n_time]

        # Received signal at pilots: Y = H * X + noise
        noise_std = 10.0 ** (-snr_db / 20.0) / math.sqrt(2.0)
        noise = noise_std * (
            rng.standard_normal(h_at_pilots.shape) + 1j * rng.standard_normal(h_at_pilots.shape)
        )
        h_noisy = h_at_pilots + noise.astype(np.complex128)

        # Y_rs = H_pilot * X_pilot (element-wise across freq)
        # Shape: [n_freq, n_time, BS, UE] * [n_freq, 1, 1, 1]
        X_expanded = pilot_seq[:, None, None, None]
        Y_at_pilots = h_noisy * X_expanded  # [n_freq, n_time, BS, UE]

        # Flatten for the pipeline: [n_freq * n_time, BS * UE]
        Y_flat = Y_at_pilots.reshape(n_freq * n_time, BS * UE)
        X_flat = pilot_tiled.astype(np.complex128)  # [n_freq * n_time]

        est = estimate_channel(
            Y_rs=Y_flat.astype(np.complex64),
            X_rs=X_flat.astype(np.complex64),
            rs_positions_freq=rs_freq,
            rs_positions_time=rs_time,
            N_sc=RB,
            N_sym=T,
            mode=mode,  # type: ignore[arg-type]
            h_true=h_true.transpose(1, 0, 2, 3).reshape(RB, T, BS * UE),
            pdp_prior={
                "tau_rms": (tau_rms_ns_override or self.tau_rms_ns) * 1e-9,
                "delta_f": 12 * self.subcarrier_spacing * pilot_rb_spacing,
            },
            snr_db=snr_db,
            dtype="complex64",
        )
        return est.reshape(RB, T, BS, UE).transpose(1, 0, 2, 3).astype(np.complex64)

    # ------------------------------------------------------------------
    # Single-sample generation
    # ------------------------------------------------------------------
    def _generate_one_sample(
        self,
        idx: int,
        sites: list,
        *,
        link_override: str | None = None,
        pilot_override: str | None = None,
        lsp_override: dict | None = None,
    ) -> ChannelSample:
        """Generate one physically meaningful ChannelSample."""
        rng = np.random.default_rng(self._seed + idx)

        effective_link = link_override or self.link
        effective_pilot = pilot_override or self.pilot_type
        sample_link: str = effective_link if effective_link in ("UL", "DL") else "DL"

        # Determine antenna dimensions based on link direction
        if sample_link == "DL":
            # DL: BS transmits, UE receives
            BS_ant = self.num_bs_tx_ant
            UE_ant = self.num_ue_rx_ant
        else:
            # UL: UE transmits, BS receives
            BS_ant = self.num_bs_rx_ant
            UE_ant = self.num_ue_tx_ant

        T = self.num_ofdm_symbols
        RB = self._num_rb
        K = len(sites)  # total number of cells (sites * sectors)

        # Doppler frequency from UE speed
        wavelength = 3e8 / self.carrier_freq_hz
        ue_speed_ms = self.ue_speed_kmh / 3.6
        doppler_hz = ue_speed_ms / wavelength

        # Rician K-factor (linear)
        rician_k_linear = 10.0 ** (self.rician_k_db / 10.0) if self.rician_k_db > -50 else 0.0

        # -- Sample per-drop LSPs from 38.901 Table 7.5-6 (Fix 1) -----------
        if lsp_override is not None:
            sample_tau_rms_ns = lsp_override["tau_rms_ns"]
            lsp_override.get("sf_db", 0.0)
        else:
            lsp_params = _LSP_TABLE_38901.get(self.scenario)
            if lsp_params is not None:
                lgDS_mu, lgDS_sigma, _ = lsp_params["lgDS"]
                sample_tau_rms_ns = 10.0 ** rng.normal(lgDS_mu, max(lgDS_sigma, 1e-6)) * 1e9
                lgSF_mu, lgSF_sigma, _ = lsp_params["lgSF"]
                rng.normal(lgSF_mu, max(lgSF_sigma, 1e-6))
            else:
                sample_tau_rms_ns = self.tau_rms_ns

        # -- Per-scenario spatial correlation (Fix 4) -------------------------
        spatial_rho = _SCENARIO_SPATIAL_RHO.get(self.scenario, 0.7)

        # -- LOS AoD/AoA for steering vectors (Fix 2) ------------------------
        los_aod = rng.uniform(-np.pi / 2, np.pi / 2)
        los_aoa = rng.uniform(-np.pi / 2, np.pi / 2)

        # -- Place one UE -----------------------------------------------------
        ue_positions = self._place_ues(rng, sites, 1)
        ue_pos = ue_positions[0]  # [3]

        # -- LOS probability check (Fix 5) ------------------------------------
        # Find serving cell candidate (closest) for LOS check
        _dists_2d = [
            math.sqrt((ue_pos[0] - s.position[0]) ** 2 + (ue_pos[1] - s.position[1]) ** 2)
            for s in sites
        ]
        _closest_idx = int(np.argmin(_dists_2d))
        d_2d_serving = max(_dists_2d[_closest_idx], 1.0)
        p_los = _los_probability(self.scenario, d_2d_serving)
        is_los = rng.random() < p_los

        # Select TDL profile based on LOS/NLOS
        effective_tdl = self._tdl_profile
        effective_k_linear = rician_k_linear
        if is_los and not self._tdl_profile.is_los:
            from msg_embedding.channel_models.tdl import get_tdl_profile

            effective_tdl = get_tdl_profile("TDL-D")
            effective_k_linear = 10.0 ** (effective_tdl.k_factor_dB / 10.0)
        elif not is_los and self._tdl_profile.is_los:
            from msg_embedding.channel_models.tdl import get_tdl_profile

            effective_tdl = get_tdl_profile("TDL-C")
            effective_k_linear = 0.0

        # -- Compute pathloss + channel from every BS to UE -------------------
        noise_power_dbm = self._noise_power_dbm()
        rx_power_dbm = np.zeros(K, dtype=np.float64)
        pl_all = np.zeros(K, dtype=np.float64)
        h_all = []  # list of [T, RB, BS_ant, UE_ant] arrays

        for k, site in enumerate(sites):
            bs_pos = np.asarray(site.position, dtype=np.float64)
            pl_db, d_3d = self._compute_pathloss(rng, bs_pos, ue_pos)
            pl_all[k] = pl_db

            # Received power in dBm
            rx_power_dbm[k] = self.tx_power_dbm - pl_db

            # Generate small-scale fading channel using selected TDL profile
            h_k = _generate_tdl_channel(
                rng=rng,
                num_taps=self.num_taps,
                num_ofdm_sym=T,
                num_rb=RB,
                num_tx_ant=BS_ant,
                num_rx_ant=UE_ant,
                tau_rms_ns=sample_tau_rms_ns,
                subcarrier_spacing_hz=self.subcarrier_spacing,
                doppler_hz=doppler_hz,
                rician_k_linear=effective_k_linear,
                tdl_profile=effective_tdl,
                los_aod_rad=los_aod,
                los_aoa_rad=los_aoa,
                spatial_corr_rho=spatial_rho,
            )

            # Keep channel normalised (unit avg power per element).
            # Pathloss is accounted for in the SNR/SIR/SINR scalars below;
            # the channel matrix represents the small-scale fading structure.
            # For multi-cell relative scaling, apply path-gain *difference*
            # relative to the strongest cell so interferer shapes remain
            # physically consistent while the serving channel stays ~O(1).
            h_all.append(h_k.astype(np.complex64))

        # -- Select serving cell (strongest received power) --------------------
        serving_idx = int(np.argmax(rx_power_dbm))
        serving_pci = sites[serving_idx].pci

        # Apply relative path-gain scaling: serving cell stays unit-power,
        # interferers are scaled by sqrt(P_interf / P_serving) so the
        # channel matrices reflect the relative received-power differences.
        p_serving_linear = 10.0 ** (rx_power_dbm[serving_idx] / 10.0)
        for k in range(K):
            if k != serving_idx:
                p_k_linear = 10.0 ** (rx_power_dbm[k] / 10.0)
                rel_amplitude = math.sqrt(max(p_k_linear / (p_serving_linear + 1e-30), 1e-15))
                h_all[k] = (h_all[k] * rel_amplitude).astype(np.complex64)

        # -- Normalize serving channel to unit power (Fix 6) ------------------
        h_serving = h_all[serving_idx]
        serving_power = float(np.mean(np.abs(h_serving) ** 2))
        if serving_power > 1e-30:
            h_all[serving_idx] = (h_serving / np.sqrt(serving_power)).astype(np.complex64)

        h_serving_true = h_all[serving_idx]  # [T, RB, BS, UE] complex64

        # -- Interferers -------------------------------------------------------
        interferer_indices = [k for k in range(K) if k != serving_idx]
        if len(interferer_indices) > 0:
            h_interferers = np.stack([h_all[k] for k in interferer_indices], axis=0).astype(
                np.complex64
            )  # [K-1, T, RB, BS, UE]
        else:
            h_interferers = None

        # -- SNR / SIR / SINR --------------------------------------------------
        # Serving signal power (dBm)
        p_serving_dbm = rx_power_dbm[serving_idx]

        # SNR = P_serving - N0
        snr_db = p_serving_dbm - noise_power_dbm

        # SIR = P_serving - P_interference (sum of all interferers in linear)
        if len(interferer_indices) > 0:
            p_interf_linear = np.sum(10.0 ** (rx_power_dbm[interferer_indices] / 10.0))
            p_interf_dbm = 10.0 * math.log10(max(p_interf_linear, 1e-30))
            sir_db = p_serving_dbm - p_interf_dbm
        else:
            sir_db = 49.9  # No interference

        # SINR = -10 log10(10^(-SNR/10) + 10^(-SIR/10))
        sinr_db = -10.0 * math.log10(10.0 ** (-snr_db / 10.0) + 10.0 ** (-sir_db / 10.0))

        # Clamp to contract bounds
        snr_db = _clamp_db(snr_db)
        sir_db = _clamp_db(sir_db)
        sinr_db = _clamp_db(sinr_db)

        # -- Generate reference signals ----------------------------------------
        if effective_pilot == "srs_zc":
            pilots = _generate_pilots_srs(RB, serving_pci)
        else:
            pilots = _generate_pilots_csirs(RB, serving_pci)

        # -- Channel estimation ------------------------------------------------
        h_serving_est = self._estimate_channel(
            h_serving_true,
            pilots,
            self.channel_est_mode,
            snr_db,
            rng,
            tau_rms_ns_override=sample_tau_rms_ns,
        )

        # -- Interference signal (observed at receiver) ------------------------
        # Sum of interferer channels * pilot (simplified model)
        interference_signal: np.ndarray | None = None
        if h_interferers is not None:
            # interference_signal: [T, RB, BS_ant] — sum over interferers & UE_ant
            # We model it as the sum of all interferer channels multiplied by
            # their respective reference signals, observed across BS antennas.
            # Shape: [T, RB * UE_ant] → [T, N_RE_obs]
            interf_sum = np.sum(h_interferers, axis=0)  # [T, RB, BS, UE]
            # Flatten the last two dims to get [T, RB * BS * UE] → too large,
            # use [T, RB, UE] by summing over BS
            interf_signal_raw = np.sum(interf_sum, axis=2)  # [T, RB, UE]
            interference_signal = interf_signal_raw.astype(np.complex64)

        # -- SSB measurements (multi-cell beam sweep) --------------------------
        ssb_rsrp: list[float] | None = None
        ssb_rsrq: list[float] | None = None
        ssb_sinr: list[float] | None = None
        ssb_best_beam: list[int] | None = None
        ssb_pcis_list: list[int] | None = None

        if self.enable_ssb and K >= 1:
            from msg_embedding.phy_sim.ssb_measurement import SSBMeasurement

            ssb_meas = SSBMeasurement(
                num_beams=self.num_ssb_beams,
                num_bs_ant=BS_ant,
            )
            all_pcis = [s.pci for s in sites]
            ssb_result = ssb_meas.measure(
                h_per_cell=h_all,
                pcis=all_pcis,
                noise_power_lin=10.0 ** (noise_power_dbm / 10.0) * 1e-3,
            )
            ssb_rsrp = ssb_result.rsrp_dBm.tolist()
            ssb_rsrq = ssb_result.rsrq_dB.tolist()
            ssb_sinr = ssb_result.ss_sinr_dB.tolist()
            ssb_best_beam = ssb_result.best_beam_idx.tolist()
            ssb_pcis_list = ssb_result.pcis

        # -- Assemble meta dict ------------------------------------------------
        meta = {
            "num_sites": self.num_sites,
            "num_cells": K,
            "isd_m": self.isd_m,
            "sectors_per_site": self.sectors_per_site,
            "num_bs_tx_ant": self.num_bs_tx_ant,
            "num_bs_rx_ant": self.num_bs_rx_ant,
            "num_ue_tx_ant": self.num_ue_tx_ant,
            "num_ue_rx_ant": self.num_ue_rx_ant,
            "carrier_freq_hz": self.carrier_freq_hz,
            "bandwidth_hz": self.bandwidth_hz,
            "subcarrier_spacing": self.subcarrier_spacing,
            "tx_power_dbm": self.tx_power_dbm,
            "ue_speed_kmh": self.ue_speed_kmh,
            "ue_distribution": self.ue_distribution,
            "pilot_type": effective_pilot,
            "scenario": self.scenario,
            "channel_model": self.channel_model_name,
            "tdd_pattern": self.tdd_pattern_name,
            "num_taps": self.num_taps,
            "tau_rms_ns": self.tau_rms_ns,
            "rician_k_db": self.rician_k_db,
            "doppler_hz": doppler_hz,
            "srs_group_hopping": self.srs_group_hopping,
            "srs_sequence_hopping": self.srs_sequence_hopping,
            "srs_periodicity": self.srs_periodicity,
            "srs_b_hop": self.srs_b_hop,
            "num_ofdm_symbols": T,
            "num_rb": RB,
            "noise_figure_db": self.noise_figure_db,
            "noise_power_dbm": noise_power_dbm,
            "serving_cell_index": serving_idx,
            "serving_pci": serving_pci,
            "pathloss_serving_db": float(pl_all[serving_idx]),
            "pathloss_all_db": [float(pl_all[k]) for k in range(K)],
            "rx_power_serving_dbm": float(p_serving_dbm),
            "rx_power_all_dbm": [float(rx_power_dbm[k]) for k in range(K)],
            "distance_3d_m": float(
                np.linalg.norm(np.asarray(sites[serving_idx].position) - ue_pos)
            ),
            "sample_tau_rms_ns": float(sample_tau_rms_ns),
            "is_los": bool(is_los),
            "los_probability": float(p_los),
            "custom_positions": self.custom_site_positions is not None,
            "num_ues": self.num_ues,
            "sample_index": int(idx),
            "seed": self._seed,
            "mock": False,
        }

        return ChannelSample(
            h_serving_true=h_serving_true,
            h_serving_est=h_serving_est,
            h_interferers=h_interferers,
            interference_signal=interference_signal,
            noise_power_dBm=noise_power_dbm,
            snr_dB=snr_db,
            sir_dB=sir_db if len(interferer_indices) > 0 else None,
            sinr_dB=sinr_db,
            ssb_rsrp_dBm=ssb_rsrp,
            ssb_rsrq_dB=ssb_rsrq,
            ssb_sinr_dB=ssb_sinr,
            ssb_best_beam_idx=ssb_best_beam,
            ssb_pcis=ssb_pcis_list,
            link=sample_link,  # type: ignore[arg-type]
            channel_est_mode=self.channel_est_mode,  # type: ignore[arg-type]
            serving_cell_id=serving_pci,
            ue_position=ue_pos.astype(np.float64),
            channel_model=self.channel_model_name,
            tdd_pattern=self.tdd_pattern_name,
            source="internal_sim",
            sample_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc),
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Trajectory LSP generation (Fix 9)
    # ------------------------------------------------------------------
    def _generate_trajectory_lsps(
        self,
        rng: np.random.Generator,
        positions: np.ndarray,
    ) -> list[dict]:
        """Generate spatially correlated LSPs for a trajectory.

        Uses exponential spatial correlation: corr(d) = exp(-d / d_decorr).
        positions: [N, 3] array of UE positions.
        Returns: list of N dicts, each with 'tau_rms_ns', 'sf_db', etc.
        """
        N = len(positions)
        lsp_params = _LSP_TABLE_38901.get(self.scenario)
        if lsp_params is None or N == 0:
            return [{"tau_rms_ns": self.tau_rms_ns, "sf_db": 0.0} for _ in range(N)]

        # Compute pairwise distances
        diffs = positions[:, None, :2] - positions[None, :, :2]  # [N, N, 2]
        dist_matrix = np.sqrt(np.sum(diffs**2, axis=-1))  # [N, N]

        lsp_values: dict[str, np.ndarray] = {}

        for param_name in ["lgDS", "lgSF"]:
            mu, sigma, d_decorr = lsp_params[param_name]
            if sigma < 1e-6 or d_decorr < 1e-6:
                lsp_values[param_name] = np.full(N, mu)
                continue
            # Spatial correlation matrix
            R = np.exp(-dist_matrix / max(d_decorr, 1.0))
            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(R + 1e-6 * np.eye(N))
            except np.linalg.LinAlgError:
                L = np.eye(N)
            # Generate correlated samples
            z = rng.standard_normal(N)
            correlated = L @ z
            lsp_values[param_name] = mu + sigma * correlated

        results = []
        for i in range(N):
            tau_rms_ns = 10.0 ** lsp_values["lgDS"][i] * 1e9
            results.append(
                {
                    "tau_rms_ns": float(np.clip(tau_rms_ns, 1.0, 10000.0)),
                    "sf_db": float(lsp_values.get("lgSF", np.zeros(N))[i]),
                }
            )

        return results

    # ------------------------------------------------------------------
    # Iterator interface
    # ------------------------------------------------------------------
    def iter_samples(self) -> Iterator[ChannelSample]:
        """Yield :class:`ChannelSample` instances from the internal simulator."""
        sites = self._build_sites()

        # Pre-generate spatially correlated LSPs for trajectory mode (Fix 9)
        trajectory_lsps: list[dict] | None = None
        if self.custom_ue_positions and len(self.custom_ue_positions) > 1:
            positions = np.array(
                [
                    [
                        float(p.get("x", 0)),
                        float(p.get("y", 0)),
                        float(p.get("z", self.ue_height_m)),
                    ]
                    for p in self.custom_ue_positions
                ]
            )
            traj_rng = np.random.default_rng(self._seed)
            trajectory_lsps = self._generate_trajectory_lsps(traj_rng, positions)

        for idx in range(self.num_samples):
            lsp_override = None
            if trajectory_lsps is not None and idx < len(trajectory_lsps):
                lsp_override = trajectory_lsps[idx]

            if self.link == "BOTH":
                # -- TDD Reciprocity (Fix 3) ----------------------------------
                # Generate DL sample first
                dl_sample = self._generate_one_sample(
                    idx,
                    sites,
                    link_override="DL",
                    pilot_override=self.pilot_type_dl,
                    lsp_override=lsp_override,
                )
                # TDD reciprocity: UL channel = conjugate transpose of DL channel
                # H_UL[t, rb, ue_ant, bs_ant] = H_DL[t, rb, bs_ant, ue_ant]^H
                h_ul_true = np.conj(dl_sample.h_serving_true.transpose(0, 1, 3, 2))
                # Add small independent noise for UL estimation imperfections
                rng_ul = np.random.default_rng(self._seed + idx + self.num_samples)
                noise_scale = 0.01
                h_ul_true = (
                    h_ul_true
                    + noise_scale
                    * (
                        rng_ul.standard_normal(h_ul_true.shape)
                        + 1j * rng_ul.standard_normal(h_ul_true.shape)
                    )
                    / math.sqrt(2.0)
                ).astype(np.complex64)

                # Generate UL pilots and estimate
                ul_pilots = _generate_pilots_srs(self._num_rb, dl_sample.serving_cell_id)
                h_ul_est = self._estimate_channel(
                    h_ul_true,
                    ul_pilots,
                    self.channel_est_mode,
                    dl_sample.snr_dB,
                    rng_ul,
                )

                # Construct UL sample reusing DL metadata
                ul_sample = ChannelSample(
                    h_serving_true=h_ul_true,
                    h_serving_est=h_ul_est,
                    h_interferers=(
                        None
                        if dl_sample.h_interferers is None
                        else np.conj(dl_sample.h_interferers.transpose(0, 1, 2, 4, 3)).astype(
                            np.complex64
                        )
                    ),
                    interference_signal=dl_sample.interference_signal,
                    noise_power_dBm=dl_sample.noise_power_dBm,
                    snr_dB=dl_sample.snr_dB,
                    sir_dB=dl_sample.sir_dB,
                    sinr_dB=dl_sample.sinr_dB,
                    ssb_rsrp_dBm=dl_sample.ssb_rsrp_dBm,
                    ssb_rsrq_dB=dl_sample.ssb_rsrq_dB,
                    ssb_sinr_dB=dl_sample.ssb_sinr_dB,
                    ssb_best_beam_idx=dl_sample.ssb_best_beam_idx,
                    ssb_pcis=dl_sample.ssb_pcis,
                    link="UL",
                    channel_est_mode=dl_sample.channel_est_mode,
                    serving_cell_id=dl_sample.serving_cell_id,
                    ue_position=dl_sample.ue_position,
                    channel_model=dl_sample.channel_model,
                    tdd_pattern=dl_sample.tdd_pattern,
                    source="internal_sim",
                    sample_id=str(uuid.uuid4()),
                    created_at=datetime.now(timezone.utc),
                    meta={**dl_sample.meta, "tdd_reciprocal": True},
                )
                yield dl_sample
                yield ul_sample
            else:
                yield self._generate_one_sample(
                    idx,
                    sites,
                    lsp_override=lsp_override,
                )

    # ------------------------------------------------------------------
    # Describe
    # ------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """Return metadata describing the source configuration."""
        K = self.num_sites * self.sectors_per_site
        multiplier = 2 if self.link == "BOTH" else 1
        return {
            "source": "internal_sim",
            "scenario": self.scenario,
            "num_sites": self.num_sites,
            "num_cells": K,
            "num_ues": self.num_ues,
            "expected_sample_count": self.num_samples * multiplier,
            "pilot_type": self.pilot_type,
            "pilot_type_dl": self.pilot_type_dl,
            "pilot_type_ul": self.pilot_type_ul,
            "channel_est_mode": self.channel_est_mode,
            "link": self.link,
            "dimensions": {
                "T": self.num_ofdm_symbols,
                "RB": self._num_rb,
                "BS_ant": self.num_bs_ant,
                "UE_ant": self.num_ue_ant,
                "K": K,
            },
            "topology": {
                "isd_m": self.isd_m,
                "sectors_per_site": self.sectors_per_site,
                "tx_height_m": self.tx_height_m,
                "custom_positions": self.custom_site_positions is not None,
            },
            "antenna": {
                "num_bs_tx_ant": self.num_bs_tx_ant,
                "num_bs_rx_ant": self.num_bs_rx_ant,
                "num_ue_tx_ant": self.num_ue_tx_ant,
                "num_ue_rx_ant": self.num_ue_rx_ant,
            },
            "carrier_freq_hz": self.carrier_freq_hz,
            "bandwidth_hz": self.bandwidth_hz,
            "tx_power_dbm": self.tx_power_dbm,
            "ue_speed_kmh": self.ue_speed_kmh,
            "ue_distribution": self.ue_distribution,
            "channel_model": {
                "num_taps": self.num_taps,
                "tau_rms_ns": self.tau_rms_ns,
                "rician_k_db": self.rician_k_db,
            },
            "mock": False,
        }


__all__ = ["InternalSimSource"]
