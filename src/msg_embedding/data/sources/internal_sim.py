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
    fading_seed: int | None = None,
    t_offset_s: float = 0.0,
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
    # When fading_seed is set, use a deterministic RNG so that tap gains
    # and Doppler SoS parameters are identical across snapshots.
    fading_rng = np.random.default_rng(fading_seed) if fading_seed is not None else rng
    h_taps_real = fading_rng.standard_normal((L, 1, N_tx, N_rx))
    h_taps_imag = fading_rng.standard_normal((L, 1, N_tx, N_rx))
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
        t_axis = t_offset_s + np.arange(T, dtype=np.float64) * T_sym  # [T]

        for l_idx in range(L):
            # Sum-of-sinusoids: sum_n cos(2*pi*f_d*cos(alpha_n)*t + phi_n)
            alpha_n = fading_rng.uniform(0, 2 * np.pi, size=N_sinusoids)  # arrival angles
            phi_n = fading_rng.uniform(0, 2 * np.pi, size=N_sinusoids)  # random phases
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
    symbol: int = 0,
    group_hopping: bool = False,
    sequence_hopping: bool = False,
    K_TC: int = 2,
    srs_num_rb: int | None = None,
) -> np.ndarray:
    """Generate SRS pilot symbols per 3GPP TS 38.211 §6.4.1.4.

    Uses the full NR SRS sequence with group/sequence hopping when enabled.

    Parameters
    ----------
    num_rb : int
        Total number of RBs in the carrier (used as fallback).
    srs_num_rb : int or None
        Number of RBs covered by SRS in this slot (from SRS bandwidth config).
        When provided, the sequence length matches the SRS bandwidth, not full band.

    Returns shape ``[effective_rb]`` complex128, where effective_rb = srs_num_rb or num_rb.
    """
    from msg_embedding.ref_signals.srs import srs_sequence

    effective_rb = srs_num_rb if srs_num_rb is not None else num_rb
    # Msc is the number of SRS subcarriers: m_SRS_b * 12 / K_TC.
    # Minimum valid Msc for ZC/CG tables is 6.
    Msc = max(effective_rb * 12 // K_TC, 6)
    n_SRS_ID = cell_id % 1024
    seq = srs_sequence(
        n_SRS_ID=n_SRS_ID,
        K_TC=K_TC,
        n_cs=0,
        N_ap=1,
        Msc=Msc,
        slot=slot,
        symbol=symbol,
        n_ap_index=0,
        group_hopping=group_hopping,
        sequence_hopping=sequence_hopping,
    )
    # Downsample from subcarrier level to one value per RB
    sc_per_rb = 12 // K_TC
    if sc_per_rb > 1 and len(seq) > effective_rb:
        seq = seq[::sc_per_rb][:effective_rb]
    if len(seq) < effective_rb:
        pad = np.ones(effective_rb - len(seq), dtype=np.complex128)
        seq = np.concatenate([seq, pad])
    else:
        seq = seq[:effective_rb]
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
        self.num_interfering_ues: int = int(_dict_get(cfg, "num_interfering_ues", max(self.num_ues - 1, 0)))
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
        # Number of RBs: 3GPP TS 38.101 standard table lookup
        from msg_embedding.phy_sim.nr_rb_table import nr_rb_lookup

        _cfg_num_rb = _dict_get(cfg, "num_rb", None)
        if _cfg_num_rb is not None:
            self._num_rb: int = int(_cfg_num_rb)
        else:
            self._num_rb = nr_rb_lookup(self.bandwidth_hz, self.subcarrier_spacing)

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
        self.mobility_mode: str = str(_dict_get(cfg, "mobility_mode", "static"))
        from ._mobility import MOBILITY_MODES
        if self.mobility_mode not in MOBILITY_MODES:
            raise ValueError(
                f"Unknown mobility_mode {self.mobility_mode!r}; expected "
                f"one of {MOBILITY_MODES}."
            )
        self.sample_interval_s: float = float(_dict_get(cfg, "sample_interval_s", 0.5e-3))
        self.ue_distribution: str = str(_dict_get(cfg, "ue_distribution", "uniform"))
        if self.ue_distribution not in {"uniform", "clustered", "hotspot"}:
            raise ValueError(
                f"Unknown ue_distribution {self.ue_distribution!r}; expected "
                "'uniform' / 'clustered' / 'hotspot'."
            )

        # -- Pilot / estimation / link ----------------------------------------
        _pilot_fallback = str(_dict_get(cfg, "pilot_type", "csi_rs_gold"))
        self.pilot_type_dl: str = str(_dict_get(cfg, "pilot_type_dl", _pilot_fallback))
        self.pilot_type_ul: str = str(_dict_get(cfg, "pilot_type_ul",
                                                 _pilot_fallback if _pilot_fallback == "srs_zc" else "srs_zc"))
        if self.pilot_type_dl not in {"srs_zc", "csi_rs_gold"}:
            raise ValueError(f"Unknown pilot_type_dl {self.pilot_type_dl!r}; expected 'srs_zc' / 'csi_rs_gold'.")
        if self.pilot_type_ul not in {"srs_zc", "csi_rs_gold"}:
            raise ValueError(f"Unknown pilot_type_ul {self.pilot_type_ul!r}; expected 'srs_zc' / 'csi_rs_gold'.")
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
                "UMi_NLOS": 33.0,
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
        if self.srs_group_hopping and self.srs_sequence_hopping:
            raise ValueError("srs_group_hopping and srs_sequence_hopping are mutually exclusive")
        self.srs_periodicity: int = int(_dict_get(cfg, "srs_periodicity", 10))
        from msg_embedding.ref_signals.srs import SRS_PERIODICITY_TABLE, SRSResourceConfig

        if self.srs_periodicity not in SRS_PERIODICITY_TABLE:
            raise ValueError(
                f"srs_periodicity={self.srs_periodicity} not in "
                f"3GPP TS 38.211 Table 6.4.1.4.4-1: {SRS_PERIODICITY_TABLE}"
            )
        self.srs_b_hop: int = int(_dict_get(cfg, "srs_b_hop", 0))
        self.srs_comb: int = int(_dict_get(cfg, "srs_comb", 2))
        if self.srs_comb not in (2, 4, 8):
            raise ValueError(f"srs_comb (K_TC) must be in {{2, 4, 8}}, got {self.srs_comb}")

        # SRS bandwidth configuration (38.211 Table 6.4.1.4.3-1)
        self.srs_c_srs: int = int(_dict_get(cfg, "srs_c_srs", 3))
        self.srs_b_srs: int = int(_dict_get(cfg, "srs_b_srs", 1))
        self.srs_n_rrc: int = int(_dict_get(cfg, "srs_n_rrc", 0))
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
        sf_override_db: float | None = None,
    ) -> tuple[float, float]:
        """Return ``(pathloss_dB, d_3d)`` for one BS→UE link.

        ``bs_pos`` and ``ue_pos`` are 3D position vectors [x, y, z].
        ``sf_override_db``: if set, use this shadow fading value instead
        of drawing a new one from ``rng``.
        """
        d_3d = float(np.linalg.norm(bs_pos - ue_pos))
        d_3d = max(d_3d, 1.0)  # Minimum distance guard
        fc_ghz = self.carrier_freq_hz / 1e9
        h_bs = float(bs_pos[2])
        h_ut = float(ue_pos[2])

        pl_func = _PATHLOSS_MODELS[self.scenario]
        pl_db, sigma_sf = pl_func(d_3d, fc_ghz, h_bs, h_ut)
        if sf_override_db is not None:
            shadow_db = sf_override_db
        else:
            shadow_db = float(rng.normal(0, sigma_sf))
        return pl_db + shadow_db, d_3d

    # ------------------------------------------------------------------
    # Noise floor
    # ------------------------------------------------------------------
    def _noise_power_dbm(self) -> float:
        """Per-RB thermal noise power in dBm: kT·(12·SCS) + NF."""
        k_t_dbm_hz = -174.0
        bw_per_rb = 12.0 * self.subcarrier_spacing
        bw_db = 10.0 * math.log10(bw_per_rb) if bw_per_rb > 0 else 0.0
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
        valid_symbol_mask: np.ndarray | None = None,
        srs_rb_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run channel estimation on h_true using the given pilots.

        ``h_true``: [T, RB, BS_ant, UE_ant] complex64
        ``pilots``: [RB] complex128  (reference signal sequence)
        ``valid_symbol_mask``: [T] bool — TDD-aware mask of symbols where
            pilots can be placed (DL symbols for CSI-RS, UL for SRS).
            If None, all symbols are valid.
        Returns: [T, RB, BS_ant, UE_ant] complex64
        """
        from msg_embedding.channel_est import estimate_channel

        T, RB, BS, UE = h_true.shape

        if mode == "ideal":
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
            h_est = est.reshape(RB, T, BS, UE).transpose(1, 0, 2, 3).astype(np.complex64)
            if valid_symbol_mask is not None:
                guard_mask = ~(valid_symbol_mask | np.roll(valid_symbol_mask, 0))
                for t_idx in range(T):
                    if not valid_symbol_mask[t_idx] and not np.any(valid_symbol_mask):
                        h_est[t_idx] = 0.0
            return h_est

        # Pilot placement: SRS RBs if provided, else full band
        if srs_rb_indices is not None and len(srs_rb_indices) > 0:
            rs_freq = srs_rb_indices.astype(np.int64)
        else:
            pilot_rb_spacing = 1
            rs_freq = np.arange(0, RB, pilot_rb_spacing, dtype=np.int64)
        pilot_sym_spacing = max(1, min(4, T))
        candidate_times = np.arange(0, T, pilot_sym_spacing, dtype=np.int64)
        if valid_symbol_mask is not None:
            valid_times = np.where(valid_symbol_mask)[0].astype(np.int64)
            if len(valid_times) > 0:
                rs_time = np.intersect1d(candidate_times, valid_times)
                if len(rs_time) == 0:
                    rs_time = valid_times[::max(1, len(valid_times) // 4)]
            else:
                rs_time = candidate_times
        else:
            rs_time = candidate_times

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
        if srs_rb_indices is not None and len(pilots) == n_freq:
            pilot_seq = pilots  # already sized for SRS bandwidth
        else:
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
                "delta_f": 12 * self.subcarrier_spacing * 1,
            },
            snr_db=snr_db,
            dtype="complex64",
        )
        h_est = est.reshape(RB, T, BS, UE).transpose(1, 0, 2, 3).astype(np.complex64)
        return h_est

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
        paired: bool = False,
        ue_pos_override: np.ndarray | None = None,
        doppler_hz_override: float | None = None,
        fading_seed: int | None = None,
        snapshot_t_offset_s: float = 0.0,
    ) -> ChannelSample:
        """Generate one physically meaningful ChannelSample."""
        rng = np.random.default_rng(self._seed + idx)

        effective_link = link_override or self.link
        sample_link: str = effective_link if effective_link in ("UL", "DL") else "DL"
        effective_pilot = pilot_override or (
            self.pilot_type_ul if sample_link == "UL" else self.pilot_type_dl
        )

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

        # -- TDD slot resolution per 3GPP TS 38.213 §11.1 ----------------------
        tdd = self._tdd_pattern
        slot_idx = idx % tdd.period_slots
        slot_direction = tdd.slot_type(slot_idx)
        symbol_map = tdd.symbol_map(slot_idx)
        dl_symbol_mask = np.array([s == "D" for s in symbol_map], dtype=bool)
        ul_symbol_mask = np.array([s == "U" for s in symbol_map], dtype=bool)
        guard_symbol_mask = np.array([s == "G" for s in symbol_map], dtype=bool)

        # SRS periodicity: check if this slot carries SRS
        srs_active_in_slot = (slot_idx % self.srs_periodicity) == 0

        # SRS frequency-domain position (bandwidth tree + hopping)
        # Accumulate across a full hopping cycle so UL estimation covers m_SRS[0] RBs
        from msg_embedding.ref_signals.srs import srs_accumulated_rb_indices

        ul_syms_for_srs = [i for i, m in enumerate(symbol_map) if m == "U"]
        _srs_sym = ul_syms_for_srs[-1] if ul_syms_for_srs else 0
        srs_rb_idx = srs_accumulated_rb_indices(self._srs_resource_cfg, idx, _srs_sym, RB)

        # Doppler frequency: prefer trajectory-derived value, then config-based
        wavelength = 3e8 / self.carrier_freq_hz
        if doppler_hz_override is not None:
            doppler_hz = doppler_hz_override
        else:
            ue_speed_ms = self.ue_speed_kmh / 3.6
            doppler_hz = ue_speed_ms / wavelength

        # Rician K-factor (linear)
        rician_k_linear = 10.0 ** (self.rician_k_db / 10.0) if self.rician_k_db > -50 else 0.0

        # -- Sample per-drop LSPs from 38.901 Table 7.5-6 (Fix 1) -----------
        sf_override_db: float | None = None
        if lsp_override is not None:
            sample_tau_rms_ns = lsp_override["tau_rms_ns"]
            sf_override_db = lsp_override.get("sf_db", None)
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
        # For mobility sequences, use deterministic fading RNG so that LOS
        # angles and LOS decision are consistent across snapshots.
        env_rng = np.random.default_rng(fading_seed) if fading_seed is not None else rng
        los_aod = env_rng.uniform(-np.pi / 2, np.pi / 2)
        los_aoa = env_rng.uniform(-np.pi / 2, np.pi / 2)

        # -- Place one UE -----------------------------------------------------
        if ue_pos_override is not None:
            ue_pos = ue_pos_override.copy()
        else:
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
        is_los = env_rng.random() < p_los

        # Select TDL profile based on LOS/NLOS
        effective_tdl = self._tdl_profile
        effective_k_linear = rician_k_linear
        if is_los and not self._tdl_profile.is_los:
            from msg_embedding.channel_models.tdl import get_tdl_profile

            effective_tdl = get_tdl_profile("TDL-D")
            effective_k_linear = 10.0 ** ((effective_tdl.k_factor_dB or 0.0) / 10.0)
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
            pl_db, d_3d = self._compute_pathloss(rng, bs_pos, ue_pos, sf_override_db=sf_override_db)
            pl_all[k] = pl_db

            # Received power in dBm
            rx_power_dbm[k] = self.tx_power_dbm - pl_db

            # Generate small-scale fading channel using selected TDL profile
            cell_fading_seed = (fading_seed + k) if fading_seed is not None else None
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
                fading_seed=cell_fading_seed,
                t_offset_s=snapshot_t_offset_s,
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

        # -- Interferers (DL: H(BS_k → target UE), already in h_all) ----------
        interferer_indices = [k for k in range(K) if k != serving_idx]
        if len(interferer_indices) > 0:
            h_interferers = np.stack([h_all[k] for k in interferer_indices], axis=0).astype(
                np.complex64
            )  # [K-1, T, RB, BS, UE]
        else:
            h_interferers = None

        # -- UL interferers: independent per-UE channels H(UE_kn → BS_serving) -
        h_ul_intf_per_ue: list[np.ndarray] | None = None
        if len(interferer_indices) > 0 and self.num_interfering_ues > 0:
            cell_radius = self.isd_m / math.sqrt(3.0)
            serving_bs_pos = np.asarray(sites[serving_idx].position, dtype=np.float64)
            h_ul_intf_per_ue = []
            for ki, k in enumerate(interferer_indices):
                intf_bs_pos = np.asarray(sites[k].position, dtype=np.float64)
                intf_ue_positions = _place_ues_uniform(
                    rng, self.num_interfering_ues, cell_radius,
                    self.ue_height_m, center=intf_bs_pos[:2],
                )
                ue_channels = []
                for n in range(self.num_interfering_ues):
                    ue_n_pos = intf_ue_positions[n]
                    pl_db_n, _ = self._compute_pathloss(rng, serving_bs_pos, ue_n_pos)
                    rx_pwr_n = self.tx_power_dbm - pl_db_n
                    rel_amp_n = math.sqrt(
                        max(10.0 ** (rx_pwr_n / 10.0) / (p_serving_linear + 1e-30), 1e-15)
                    )
                    ue_seed = (fading_seed + K + ki * self.num_interfering_ues + n) if fading_seed is not None else None
                    h_ue_n = _generate_tdl_channel(
                        rng=rng,
                        num_taps=self.num_taps,
                        num_ofdm_sym=T,
                        num_rb=RB,
                        num_tx_ant=UE_ant,
                        num_rx_ant=BS_ant,
                        tau_rms_ns=sample_tau_rms_ns,
                        subcarrier_spacing_hz=self.subcarrier_spacing,
                        doppler_hz=doppler_hz,
                        rician_k_linear=effective_k_linear,
                        tdl_profile=effective_tdl,
                        los_aod_rad=los_aod,
                        los_aoa_rad=los_aoa,
                        spatial_corr_rho=spatial_rho,
                        fading_seed=ue_seed,
                        t_offset_s=snapshot_t_offset_s,
                    )
                    h_ue_n = (h_ue_n * rel_amp_n).astype(np.complex64)
                    ue_channels.append(h_ue_n)
                h_ul_intf_per_ue.append(
                    np.stack(ue_channels, axis=0)  # [N_ues, T, RB, UE_ant, BS_ant]
                )

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

        # -- Interference-aware channel estimation ----------------------------
        from msg_embedding.data.sources._interference_estimation import (
            estimate_channel_with_interference,
            estimate_paired_channels,
        )

        intf_cell_ids = (
            [sites[k].pci for k in interferer_indices] if interferer_indices else None
        )
        n_intf_ues = self.num_interfering_ues

        h_ul_true_out: np.ndarray | None = None
        h_ul_est_out: np.ndarray | None = None
        h_dl_true_out: np.ndarray | None = None
        h_dl_est_out: np.ndarray | None = None
        ul_sir_db_out: float | None = None
        dl_sir_db_out: float | None = None
        n_intf_ues_out: int | None = None
        link_pairing = "single"

        if paired:
            paired_result = estimate_paired_channels(
                h_dl_true=h_serving_true,
                h_interferers_dl=h_interferers,
                serving_cell_id=serving_pci,
                interferer_cell_ids=intf_cell_ids,
                snr_dB=snr_db,
                rng=rng,
                est_mode=self.channel_est_mode,
                tau_rms_ns=sample_tau_rms_ns,
                subcarrier_spacing=self.subcarrier_spacing,
                num_interfering_ues=n_intf_ues,
                srs_group_hopping=self.srs_group_hopping,
                srs_sequence_hopping=self.srs_sequence_hopping,
                srs_comb=self.srs_comb,
                slot_idx=slot_idx,
                dl_symbol_mask=dl_symbol_mask,
                ul_symbol_mask=ul_symbol_mask,
                srs_rb_indices=srs_rb_idx,
                h_interferers_ul_per_ue=h_ul_intf_per_ue,
            )
            h_serving_est = paired_result["h_dl_est"]
            h_ul_true_out = paired_result["h_ul_true"]
            h_ul_est_out = paired_result["h_ul_est"]
            h_dl_true_out = paired_result["h_dl_true"]
            h_dl_est_out = paired_result["h_dl_est"]
            ul_sir_db_out = paired_result["ul_sir_dB"]
            dl_sir_db_out = paired_result["dl_sir_dB"]
            n_intf_ues_out = paired_result["num_interfering_ues"]
            link_pairing = "paired"
            sample_link = "DL"
        else:
            # Find the first valid pilot symbol for the current TDD direction
            _srs_rb = None  # SRS RB indices for UL, None for DL (full band)
            if effective_pilot == "srs_zc":
                ul_syms = [i for i, m in enumerate(symbol_map) if m == "U"]
                pilot_symbol = ul_syms[-1] if ul_syms else 0
                _srs_rb = srs_rb_idx
                pilots = _generate_pilots_srs(
                    RB,
                    serving_pci,
                    slot=slot_idx,
                    symbol=pilot_symbol,
                    group_hopping=self.srs_group_hopping,
                    sequence_hopping=self.srs_sequence_hopping,
                    K_TC=self.srs_comb,
                    srs_num_rb=len(_srs_rb),
                )
            else:
                dl_syms = [i for i, m in enumerate(symbol_map) if m == "D"]
                pilot_symbol = dl_syms[0] if dl_syms else 0
                pilots = _generate_pilots_csirs(RB, serving_pci, slot=slot_idx, symbol=pilot_symbol)

            direction: str = "UL" if sample_link == "UL" else "DL"
            _sym_mask = ul_symbol_mask if direction == "UL" else dl_symbol_mask
            est_result = estimate_channel_with_interference(
                h_serving_true=h_serving_true,
                h_interferers=h_interferers,
                pilots_serving=pilots,
                interferer_cell_ids=intf_cell_ids,
                direction=direction,  # type: ignore[arg-type]
                snr_dB=snr_db,
                rng=rng,
                est_mode=self.channel_est_mode,
                tau_rms_ns=sample_tau_rms_ns,
                subcarrier_spacing=self.subcarrier_spacing,
                serving_cell_id=serving_pci,
                num_interfering_ues=n_intf_ues,
                valid_symbol_mask=_sym_mask,
                srs_rb_indices=_srs_rb,
                srs_slot=slot_idx,
                srs_symbol=pilot_symbol if direction == "UL" else 0,
                srs_group_hopping=self.srs_group_hopping,
                srs_sequence_hopping=self.srs_sequence_hopping,
                srs_K_TC=self.srs_comb,
                srs_num_rb=len(_srs_rb) if _srs_rb is not None else None,
                h_interferers_ul_per_ue=h_ul_intf_per_ue if direction == "UL" else None,
            )
            h_serving_est = est_result.h_est
            if est_result.sir_dB is not None:
                sir_db = _clamp_db(est_result.sir_dB)
                sinr_db = _clamp_db(
                    -10.0 * math.log10(10.0 ** (-snr_db / 10.0) + 10.0 ** (-sir_db / 10.0))
                )

        # -- DL precoding weights from UL estimate (SRS-based beamforming) ---
        from msg_embedding.phy_sim.precoding import compute_dl_precoding

        w_dl_out: np.ndarray | None = None
        rank_out: int | None = None
        _max_r = min(4, UE_ant, BS_ant)
        if paired and h_ul_est_out is not None:
            prec = compute_dl_precoding(h_ul_est_out, max_rank=_max_r)
            w_dl_out = prec.w_dl
            rank_out = prec.rank
        elif sample_link == "UL" and h_serving_est is not None:
            prec = compute_dl_precoding(h_serving_est, max_rank=_max_r)
            w_dl_out = prec.w_dl
            rank_out = prec.rank
        elif sample_link == "DL" and h_serving_est is not None:
            # DL only: derive UL via reciprocity (conj in contract convention)
            prec = compute_dl_precoding(np.conj(h_serving_est), max_rank=_max_r)
            w_dl_out = prec.w_dl
            rank_out = prec.rank

        # Legacy interference signal (observed at receiver)
        interference_signal: np.ndarray | None = None
        if h_interferers is not None:
            interf_sum = np.sum(h_interferers, axis=0)  # [T, RB, BS, UE]
            interf_signal_raw = np.sum(interf_sum, axis=2)  # [T, RB, UE]
            interference_signal = interf_signal_raw.astype(np.complex64)

        # -- SSB measurements (multi-cell beam sweep) --------------------------
        ssb_rsrp: list[float] | None = None
        ssb_rsrq: list[float] | None = None
        ssb_sinr: list[float] | None = None
        ssb_best_beam: list[int] | None = None
        ssb_pcis_list: list[int] | None = None
        ssb_rsrp_true: list[float] | None = None
        ssb_sinr_true: list[float] | None = None

        if self.enable_ssb and K >= 1:
            from msg_embedding.phy_sim.ssb_measurement import SSBMeasurement

            ssb_meas = SSBMeasurement(
                num_beams=self.num_ssb_beams,
                num_bs_ant=BS_ant,
            )
            all_pcis = [s.pci for s in sites]
            noise_power_lin = 10.0 ** (noise_power_dbm / 10.0) * 1e-3
            ssb_result = ssb_meas.measure(
                h_per_cell=h_all,
                pcis=all_pcis,
                noise_power_lin=noise_power_lin,
            )
            ssb_rsrp = ssb_result.rsrp_dBm.tolist()
            ssb_rsrq = ssb_result.rsrq_dB.tolist()
            ssb_sinr = ssb_result.ss_sinr_dB.tolist()
            ssb_best_beam = ssb_result.best_beam_idx.tolist()
            ssb_pcis_list = ssb_result.pcis

            if paired:
                ssb_result_ideal = ssb_meas.measure(
                    h_per_cell=h_all,
                    pcis=all_pcis,
                    noise_power_lin=1e-30,
                )
                ssb_rsrp_true = ssb_result_ideal.rsrp_dBm.tolist()
                ssb_sinr_true = ssb_result_ideal.ss_sinr_dB.tolist()

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
            "mobility_mode": self.mobility_mode,
            "ue_distribution": self.ue_distribution,
            "pilot_type": effective_pilot,
            "pilot_type_dl": self.pilot_type_dl,
            "pilot_type_ul": self.pilot_type_ul,
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
            "srs_comb": self.srs_comb,
            "srs_c_srs": self.srs_c_srs,
            "srs_b_srs": self.srs_b_srs,
            "srs_n_rrc": self.srs_n_rrc,
            "srs_rb_indices": srs_rb_idx.tolist(),
            "tdd_slot_index": slot_idx,
            "tdd_slot_direction": slot_direction,
            "tdd_symbol_map": symbol_map,
            "srs_active_in_slot": srs_active_in_slot,
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
            "dl_rank": rank_out,
        }
        if w_dl_out is not None:
            meta["w_dl_shape"] = list(w_dl_out.shape)
            meta["w_dl"] = w_dl_out

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
            link_pairing=link_pairing,  # type: ignore[arg-type]
            h_ul_true=h_ul_true_out,
            h_ul_est=h_ul_est_out,
            h_dl_true=h_dl_true_out,
            h_dl_est=h_dl_est_out,
            ul_sir_dB=ul_sir_db_out,
            dl_sir_dB=dl_sir_db_out,
            num_interfering_ues=n_intf_ues_out,
            ssb_rsrp_true_dBm=ssb_rsrp_true,
            ssb_sinr_true_dB=ssb_sinr_true,
            w_dl=w_dl_out,
            dl_rank=rank_out,
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

        # -- Mobility: generate trajectory if mode is not static ---------------
        trajectory_positions: np.ndarray | None = None
        trajectory_dopplers: np.ndarray | None = None
        static_pos: np.ndarray | None = None

        if self.mobility_mode == "static" or self.ue_speed_kmh <= 0:
            static_rng = np.random.default_rng(self._seed + 8888)
            if self.custom_ue_positions:
                p = self.custom_ue_positions[0]
                static_pos = np.array(
                    [float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("z", self.ue_height_m))],
                    dtype=np.float64,
                )
            else:
                static_pos = self._place_ues(static_rng, sites, 1)[0]
        elif self.mobility_mode != "static" and self.ue_speed_kmh > 0:
            from ._mobility import compute_doppler_from_trajectory, generate_trajectory

            traj_rng = np.random.default_rng(self._seed + 9999)
            # Place initial UE position
            init_pos = self._place_ues(traj_rng, sites, 1)[0]

            cell_radius = self.isd_m / math.sqrt(3.0)
            num_rings = _sites_to_rings(self.num_sites)
            network_radius = max(cell_radius, self.isd_m * num_rings + cell_radius)

            trajectory_positions = generate_trajectory(
                rng=traj_rng,
                start_pos=init_pos,
                speed_kmh=self.ue_speed_kmh,
                num_steps=self.num_samples,
                dt_s=self.sample_interval_s,
                mode=self.mobility_mode,
                boundary_radius_m=network_radius,
                boundary_center=np.zeros(2, dtype=np.float64),
            )

            # Doppler relative to serving cell (use site 0 as rough reference)
            serving_pos = np.asarray(sites[0].position, dtype=np.float64)
            trajectory_dopplers = compute_doppler_from_trajectory(
                positions=trajectory_positions,
                bs_pos=serving_pos,
                carrier_freq_hz=self.carrier_freq_hz,
                dt_s=self.sample_interval_s,
            )

        # -- Custom positions: spatially correlated LSPs -----------------------
        trajectory_lsps: list[dict] | None = None
        if trajectory_positions is not None:
            traj_rng2 = np.random.default_rng(self._seed)
            trajectory_lsps = self._generate_trajectory_lsps(traj_rng2, trajectory_positions)
        elif self.custom_ue_positions and len(self.custom_ue_positions) > 1:
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
            traj_rng2 = np.random.default_rng(self._seed)
            trajectory_lsps = self._generate_trajectory_lsps(traj_rng2, positions)

        # Fading seed for temporal continuity across snapshots in mobility mode
        mobility_fading_seed: int | None = None
        if trajectory_positions is not None:
            mobility_fading_seed = self._seed + 7777

        for idx in range(self.num_samples):
            lsp_override = None
            if trajectory_lsps is not None and idx < len(trajectory_lsps):
                lsp_override = trajectory_lsps[idx]

            ue_pos_ov: np.ndarray | None = None
            doppler_ov: float | None = None
            if static_pos is not None:
                ue_pos_ov = static_pos
            elif trajectory_positions is not None:
                ue_pos_ov = trajectory_positions[idx]
            if trajectory_dopplers is not None:
                doppler_ov = float(np.abs(trajectory_dopplers[idx]))

            t_offset = idx * self.sample_interval_s if mobility_fading_seed is not None else 0.0

            if self.link == "BOTH":
                yield self._generate_one_sample(
                    idx,
                    sites,
                    lsp_override=lsp_override,
                    paired=True,
                    ue_pos_override=ue_pos_ov,
                    doppler_hz_override=doppler_ov,
                    fading_seed=mobility_fading_seed,
                    snapshot_t_offset_s=t_offset,
                )
            else:
                yield self._generate_one_sample(
                    idx,
                    sites,
                    lsp_override=lsp_override,
                    ue_pos_override=ue_pos_ov,
                    doppler_hz_override=doppler_ov,
                    fading_seed=mobility_fading_seed,
                    snapshot_t_offset_s=t_offset,
                )

    # ------------------------------------------------------------------
    # Describe
    # ------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """Return metadata describing the source configuration."""
        K = self.num_sites * self.sectors_per_site
        multiplier = 1
        return {
            "source": "internal_sim",
            "scenario": self.scenario,
            "num_sites": self.num_sites,
            "num_cells": K,
            "num_ues": self.num_ues,
            "expected_sample_count": self.num_samples * multiplier,
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
            "mobility_mode": self.mobility_mode,
            "sample_interval_s": self.sample_interval_s,
            "ue_distribution": self.ue_distribution,
            "channel_model": {
                "num_taps": self.num_taps,
                "tau_rms_ns": self.tau_rms_ns,
                "rician_k_db": self.rician_k_db,
            },
            "mock": False,
        }


__all__ = ["InternalSimSource"]
