"""Sionna 2.0 Ray Tracing data source with TDL fallback.

This module provides two concrete :class:`DataSource` implementations that
feed the unified :class:`ChannelSample` contract:

* :class:`SionnaRTSource` — when Sionna 2.0.1 is installed, drives real ray
  tracing via ``scene.compute_paths()`` → CIR → frequency response.  When
  Sionna is **not** installed, falls back to a 3GPP 38.901 TDL-based channel
  model (real physics, just not ray-traced).  The fallback is logged as a
  warning, but the source never hard-fails on import.

* :class:`SionnaRTMockSource` — a pure-NumPy structural mock for unit tests.
  Produces valid :class:`ChannelSample` objects with i.i.d. Gaussian channels
  and ``meta['mock'] = True``.

Sionna 2.0 limitation (acknowledged upstream): ``scene.tx_array`` is a
single global :class:`~sionna.rt.PlanarArray` shared by every transmitter
— you cannot give different BS different antenna panels in the same scene.
We document this and assume homogeneous BS arrays across the cluster.

See ``docs/phase1_5_sionna_setup.md`` for the concrete environment setup.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from ..contract import ChannelSample
from .base import DataSource, register_source

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Sionna / PyTorch imports
# ---------------------------------------------------------------------------

try:  # pragma: no cover - exercised only in real-Sionna environments
    import sionna as _sn  # type: ignore  # noqa: F401
    import sionna.rt as _rt  # type: ignore
    import torch  # type: ignore

    _SIONNA_AVAILABLE = True
    _SIONNA_IMPORT_ERROR: Optional[BaseException] = None  # noqa: UP045
except Exception as _exc:  # pragma: no cover - hit on all CI boxes
    torch = None  # type: ignore[assignment]
    _rt = None  # type: ignore[assignment]
    _SIONNA_AVAILABLE = False
    _SIONNA_IMPORT_ERROR = _exc


# ---------------------------------------------------------------------------
# Shared defaults
# ---------------------------------------------------------------------------

_DEFAULT_NOISE_DBM: float = -100.0
_DEFAULT_SNR_DB: float = 15.0
_DEFAULT_SIR_DB: float = 6.0


def _dict_get(cfg: Any, key: str, default: Any) -> Any:
    """Small helper that works for DictConfig / dict / SimpleNamespace / None."""
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except Exception:  # pragma: no cover - exotic config types
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
        if r > 20:  # safety cap
            return r


def _clamp_db(val: float, lo: float = -49.9, hi: float = 49.9) -> float:
    """Clamp a dB value to the ChannelSample contract range."""
    return float(np.clip(val, lo, hi))


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
# Pathloss models (3GPP TR 38.901 Table 7.4.1-1)
# ---------------------------------------------------------------------------


def _pathloss_uma_nlos(d_3d: float, fc_ghz: float, h_bs: float, h_ut: float) -> tuple[float, float]:
    """38.901 UMa NLOS: Table 7.4.1-1."""
    d_3d = max(d_3d, 1.0)
    pl = 13.54 + 39.08 * math.log10(d_3d) + 20.0 * math.log10(fc_ghz) - 0.6 * (h_ut - 1.5)
    return pl, 6.0


def _pathloss_umi_nlos(d_3d: float, fc_ghz: float, h_bs: float, h_ut: float) -> tuple[float, float]:
    """38.901 UMi Street Canyon NLOS (simplified)."""
    d_3d = max(d_3d, 1.0)
    pl = 22.4 + 35.3 * math.log10(d_3d) + 21.3 * math.log10(fc_ghz) - 0.3 * (h_ut - 1.5)
    return pl, 7.82


def _pathloss_uma_los(d_3d: float, fc_ghz: float, h_bs: float, h_ut: float) -> tuple[float, float]:
    """38.901 UMa LOS: Table 7.4.1-1 with breakpoint distance."""
    d_3d = max(d_3d, 1.0)
    h_e = 1.0
    h_bs_eff = h_bs - h_e
    h_ut_eff = max(h_ut - h_e, 0.5)
    c = 3.0e8
    d_bp = 4.0 * h_bs_eff * h_ut_eff * fc_ghz * 1e9 / c
    d_2d = math.sqrt(max(d_3d ** 2 - (h_bs - h_ut) ** 2, 1.0))
    if d_2d <= d_bp:
        pl = 28.0 + 22.0 * math.log10(d_3d) + 20.0 * math.log10(fc_ghz)
    else:
        pl = (
            28.0
            + 40.0 * math.log10(d_3d)
            + 20.0 * math.log10(fc_ghz)
            - 9.0 * math.log10(d_bp ** 2 + (h_bs - h_ut) ** 2)
        )
    return pl, 4.0


def _pathloss_umi_los(d_3d: float, fc_ghz: float, h_bs: float, h_ut: float) -> tuple[float, float]:
    """38.901 UMi Street Canyon LOS: Table 7.4.1-1 with breakpoint distance."""
    d_3d = max(d_3d, 1.0)
    h_e = 1.0
    h_bs_eff = h_bs - h_e
    h_ut_eff = max(h_ut - h_e, 0.5)
    c = 3.0e8
    d_bp = 4.0 * h_bs_eff * h_ut_eff * fc_ghz * 1e9 / c
    d_2d = math.sqrt(max(d_3d ** 2 - (h_bs - h_ut) ** 2, 1.0))
    if d_2d <= d_bp:
        pl = 32.4 + 21.0 * math.log10(d_3d) + 20.0 * math.log10(fc_ghz)
    else:
        pl = (
            32.4
            + 40.0 * math.log10(d_3d)
            + 20.0 * math.log10(fc_ghz)
            - 9.5 * math.log10(d_bp ** 2 + (h_bs - h_ut) ** 2)
        )
    return pl, 4.0


def _pathloss_inf(d_3d: float, fc_ghz: float, h_bs: float, h_ut: float) -> tuple[float, float]:
    """38.901 InF-SH NLOS (simplified)."""
    d_3d = max(d_3d, 1.0)
    pl = 31.84 + 21.50 * math.log10(d_3d) + 19.0 * math.log10(fc_ghz)
    return pl, 7.56


_PATHLOSS_MODELS = {
    "UMa_NLOS": _pathloss_uma_nlos,
    "UMa_LOS": _pathloss_uma_los,
    "UMi_NLOS": _pathloss_umi_nlos,
    "UMi_LOS": _pathloss_umi_los,
    "InF": _pathloss_inf,
}

# Scenario aliases: map Sionna scene names to pathloss model keys
_SCENARIO_TO_PATHLOSS = {
    "munich": "UMa_NLOS",
    "etoile": "UMa_NLOS",
    "custom_osm": "UMa_NLOS",
}

_LOS_SCENARIO_MAP = {
    "UMa_NLOS": "UMa_LOS",
    "UMi_NLOS": "UMi_LOS",
    "InF": "InF",
}

_SCENARIO_SPATIAL_RHO = {
    "UMa_NLOS": 0.7,
    "UMa_LOS": 0.8,
    "UMi_NLOS": 0.5,
    "UMi_LOS": 0.6,
    "InF": 0.3,
}


def _los_probability(scenario: str, d_2d: float) -> float:
    """38.901 Table 7.4.2-1 LOS probability."""
    d_2d = max(d_2d, 1.0)
    if scenario in ("UMa_NLOS", "UMa_LOS"):
        return min(18.0 / d_2d, 1.0) * (1.0 - math.exp(-d_2d / 63.0)) + math.exp(-d_2d / 63.0)
    elif scenario in ("UMi_NLOS", "UMi_LOS"):
        return min(18.0 / d_2d, 1.0) * (1.0 - math.exp(-d_2d / 36.0)) + math.exp(-d_2d / 36.0)
    elif scenario.startswith("InF"):
        d_clutter = 10.0
        return math.exp(-d_2d / d_clutter)
    return 0.0


# ---------------------------------------------------------------------------
# Panel array antenna model (dual-polarization)
# ---------------------------------------------------------------------------


@dataclass
class PanelConfig:
    """Antenna panel geometry: N_H x N_V x N_P."""

    n_h: int = 1
    n_v: int = 1
    n_p: int = 1

    @property
    def total(self) -> int:
        return self.n_h * self.n_v * self.n_p

    @property
    def is_dual_pol(self) -> bool:
        return self.n_p == 2


def _panel_correlation_matrix(
    panel: PanelConfig,
    rho_h: float = 0.7,
    rho_v: float = 0.7,
    xpd_db: float = 8.0,
) -> np.ndarray:
    """Build spatial correlation matrix for a panel array: R = R_H ⊗ R_V ⊗ R_P."""
    N_h, N_v, N_p = panel.n_h, panel.n_v, panel.n_p
    R_h = np.zeros((N_h, N_h), dtype=np.float64)
    for i in range(N_h):
        for j in range(N_h):
            R_h[i, j] = rho_h ** abs(i - j)
    R_v = np.zeros((N_v, N_v), dtype=np.float64)
    for i in range(N_v):
        for j in range(N_v):
            R_v[i, j] = rho_v ** abs(i - j)
    if N_p == 2:
        mu = 10.0 ** (-xpd_db / 10.0)
        R_p = np.array([[1.0, mu], [mu, 1.0]], dtype=np.float64)
    else:
        R_p = np.ones((1, 1), dtype=np.float64)
    R = np.kron(np.kron(R_h, R_v), R_p)
    return R


def _panel_steering_vector(
    panel: PanelConfig,
    azimuth_rad: float,
    elevation_rad: float = 0.0,
    d_spacing: float = 0.5,
) -> np.ndarray:
    """2D panel array steering vector with dual-polarization."""
    N_h, N_v, N_p = panel.n_h, panel.n_v, panel.n_p
    a_h = np.exp(1j * 2 * np.pi * d_spacing * np.arange(N_h) * np.sin(azimuth_rad))
    a_v = np.exp(1j * 2 * np.pi * d_spacing * np.arange(N_v) * np.cos(elevation_rad))
    a_spatial = np.kron(a_h, a_v)
    if N_p == 2:
        a = np.kron(a_spatial, np.ones(2))
    else:
        a = a_spatial
    return a / np.sqrt(panel.total)


# ---------------------------------------------------------------------------
# TDL channel generation (standalone, same approach as internal_sim)
# ---------------------------------------------------------------------------


def _generate_tdl_channel(
    rng: np.random.Generator,
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
    tx_panel: PanelConfig | None = None,
    rx_panel: PanelConfig | None = None,
    xpd_db: float = 8.0,
) -> np.ndarray:
    """Generate a frequency-domain channel matrix via a tapped-delay-line model.

    Returns shape ``[T, RB, tx_ant, rx_ant]`` complex128 (cast to c64 by caller).

    When ``tdl_profile`` is provided (a :class:`TDLProfile` from
    ``channel_models.tdl``), uses the profile's tap delays and powers per
    3GPP 38.901.  Otherwise falls back to an exponential PDP.
    """
    T = num_ofdm_sym
    N_tx = num_tx_ant
    N_rx = num_rx_ant
    N_fft = num_rb

    tau_rms_s = tau_rms_ns * 1e-9

    if tdl_profile is not None:
        L = tdl_profile.num_taps
        pdp = tdl_profile.powers_normalized()
        delays_s = tdl_profile.delays_seconds(tau_rms_s)
        if tdl_profile.is_los and tdl_profile.k_factor_dB is not None:
            rician_k_linear = 10.0 ** (tdl_profile.k_factor_dB / 10.0)
            los_tap = tdl_profile.los_tap_index
        else:
            los_tap = 0
    else:
        L = max(12, 1)
        tap_spacing_s = 1.0 / (subcarrier_spacing_hz * num_rb) if num_rb > 0 else 1e-6
        decay = tap_spacing_s / tau_rms_s if tau_rms_s > 0 and tap_spacing_s > 0 else 1.0
        pdp = np.exp(-np.arange(L, dtype=np.float64) * decay)
        pdp /= pdp.sum()
        delays_s = np.arange(L, dtype=np.float64) * tap_spacing_s
        los_tap = 0

    # Spatial correlation (panel-aware or exponential fallback)
    if tx_panel is not None and tx_panel.total == N_tx:
        R_tx = _panel_correlation_matrix(tx_panel, rho_h=spatial_corr_rho, rho_v=spatial_corr_rho, xpd_db=xpd_db)
    else:
        rho = spatial_corr_rho
        R_tx = np.zeros((N_tx, N_tx), dtype=np.float64)
        for i in range(N_tx):
            for j in range(N_tx):
                R_tx[i, j] = rho ** abs(i - j)
    if rx_panel is not None and rx_panel.total == N_rx:
        R_rx = _panel_correlation_matrix(rx_panel, rho_h=spatial_corr_rho, rho_v=spatial_corr_rho, xpd_db=xpd_db)
    else:
        rho = spatial_corr_rho
        R_rx = np.zeros((N_rx, N_rx), dtype=np.float64)
        for i in range(N_rx):
            for j in range(N_rx):
                R_rx[i, j] = rho ** abs(i - j)
    L_tx = np.linalg.cholesky(R_tx + 1e-8 * np.eye(N_tx))
    L_rx = np.linalg.cholesky(R_rx + 1e-8 * np.eye(N_rx))

    # Per-tap complex gains with spatial correlation
    h_taps_real = rng.standard_normal((L, 1, N_tx, N_rx))
    h_taps_imag = rng.standard_normal((L, 1, N_tx, N_rx))
    h_taps_iid = (h_taps_real + 1j * h_taps_imag) / math.sqrt(2.0)
    h_taps_corr = np.zeros_like(h_taps_iid)
    for l_idx in range(L):
        h_taps_corr[l_idx, 0] = L_tx @ h_taps_iid[l_idx, 0] @ L_rx.T
    h_taps = np.broadcast_to(h_taps_corr, (L, T, N_tx, N_rx)).copy()

    # Apply PDP scaling
    pdp_sqrt = np.sqrt(pdp)
    h_taps *= pdp_sqrt[:, None, None, None]

    # Rician LOS component with steering vectors
    if rician_k_linear > 0:
        K_r = rician_k_linear
        nlos_scale = math.sqrt(1.0 / (1.0 + K_r))
        los_scale = math.sqrt(K_r / (1.0 + K_r))
        d_spacing = 0.5
        a_tx = np.exp(1j * 2 * np.pi * d_spacing * np.arange(N_tx) * np.sin(los_aod_rad)) / np.sqrt(
            N_tx
        )
        a_rx = np.exp(1j * 2 * np.pi * d_spacing * np.arange(N_rx) * np.sin(los_aoa_rad)) / np.sqrt(
            N_rx
        )
        los_matrix = np.outer(a_tx, a_rx.conj())
        h_taps[los_tap] = (
            h_taps[los_tap] * nlos_scale + los_scale * pdp_sqrt[los_tap] * los_matrix[None, :, :]
        )

    # Doppler via improved Jakes sum-of-sinusoids model
    if doppler_hz > 0 and T > 1:
        N_sinusoids = 16
        T_sym = 1.0 / subcarrier_spacing_hz
        t_axis = np.arange(T, dtype=np.float64) * T_sym

        for l_idx in range(L):
            alpha_n = rng.uniform(0, 2 * np.pi, size=N_sinusoids)
            phi_n = rng.uniform(0, 2 * np.pi, size=N_sinusoids)
            doppler_real = np.zeros(T, dtype=np.float64)
            doppler_imag = np.zeros(T, dtype=np.float64)
            for n in range(N_sinusoids):
                f_shift = doppler_hz * np.cos(alpha_n[n])
                doppler_real += np.cos(2.0 * np.pi * f_shift * t_axis + phi_n[n])
                doppler_imag += np.sin(2.0 * np.pi * f_shift * t_axis + phi_n[n])
            doppler_process = (doppler_real + 1j * doppler_imag) / math.sqrt(N_sinusoids)
            h_taps[l_idx] *= doppler_process[:, None, None]

    # DFT: time-delay -> frequency domain
    k_axis = np.arange(N_fft, dtype=np.float64)
    delta_f = 12 * subcarrier_spacing_hz
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
    r = cell_radius_m * np.sqrt(rng.uniform(0, 1, size=num_ues))
    theta = rng.uniform(0, 2 * np.pi, size=num_ues)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    z = np.full(num_ues, h_ut, dtype=np.float64)
    return np.stack([x, y, z], axis=-1)


def _place_ues_clustered(
    rng: np.random.Generator,
    num_ues: int,
    cell_radius_m: float,
    h_ut: float,
    num_clusters: int = 3,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """Clustered UE placement."""
    if center is None:
        center = np.zeros(2, dtype=np.float64)
    cluster_r = cell_radius_m * 0.6 * np.sqrt(rng.uniform(0, 1, size=num_clusters))
    cluster_theta = rng.uniform(0, 2 * np.pi, size=num_clusters)
    cx = center[0] + cluster_r * np.cos(cluster_theta)
    cy = center[1] + cluster_r * np.sin(cluster_theta)

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
    ues_c = _place_ues_uniform(rng, max(n_centre, 1), cell_radius_m * 0.3, h_ut, center)
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

    Delegates to the canonical implementation in internal_sim.
    """
    from msg_embedding.data.sources.internal_sim import (
        _generate_pilots_srs as _impl,
    )

    return _impl(
        num_rb, cell_id,
        slot=slot, symbol=symbol,
        group_hopping=group_hopping,
        sequence_hopping=sequence_hopping,
        K_TC=K_TC,
        srs_num_rb=srs_num_rb,
    )


def _generate_pilots_csirs(
    num_rb: int,
    cell_id: int,
    slot: int = 0,
    symbol: int = 0,
) -> np.ndarray:
    """Generate CSI-RS pilot symbols for DL.  Returns [num_rb] complex128."""
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


# ---------------------------------------------------------------------------
# Real Sionna RT source
# ---------------------------------------------------------------------------


@register_source
class SionnaRTSource(DataSource):
    """Sionna 2.0 Ray Tracing with TDL fallback -> :class:`ChannelSample` iterator.

    When Sionna 2.0.1 is installed, uses real ``scene.compute_paths()`` for
    ray-traced CIR.  When Sionna is NOT installed, falls back to a TDL-based
    channel generator using 3GPP 38.901 TDL profiles — real physics, just
    not ray-traced.

    The constructor does NOT raise :class:`ImportError` when Sionna is
    absent.  Instead it sets ``self._use_real_sionna = False`` and logs a
    warning.

    Accepts ALL frontend config parameters for parity with
    :class:`InternalSimSource`.
    """

    name = "sionna_rt"

    def __init__(self, config: Any) -> None:
        bypass = _dict_get(config if isinstance(config, dict) else {}, "bypass_rt", None)
        if bypass is not None:
            self._use_real_sionna = not bool(bypass)
        else:
            self._use_real_sionna = _SIONNA_AVAILABLE
        if not self._use_real_sionna:
            reason = "bypass_rt=True in config" if bypass else repr(_SIONNA_IMPORT_ERROR)
            logger.warning(
                "SionnaRTSource will use TDL-based channel generation "
                "(reason: %s) — real 38.901 physics, but no ray tracing.",
                reason,
            )
        super().__init__(config)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def validate_config(self) -> None:  # noqa: C901 — complex but linear
        cfg = self.config

        # -- Scene / scenario --------------------------------------------------
        self.scenario: str = str(_dict_get(cfg, "scenario", "munich"))
        if self.scenario not in {"munich", "etoile", "custom_osm"}:
            raise ValueError(
                f"Unknown scenario {self.scenario!r}; expected "
                "'munich' / 'etoile' / 'custom_osm'."
            )
        self.osm_path: str | None = _dict_get(cfg, "osm_path", None)
        if self.scenario == "custom_osm" and not self.osm_path:
            raise ValueError("scenario='custom_osm' requires `osm_path`.")

        # Map scenario to pathloss model key for TDL fallback
        self._pathloss_scenario: str = _SCENARIO_TO_PATHLOSS.get(self.scenario, "UMa_NLOS")

        # -- Cell / site count (num_sites preferred over num_cells) ------------
        _num_sites_raw = _dict_get(cfg, "num_sites", None)
        _num_cells_raw = _dict_get(cfg, "num_cells", None)
        if _num_sites_raw is not None:
            self.num_cells: int = int(_num_sites_raw)
        elif _num_cells_raw is not None:
            self.num_cells = int(_num_cells_raw)
        else:
            self.num_cells = 7

        self.num_ues: int = int(_dict_get(cfg, "num_ues", 1))
        self.num_interfering_ues: int = int(_dict_get(cfg, "num_interfering_ues", max(self.num_ues - 1, 0)))
        self.num_samples: int = int(_dict_get(cfg, "num_samples", 1))

        # -- Topology params ---------------------------------------------------
        self.isd_m: float = float(_dict_get(cfg, "isd_m", 500.0))
        self.sectors_per_site: int = int(_dict_get(cfg, "sectors_per_site", 3))
        if self.sectors_per_site not in (1, 3):
            raise ValueError(f"sectors_per_site must be 1 or 3, got {self.sectors_per_site}.")
        self.tx_height_m: float = float(_dict_get(cfg, "tx_height_m", 25.0))

        # -- OFDM grid params --------------------------------------------------
        self.carrier_freq_hz: float = float(_dict_get(cfg, "carrier_freq_hz", 3.5e9))
        self.bandwidth_hz: float = float(_dict_get(cfg, "bandwidth_hz", 100e6))
        self.fft_size: int = int(_dict_get(cfg, "fft_size", 1024))
        self.num_ofdm_symbols: int = int(_dict_get(cfg, "num_ofdm_symbols", 14))
        self.subcarrier_spacing: float = float(_dict_get(cfg, "subcarrier_spacing", 30e3))
        # Number of RBs: 3GPP TS 38.101 standard table lookup
        from msg_embedding.phy_sim.nr_rb_table import nr_rb_lookup

        _cfg_num_rb = _dict_get(cfg, "num_rb", None)
        if _cfg_num_rb is not None:
            self._num_rb: int = int(_cfg_num_rb)
        else:
            self._num_rb = nr_rb_lookup(self.bandwidth_hz, self.subcarrier_spacing)

        # -- Antenna arrays (split TX/RX; legacy kept for backward compat) ----
        _legacy_bs = int(_dict_get(cfg, "num_bs_ant", 4))
        _legacy_ue = int(_dict_get(cfg, "num_ue_ant", 2))
        self.num_bs_tx_ant: int = int(_dict_get(cfg, "num_bs_tx_ant", _legacy_bs))
        self.num_bs_rx_ant: int = int(_dict_get(cfg, "num_bs_rx_ant", _legacy_bs))
        self.num_ue_tx_ant: int = int(_dict_get(cfg, "num_ue_tx_ant", _legacy_ue))
        self.num_ue_rx_ant: int = int(_dict_get(cfg, "num_ue_rx_ant", _legacy_ue))
        self.num_bs_ant: int = max(self.num_bs_tx_ant, self.num_bs_rx_ant)
        self.num_ue_ant: int = max(self.num_ue_tx_ant, self.num_ue_rx_ant)

        # -- Polarization (Sionna PlanarArray) ---------------------------------
        self.bs_polarization: str = str(_dict_get(cfg, "bs_polarization", "V"))
        if self.bs_polarization not in {"V", "H", "VH"}:
            raise ValueError(
                f"bs_polarization must be 'V', 'H', or 'VH', " f"got {self.bs_polarization!r}."
            )
        self.ue_polarization: str = str(_dict_get(cfg, "ue_polarization", "V"))
        if self.ue_polarization not in {"V", "H", "VH"}:
            raise ValueError(
                f"ue_polarization must be 'V', 'H', or 'VH', " f"got {self.ue_polarization!r}."
            )

        # -- Panel array (dual-polarization) -----------------------------------
        _bs_panel_raw = _dict_get(cfg, "bs_panel", None)
        _ue_panel_raw = _dict_get(cfg, "ue_panel", None)
        if _bs_panel_raw is not None:
            _bp = [int(x) for x in _bs_panel_raw]
            self.bs_panel = PanelConfig(n_h=_bp[0], n_v=_bp[1], n_p=_bp[2])
            self.num_bs_tx_ant = self.bs_panel.total
            self.num_bs_rx_ant = self.bs_panel.total
            self.num_bs_ant = self.bs_panel.total
        else:
            self.bs_panel: PanelConfig | None = None
        if _ue_panel_raw is not None:
            _up = [int(x) for x in _ue_panel_raw]
            self.ue_panel = PanelConfig(n_h=_up[0], n_v=_up[1], n_p=_up[2])
            self.num_ue_tx_ant = self.ue_panel.total
            self.num_ue_rx_ant = self.ue_panel.total
            self.num_ue_ant = self.ue_panel.total
        else:
            self.ue_panel: PanelConfig | None = None
        self.xpd_db: float = float(_dict_get(cfg, "xpd_db", 8.0))

        # -- Power / mobility --------------------------------------------------
        _tx_power_explicitly_set = _dict_get(cfg, "tx_power_dbm", None) is not None
        self.tx_power_dbm: float = float(_dict_get(cfg, "tx_power_dbm", 43.0))
        if not _tx_power_explicitly_set:
            _default_tx_power = {
                "UMa_NLOS": 43.0,
                "UMa_LOS": 43.0,
                "UMi_NLOS": 33.0,
                "UMi_LOS": 33.0,
                "InF": 24.0,
            }
            self.tx_power_dbm = _default_tx_power.get(self._pathloss_scenario, 43.0)
        self.ue_tx_power_dbm: float = float(_dict_get(cfg, "ue_tx_power_dbm", 23.0))
        self.ue_speed_kmh: float = float(_dict_get(cfg, "ue_speed_kmh", 3.0))
        self.mobility_mode: str = str(_dict_get(cfg, "mobility_mode", "static"))
        from ._mobility import MOBILITY_MODES as _MM
        if self.mobility_mode not in _MM:
            raise ValueError(
                f"Unknown mobility_mode {self.mobility_mode!r}; expected one of {_MM}."
            )
        self.sample_interval_s: float = float(_dict_get(cfg, "sample_interval_s", 0.5e-3))
        self.ue_distribution: str = str(_dict_get(cfg, "ue_distribution", "uniform"))
        if self.ue_distribution not in {"uniform", "clustered", "hotspot"}:
            raise ValueError(
                f"Unknown ue_distribution {self.ue_distribution!r}; expected "
                "'uniform' / 'clustered' / 'hotspot'."
            )

        # -- HSR (High-Speed Rail) parameters -----------------------------------
        self.topology_layout: str = str(_dict_get(cfg, "topology_layout", "hexagonal"))
        if self.topology_layout not in {"hexagonal", "linear"}:
            raise ValueError(f"topology_layout must be 'hexagonal' or 'linear', got {self.topology_layout!r}.")
        self.hypercell_size: int = int(_dict_get(cfg, "hypercell_size", 1))
        if self.hypercell_size < 1:
            raise ValueError(f"hypercell_size must be >= 1, got {self.hypercell_size}")
        self.track_offset_m: float = float(_dict_get(cfg, "track_offset_m", 80.0))
        self.train_penetration_loss_db: float = float(_dict_get(cfg, "train_penetration_loss_db", 0.0))

        # -- Custom positions (from frontend drag-and-drop) -------------------
        self.custom_site_positions: list[dict[str, float]] | None = _dict_get(
            cfg, "custom_site_positions", None
        )
        self.custom_ue_positions: list[dict[str, float]] | None = _dict_get(
            cfg, "custom_ue_positions", None
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
        if self.channel_est_mode not in {"ideal", "ls_linear", "ls_mmse", "ls_hop_concat"}:
            raise ValueError(f"Unknown channel_est_mode {self.channel_est_mode!r}.")
        self.link: str = str(_dict_get(cfg, "link", "DL")).upper()
        if self.link not in {"UL", "DL", "BOTH"}:
            raise ValueError(f"link must be 'UL', 'DL', or 'both', got {self.link!r}.")
        self.device: str = str(_dict_get(cfg, "device", "cuda"))
        self.serving_cell_id: int = int(_dict_get(cfg, "serving_cell_id", 0))
        if not (0 <= self.serving_cell_id < self.num_cells):
            raise ValueError(
                f"serving_cell_id={self.serving_cell_id} out of range " f"[0, {self.num_cells})."
            )

        # -- Seed ---------------------------------------------------------------
        self._seed: int = int(_dict_get(cfg, "seed", 42))

        # -- RT solver tuning ---------------------------------------------------
        self._rt_samples_per_src: int = int(_dict_get(cfg, "rt_samples_per_src", 5_000_000))
        self._rt_max_depth: int = int(_dict_get(cfg, "rt_max_depth", 5))

        # -- Channel model (TDL profile for fallback path) ---------------------
        from msg_embedding.channel_models.tdl import get_tdl_profile, list_tdl_profiles

        self.channel_model_name: str = (
            str(_dict_get(cfg, "channel_model", "TDL-C")).upper().replace("_", "-")
        )
        try:
            self._tdl_profile = get_tdl_profile(self.channel_model_name)
        except (KeyError, ValueError):
            logger.warning(
                "Unknown channel_model %r, falling back to TDL-C. " "Available: %s",
                self.channel_model_name,
                list_tdl_profiles(),
            )
            self.channel_model_name = "TDL-C"
            self._tdl_profile = get_tdl_profile("TDL-C")

        # TDL model params
        _default_tau_rms = {"UMa_NLOS": 363.0, "UMi_NLOS": 129.0, "InF": 56.0}
        self.tau_rms_ns: float = float(
            _dict_get(cfg, "tau_rms_ns", _default_tau_rms.get(self._pathloss_scenario, 200.0))
        )
        if self._tdl_profile.is_los and self._tdl_profile.k_factor_dB is not None:
            _default_k = self._tdl_profile.k_factor_dB
        else:
            _default_k = -100.0
        self.rician_k_db: float = float(_dict_get(cfg, "rician_k_db", _default_k))

        # -- TDD slot pattern --------------------------------------------------
        from msg_embedding.phy_sim.tdd_config import get_tdd_pattern

        self.tdd_pattern_name: str = str(_dict_get(cfg, "tdd_pattern", "DDDSU"))
        self._tdd_pattern = get_tdd_pattern(self.tdd_pattern_name)

        # -- SRS frequency hopping ---------------------------------------------
        self.srs_group_hopping: bool = bool(_dict_get(cfg, "srs_group_hopping", False))
        self.srs_sequence_hopping: bool = bool(_dict_get(cfg, "srs_sequence_hopping", False))
        self.srs_periodicity: int = int(_dict_get(cfg, "srs_periodicity", 10))
        self.srs_b_hop: int = int(_dict_get(cfg, "srs_b_hop", 0))
        self.srs_comb: int = int(_dict_get(cfg, "srs_comb", 2))
        self.srs_c_srs: int = int(_dict_get(cfg, "srs_c_srs", 3))
        self.srs_b_srs: int = int(_dict_get(cfg, "srs_b_srs", 1))
        self.srs_n_rrc: int = int(_dict_get(cfg, "srs_n_rrc", 0))
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

        # -- SSB measurement ---------------------------------------------------
        self.num_ssb_beams: int = int(_dict_get(cfg, "num_ssb_beams", 8))
        self.enable_ssb: bool = bool(_dict_get(cfg, "enable_ssb", True))

        # -- UE height ---------------------------------------------------------
        self.ue_height_m: float = float(_dict_get(cfg, "ue_height_m", 1.5))

        # -- Noise figure and thermal noise ------------------------------------
        self.noise_figure_db: float = float(_dict_get(cfg, "noise_figure_db", 7.0))

        # -- Precoding parameters -----------------------------------------------
        self.max_rank: int = int(_dict_get(cfg, "max_rank", 4))
        self.rank_threshold: float = float(_dict_get(cfg, "rank_threshold", 0.1))

        # -- Interferer precoding projection -----------------------------------
        self.apply_interferer_precoding: bool = bool(
            _dict_get(cfg, "apply_interferer_precoding", True)
        )
        self.store_interferer_channels: bool = bool(
            _dict_get(cfg, "store_interferer_channels", False)
        )

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
    # Scene setup (REAL-Sionna code path)
    # ------------------------------------------------------------------
    def _load_scene(self) -> Any:  # pragma: no cover - real-Sionna only
        """Load an OSM scene and attach global tx/rx arrays."""
        if self.scenario == "munich":
            scene = _rt.load_scene(_rt.scene.munich)
        elif self.scenario == "etoile":
            scene = _rt.load_scene(_rt.scene.etoile)
        else:
            scene = _rt.load_scene(self.osm_path)

        scene.frequency = self.carrier_freq_hz

        # --- BS (tx) planar array: use panel config if available ---------------
        if self.bs_panel is not None:
            bs_num_rows = self.bs_panel.n_v
            bs_num_cols = self.bs_panel.n_h
            bs_pol = "VH" if self.bs_panel.is_dual_pol else self.bs_polarization
        else:
            bs_pol = self.bs_polarization
            bs_pol_factor = 2 if bs_pol == "VH" else 1
            bs_elements = max(1, self.num_bs_ant // bs_pol_factor)
            bs_num_cols = max(1, int(math.ceil(math.sqrt(bs_elements))))
            bs_num_rows = max(1, bs_elements // bs_num_cols)
        scene.tx_array = _rt.PlanarArray(
            num_rows=bs_num_rows,
            num_cols=bs_num_cols,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="tr38901",
            polarization=bs_pol,
        )

        # --- UE (rx) planar array: use panel config if available ---------------
        if self.ue_panel is not None:
            ue_num_rows = self.ue_panel.n_v
            ue_num_cols = self.ue_panel.n_h
            ue_pol = "VH" if self.ue_panel.is_dual_pol else self.ue_polarization
        else:
            ue_pol = self.ue_polarization
            ue_pol_factor = 2 if ue_pol == "VH" else 1
            ue_elements = max(1, self.num_ue_ant // ue_pol_factor)
            ue_num_cols = max(1, int(math.ceil(math.sqrt(ue_elements))))
            ue_num_rows = max(1, ue_elements // ue_num_cols)
        scene.rx_array = _rt.PlanarArray(
            num_rows=ue_num_rows,
            num_cols=ue_num_cols,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization=ue_pol,
        )
        return scene

    def _add_transmitters(self, scene: Any, sites: list[Any]) -> list[Any]:
        """Attach one ``rt.Transmitter`` per cell site (real-Sionna path)."""
        txs = []
        for k, site in enumerate(sites[: self.num_cells]):
            tx = _rt.Transmitter(
                name=f"bs_{k}",
                position=np.asarray(site.position, dtype=np.float32),
                orientation=np.asarray(
                    [np.deg2rad(float(getattr(site, "azimuth_deg", 0.0))), 0.0, 0.0],
                    dtype=np.float32,
                ),
            )
            scene.add(tx)
            txs.append(tx)
        return txs

    # ------------------------------------------------------------------
    # Topology building
    # ------------------------------------------------------------------
    def _build_topology(self) -> list[Any]:
        """Build cell-site topology from config params.

        Uses custom drag-placed positions when ``custom_site_positions`` is
        set; linear layout for HSR; otherwise generates a hex grid.
        """
        from msg_embedding.topology.hex_grid import CellSite, make_hex_grid, make_linear_grid

        try:
            from msg_embedding.topology.pci_planner import assign_pci_hypercell, assign_pci_mod3

            _has_pci_planner = True
        except ImportError:
            _has_pci_planner = False

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
                )
                for i, p in enumerate(self.custom_site_positions)
            ]
        elif self.topology_layout == "linear":
            sites = make_linear_grid(
                num_sites=self.num_cells,
                isd_m=self.isd_m,
                sectors=self.sectors_per_site,
                tx_height_m=self.tx_height_m,
                track_offset_m=self.track_offset_m,
            )
        else:
            num_rings = _sites_to_rings(self.num_cells)
            sites = make_hex_grid(
                num_rings=num_rings,
                isd_m=self.isd_m,
                sectors=self.sectors_per_site,
                tx_height_m=self.tx_height_m,
            )

        # Assign PCIs
        if _has_pci_planner:
            if self.topology_layout == "linear" and self.hypercell_size > 1:
                sites = assign_pci_hypercell(sites, self.hypercell_size)
            else:
                sites = assign_pci_mod3(sites)
        else:
            # Fallback: assign sequential PCIs
            for i, s in enumerate(sites):
                if not hasattr(s, "pci") or s.pci is None:
                    s.pci = i

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
        """Place UEs.  Returns shape ``[num_ues, 3]`` float64."""
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
            if len(ue_pos) < num_ues:
                repeats = (num_ues // len(ue_pos)) + 1
                ue_pos = np.tile(ue_pos, (repeats, 1))[:num_ues]
            else:
                ue_pos = ue_pos[:num_ues]
            return ue_pos

        cell_radius = self.isd_m / math.sqrt(3.0)
        centre = np.zeros(2, dtype=np.float64)
        num_rings = _sites_to_rings(self.num_cells)
        network_radius = max(cell_radius, self.isd_m * num_rings + cell_radius)

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
    # Pathloss computation (38.901)
    # ------------------------------------------------------------------
    def _compute_pathloss(
        self,
        rng: np.random.Generator,
        bs_pos: np.ndarray,
        ue_pos: np.ndarray,
        is_los: bool = False,
    ) -> tuple[float, float]:
        """Return ``(pathloss_dB, d_3d)`` for one BS->UE link."""
        d_3d = float(np.linalg.norm(bs_pos - ue_pos))
        d_3d = max(d_3d, 1.0)
        fc_ghz = self.carrier_freq_hz / 1e9
        h_bs = float(bs_pos[2])
        h_ut = float(ue_pos[2])

        effective_scenario = (
            _LOS_SCENARIO_MAP.get(self._pathloss_scenario, self._pathloss_scenario)
            if is_los
            else self._pathloss_scenario
        )
        pl_func = _PATHLOSS_MODELS[effective_scenario]
        pl_db, sigma_sf = pl_func(d_3d, fc_ghz, h_bs, h_ut)
        shadow_db = float(rng.normal(0, sigma_sf))
        total_pl = pl_db + shadow_db + self.train_penetration_loss_db
        return total_pl, d_3d

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
    ) -> np.ndarray:
        """Run channel estimation.  ``h_true``: [T, RB, BS, UE] complex64."""
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
            return est.reshape(RB, T, BS, UE).transpose(1, 0, 2, 3).astype(np.complex64)

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

        h_at_pilots = h_true[np.ix_(rs_time, rs_freq)]
        h_at_pilots = h_at_pilots.transpose(1, 0, 2, 3)

        pilot_seq = pilots[rs_freq]
        pilot_tiled = np.repeat(pilot_seq, n_time)

        noise_std = 10.0 ** (-snr_db / 20.0) / math.sqrt(2.0)
        noise = noise_std * (
            rng.standard_normal(h_at_pilots.shape) + 1j * rng.standard_normal(h_at_pilots.shape)
        )
        h_noisy = h_at_pilots + noise.astype(np.complex128)

        X_expanded = pilot_seq[:, None, None, None]
        Y_at_pilots = h_noisy * X_expanded

        Y_flat = Y_at_pilots.reshape(n_freq * n_time, BS * UE)
        X_flat = pilot_tiled.astype(np.complex128)

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
                "tau_rms": self.tau_rms_ns * 1e-9,
                "delta_f": 12 * self.subcarrier_spacing * pilot_rb_spacing,
            },
            snr_db=snr_db,
            dtype="complex64",
        )
        return est.reshape(RB, T, BS, UE).transpose(1, 0, 2, 3).astype(np.complex64)

    # ------------------------------------------------------------------
    # Real Sionna RT channel computation
    # ------------------------------------------------------------------
    def _compute_channels_sionna(
        self,
        idx: int,
        sites: list[Any],
        ue_pos: np.ndarray,
        T: int,
        RB: int,
        BS: int,
        UE: int,
    ) -> list[np.ndarray]:  # pragma: no cover - real-Sionna only
        """Compute per-cell channels using Sionna 2.0.1 ray tracing.

        Returns a list of K numpy arrays, each [T, RB, BS, UE] complex64.
        """
        import drjit as _dr

        if not hasattr(self, "_rt_scene_cache"):
            self._rt_scene_cache = self._load_scene()
            self._rt_txs_cache = self._add_transmitters(self._rt_scene_cache, sites)
            self._rt_solver_cache = _rt.PathSolver()
        scene = self._rt_scene_cache

        rx_name = f"ue_{idx}"
        rx = _rt.Receiver(name=rx_name, position=ue_pos.astype(np.float32))
        scene.add(rx)

        solver = self._rt_solver_cache
        paths = solver(
            scene,
            max_depth=self._rt_max_depth,
            samples_per_src=self._rt_samples_per_src,
            seed=self._seed + idx,
            los=True,
            specular_reflection=True,
            diffuse_reflection=True,
            refraction=True,
            diffraction=True,
        )

        # Subcarrier frequencies for CFR computation
        num_sc = RB
        sc_freq_hz = self.subcarrier_spacing * 12
        freqs_np = np.arange(num_sc, dtype=np.float32) * sc_freq_hz
        freqs_dr = _dr.llvm.ad.Float(freqs_np)

        # CFR: [num_rx, rx_ant, num_tx, tx_ant, T, num_sc] complex64
        h_cfr = paths.cfr(freqs_dr, num_time_steps=T, out_type="numpy")

        num_tx_actual = min(self.num_cells, h_cfr.shape[2])
        h_per_cell = []

        for k in range(num_tx_actual):
            # h_k: [rx_ant, tx_ant, T, num_sc] -> extract rx=0
            h_k = h_cfr[0, :UE, k, :BS, :, :]  # [UE, BS, T, RB]
            # Rearrange to [T, RB, BS, UE]
            h_k = np.transpose(h_k, (2, 3, 1, 0)).astype(np.complex64)
            h_per_cell.append(np.ascontiguousarray(h_k))

        for _ in range(self.num_cells - num_tx_actual):
            h_per_cell.append(np.zeros((T, RB, BS, UE), dtype=np.complex64))

        scene.remove(rx_name)
        return h_per_cell

    # ------------------------------------------------------------------
    # TDL fallback channel computation
    # ------------------------------------------------------------------
    def _compute_channels_tdl(
        self,
        rng: np.random.Generator,
        sites: list[Any],
        ue_pos: np.ndarray,
        T: int,
        RB: int,
        BS: int,
        UE: int,
        sample_link: str = "DL",
    ) -> tuple[list[np.ndarray], np.ndarray, float, bool]:
        """Compute per-cell channels using 38.901 TDL model.

        Returns:
            h_all: list of K [T, RB, BS, UE] complex64 arrays
            rx_power_dbm: [K] array of received power per cell in dBm
            noise_power_dbm: float noise floor in dBm
            is_los: whether the serving link is LOS
        """
        wavelength = 3e8 / self.carrier_freq_hz
        ue_speed_ms = self.ue_speed_kmh / 3.6
        doppler_hz = ue_speed_ms / wavelength

        rician_k_linear = 10.0 ** (self.rician_k_db / 10.0) if self.rician_k_db > -50 else 0.0

        K = len(sites)
        noise_power_dbm = self._noise_power_dbm()
        rx_power_dbm = np.zeros(K, dtype=np.float64)
        pl_all = np.zeros(K, dtype=np.float64)
        h_all: list[np.ndarray] = []
        is_los_per_cell: list[bool] = []

        env_rng = np.random.default_rng(rng.integers(0, 2**32))

        _tx_panel = self.bs_panel if sample_link == "DL" else self.ue_panel
        _rx_panel = self.ue_panel if sample_link == "DL" else self.bs_panel

        for k, site in enumerate(sites):
            bs_pos = np.asarray(site.position, dtype=np.float64)
            d_3d_k = float(np.linalg.norm(bs_pos - ue_pos))
            h_bs_k = float(bs_pos[2])
            h_ut_k = float(ue_pos[2])
            d_2d_k = math.sqrt(max(d_3d_k ** 2 - (h_bs_k - h_ut_k) ** 2, 1.0))
            p_los_k = _los_probability(self._pathloss_scenario, d_2d_k)
            is_los_k = env_rng.random() < p_los_k
            is_los_per_cell.append(is_los_k)

            _spatial_rho = _SCENARIO_SPATIAL_RHO.get(
                _LOS_SCENARIO_MAP.get(self._pathloss_scenario, self._pathloss_scenario)
                if is_los_k else self._pathloss_scenario,
                0.7,
            )

            pl_db, d_3d = self._compute_pathloss(rng, bs_pos, ue_pos, is_los=is_los_k)
            pl_all[k] = pl_db
            rx_power_dbm[k] = self.tx_power_dbm - pl_db

            h_k = _generate_tdl_channel(
                rng=rng,
                num_ofdm_sym=T,
                num_rb=RB,
                num_tx_ant=BS,
                num_rx_ant=UE,
                tau_rms_ns=self.tau_rms_ns,
                subcarrier_spacing_hz=self.subcarrier_spacing,
                doppler_hz=doppler_hz,
                rician_k_linear=rician_k_linear,
                tdl_profile=self._tdl_profile,
                spatial_corr_rho=_spatial_rho,
                tx_panel=_tx_panel,
                rx_panel=_rx_panel,
                xpd_db=self.xpd_db,
            )
            h_all.append(h_k.astype(np.complex64))

        # Apply relative path-gain scaling: serving stays unit-power,
        # interferers scaled by sqrt(P_interf / P_serving)
        serving_idx_tmp = int(np.argmax(rx_power_dbm))
        is_los = is_los_per_cell[serving_idx_tmp] if is_los_per_cell else False
        p_serving_linear = 10.0 ** (rx_power_dbm[serving_idx_tmp] / 10.0)
        for k in range(K):
            if k != serving_idx_tmp:
                p_k_linear = 10.0 ** (rx_power_dbm[k] / 10.0)
                rel_amplitude = math.sqrt(max(p_k_linear / (p_serving_linear + 1e-30), 1e-15))
                h_all[k] = (h_all[k] * rel_amplitude).astype(np.complex64)

        return h_all, rx_power_dbm, noise_power_dbm, is_los

    # ------------------------------------------------------------------
    # Single-sample generation
    # ------------------------------------------------------------------
    def _generate_one_sample(
        self,
        idx: int,
        sites: list[Any],
        *,
        link_override: str | None = None,
        pilot_override: str | None = None,
        paired: bool = False,
        ue_pos_override: np.ndarray | None = None,
        doppler_hz_override: float | None = None,
        last_rt_channels: tuple[list[np.ndarray], np.ndarray, float] | None = None,
        train_intf_ue_positions: np.ndarray | None = None,
    ) -> ChannelSample:
        """Generate one physically meaningful ChannelSample.

        Tries Sionna RT if available, else uses TDL fallback.  In both
        cases applies channel estimation, SSB measurement, and populates
        all ChannelSample fields.
        """
        rng = np.random.default_rng(self._seed + idx)

        effective_link = link_override or self.link
        sample_link: str = effective_link if effective_link in ("UL", "DL") else "DL"
        _default_pilot = self.pilot_type_ul if sample_link == "UL" else self.pilot_type_dl
        effective_pilot = pilot_override or _default_pilot

        # Determine antenna dimensions based on link direction
        if sample_link == "DL":
            BS_ant = self.num_bs_tx_ant
            UE_ant = self.num_ue_rx_ant
        else:
            BS_ant = self.num_bs_rx_ant
            UE_ant = self.num_ue_tx_ant

        T = self.num_ofdm_symbols
        RB = self._num_rb
        sites = sites[: self.num_cells]
        K = len(sites)

        # Place one UE (with retry for RT zero-channel cases)
        if ue_pos_override is not None:
            ue_pos = ue_pos_override.copy()
        elif self.custom_ue_positions:
            pos_idx = idx % len(self.custom_ue_positions)
            p = self.custom_ue_positions[pos_idx]
            ue_pos = np.array(
                [float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("z", self.ue_height_m))],
                dtype=np.float64,
            )
        else:
            ue_positions = self._place_ues(rng, sites, 1)
            ue_pos = ue_positions[0]  # [3]

        # --- Compute channels -----------------------------------------------
        sionna_rt_used = False
        _is_los = False
        _MAX_RT_RETRIES = 5
        # Preserve trajectory position for TDL fallback and metadata
        ue_pos_canonical = ue_pos.copy()

        if self._use_real_sionna:
            rt_ue_pos = ue_pos.copy()
            for _rt_attempt in range(_MAX_RT_RETRIES):
                try:
                    h_all = self._compute_channels_sionna(
                        idx + _rt_attempt * 10000,
                        sites,
                        rt_ue_pos,
                        T,
                        RB,
                        BS_ant,
                        UE_ant,
                    )
                    sionna_rt_used = True
                    noise_power_dbm = self._noise_power_dbm()
                    rx_power_dbm = np.array(
                        [
                            float(
                                10.0 * math.log10(max(float(np.mean(np.abs(h) ** 2)), 1e-30))
                                + self.tx_power_dbm
                            )
                            for h in h_all
                        ],
                        dtype=np.float64,
                    )
                    # Check if serving channel has usable power
                    max_power = max(float(np.mean(np.abs(h) ** 2)) for h in h_all)
                    if max_power > 1e-20:
                        ue_pos = rt_ue_pos
                        self._last_rt_channels = (
                            [h.copy() for h in h_all],
                            rx_power_dbm.copy(),
                            noise_power_dbm,
                        )
                        break
                    if _rt_attempt < _MAX_RT_RETRIES - 1:
                        logger.info(
                            "RT attempt %d/%d for sample %d: channel power %.2e too low, "
                            "retrying with new UE position",
                            _rt_attempt + 1,
                            _MAX_RT_RETRIES,
                            idx,
                            max_power,
                        )
                        if ue_pos_override is None:
                            retry_rng = np.random.default_rng(self._seed + idx + _rt_attempt + 1)
                            cell_radius = self.isd_m / math.sqrt(3.0)
                            num_rings = _sites_to_rings(self.num_cells)
                            network_radius = max(cell_radius, self.isd_m * num_rings + cell_radius)
                            rt_ue_pos = _place_ues_uniform(
                                retry_rng,
                                1,
                                network_radius,
                                self.ue_height_m,
                            )[0]
                    else:
                        if last_rt_channels is not None:
                            logger.info(
                                "RT: all %d attempts produced zero channels for sample %d, "
                                "reusing previous RT channels to preserve temporal consistency.",
                                _MAX_RT_RETRIES, idx,
                            )
                            h_all, rx_power_dbm, noise_power_dbm = last_rt_channels
                            sionna_rt_used = True
                        else:
                            logger.warning(
                                "RT: all %d attempts produced zero channels for sample %d, "
                                "falling back to TDL.",
                                _MAX_RT_RETRIES, idx,
                            )
                            ue_pos = ue_pos_canonical
                            h_all, rx_power_dbm, noise_power_dbm, _is_los = self._compute_channels_tdl(
                                rng, sites, ue_pos, T, RB, BS_ant, UE_ant, sample_link,
                            )
                            sionna_rt_used = False
                except Exception as exc:
                    logger.warning(
                        "Sionna RT failed for sample %d attempt %d (%s)",
                        idx, _rt_attempt, exc,
                    )
                    if _rt_attempt == _MAX_RT_RETRIES - 1:
                        if last_rt_channels is not None:
                            logger.info("All RT attempts failed for sample %d, reusing previous RT channels.", idx)
                            h_all, rx_power_dbm, noise_power_dbm = last_rt_channels
                            sionna_rt_used = True
                        else:
                            logger.warning("All RT attempts failed, falling back to TDL.")
                            ue_pos = ue_pos_canonical
                            h_all, rx_power_dbm, noise_power_dbm, _is_los = self._compute_channels_tdl(
                                rng, sites, ue_pos, T, RB, BS_ant, UE_ant, sample_link,
                            )
                            sionna_rt_used = False
                    break
        else:
            h_all, rx_power_dbm, noise_power_dbm, _is_los = self._compute_channels_tdl(
                rng,
                sites,
                ue_pos,
                T,
                RB,
                BS_ant,
                UE_ant,
                sample_link,
            )

        if not sionna_rt_used:
            noise_power_dbm = self._noise_power_dbm()
            # rx_power_dbm already set by _compute_channels_tdl

        # --- Patch RT interferer channels that are near-zero ----------------
        # RT may find no ray paths for distant cells, giving near-zero
        # channels. Replace with TDL channels at path-loss-model power so
        # interference estimation can produce meaningful SIR gradients.
        if sionna_rt_used and K > 1:
            _s_idx = int(np.argmax(rx_power_dbm))
            _p_serv = 10.0 ** (rx_power_dbm[_s_idx] / 10.0)
            for _k in range(K):
                if _k == _s_idx:
                    continue
                _p_k = 10.0 ** (rx_power_dbm[_k] / 10.0)
                if _p_serv > 1e-30 and _p_k / _p_serv < 1e-5:
                    bs_pos_k = np.asarray(sites[_k].position, dtype=np.float64)
                    _pl_db, _ = self._compute_pathloss(rng, bs_pos_k, ue_pos)
                    _exp_rx = self.tx_power_dbm - _pl_db
                    _exp_lin = 10.0 ** (_exp_rx / 10.0)
                    _rel_amp = math.sqrt(max(_exp_lin / (_p_serv + 1e-30), 1e-15))
                    _tdl_rng = np.random.default_rng(self._seed + idx + _k * 7777)
                    _wl = 3e8 / self.carrier_freq_hz
                    _dop = (self.ue_speed_kmh / 3.6) / _wl
                    _ric = 10.0 ** (self.rician_k_db / 10.0) if self.rician_k_db > -50 else 0.0
                    _tx_p = self.bs_panel if sample_link == "DL" else self.ue_panel
                    _rx_p = self.ue_panel if sample_link == "DL" else self.bs_panel
                    _h_tdl = _generate_tdl_channel(
                        rng=_tdl_rng,
                        num_ofdm_sym=T,
                        num_rb=RB,
                        num_tx_ant=BS_ant,
                        num_rx_ant=UE_ant,
                        tau_rms_ns=self.tau_rms_ns,
                        subcarrier_spacing_hz=self.subcarrier_spacing,
                        doppler_hz=_dop,
                        rician_k_linear=_ric,
                        tdl_profile=self._tdl_profile,
                        tx_panel=_tx_p,
                        rx_panel=_rx_p,
                        xpd_db=self.xpd_db,
                    )
                    _h_tdl_pw = float(np.mean(np.abs(_h_tdl) ** 2))
                    _serv_pw = float(np.mean(np.abs(h_all[_s_idx]) ** 2))
                    if _h_tdl_pw > 1e-30 and _serv_pw > 1e-30:
                        _h_tdl = _h_tdl * np.sqrt(_serv_pw / _h_tdl_pw)
                    h_all[_k] = (_h_tdl * _rel_amp).astype(np.complex64)
                    rx_power_dbm[_k] = _exp_rx

        # --- Select serving cell (strongest received power) ------------------
        serving_idx = int(np.argmax(rx_power_dbm))
        serving_pci = getattr(sites[serving_idx], "pci", serving_idx)

        h_serving_raw = h_all[serving_idx]  # [T, RB, BS, UE] complex64

        # Unit-normalize serving channel (pathloss tracked via rx_power_dbm)
        serving_power = float(np.mean(np.abs(h_serving_raw) ** 2))
        if serving_power > 1e-30:
            h_serving_true = (h_serving_raw / np.sqrt(serving_power)).astype(np.complex64)
        else:
            h_serving_true = h_serving_raw

        # Normalize interferer channels by same factor for consistent SIR
        h_all_norm = [h / np.sqrt(serving_power) for h in h_all] if serving_power > 1e-30 else h_all

        # --- Interferers -----------------------------------------------------
        interferer_indices = [k for k in range(K) if k != serving_idx]
        if len(interferer_indices) > 0:
            h_interferers: np.ndarray | None = np.stack(
                [h_all_norm[k] for k in interferer_indices], axis=0
            ).astype(np.complex64)
        else:
            h_interferers = None

        # --- Place interfering UEs (positions shared by precoding + UL intf) --
        intf_ue_positions_per_cell: list[np.ndarray] = []
        if len(interferer_indices) > 0 and self.num_interfering_ues > 0:
            cell_radius = self.isd_m / math.sqrt(3.0)
            for ki, k in enumerate(interferer_indices):
                intf_bs_pos = np.asarray(sites[k].position, dtype=np.float64)
                if train_intf_ue_positions is not None:
                    _positions = train_intf_ue_positions[:self.num_interfering_ues]
                else:
                    _positions = _place_ues_uniform(
                        rng, max(self.num_interfering_ues, 1), cell_radius,
                        self.ue_height_m, center=intf_bs_pos[:2],
                    )
                intf_ue_positions_per_cell.append(_positions)

        # --- Per-interferer DL precoding projection ---------------------------
        intf_ranks: list[int] | None = None
        if (
            self.apply_interferer_precoding
            and h_interferers is not None
            and len(intf_ue_positions_per_cell) > 0
        ):
            from msg_embedding.phy_sim.precoding import project_interference_channels

            _wl_prec = 3e8 / self.carrier_freq_hz
            _dop_prec = (self.ue_speed_kmh / 3.6) / _wl_prec
            h_bs_to_own: list[np.ndarray] = []
            for ki, k in enumerate(interferer_indices):
                intf_bs_pos = np.asarray(sites[k].position, dtype=np.float64)
                sched_idx = int(rng.integers(len(intf_ue_positions_per_cell[ki])))
                sched_ue_pos = intf_ue_positions_per_cell[ki][sched_idx]
                pl_db_own, _ = self._compute_pathloss(rng, intf_bs_pos, sched_ue_pos)
                own_seed = self._seed + idx + 5000 + ki
                _own_rng = np.random.default_rng(own_seed)
                h_dl_own = _generate_tdl_channel(
                    rng=_own_rng,
                    num_ofdm_sym=T,
                    num_rb=RB,
                    num_tx_ant=BS_ant,
                    num_rx_ant=UE_ant,
                    tau_rms_ns=self.tau_rms_ns,
                    subcarrier_spacing_hz=self.subcarrier_spacing,
                    doppler_hz=_dop_prec,
                    rician_k_linear=0.0,
                    tdl_profile=self._tdl_profile,
                    tx_panel=self.bs_panel,
                    rx_panel=self.ue_panel,
                    xpd_db=self.xpd_db,
                )
                h_bs_to_own.append(h_dl_own)

            h_interferers, intf_ranks = project_interference_channels(
                h_interferers,
                h_bs_to_own,
                max_rank=self.max_rank,
                rank_threshold=self.rank_threshold,
            )
            del h_bs_to_own

        # --- UL interferers: independent per-UE channels H(UE_kn → BS_serving)
        h_ul_intf_per_ue: list[np.ndarray] | None = None
        if len(interferer_indices) > 0 and self.num_interfering_ues > 0:
            serving_bs_pos = np.asarray(sites[serving_idx].position, dtype=np.float64)
            p_serving_linear = 10.0 ** (rx_power_dbm[serving_idx] / 10.0)
            _wl = 3e8 / self.carrier_freq_hz
            _dop = (self.ue_speed_kmh / 3.6) / _wl
            _ric = 10.0 ** (self.rician_k_db / 10.0) if self.rician_k_db > -50 else 0.0
            h_ul_intf_per_ue = []
            for ki, k in enumerate(interferer_indices):
                intf_bs_pos = np.asarray(sites[k].position, dtype=np.float64)
                intf_ue_positions = intf_ue_positions_per_cell[ki]
                ue_channels = []
                for n in range(self.num_interfering_ues):
                    ue_n_pos = intf_ue_positions[n]
                    pl_db_n, _ = self._compute_pathloss(rng, serving_bs_pos, ue_n_pos)
                    rx_pwr_n = self.tx_power_dbm - pl_db_n
                    rel_amp_n = math.sqrt(
                        max(10.0 ** (rx_pwr_n / 10.0) / (p_serving_linear + 1e-30), 1e-15)
                    )
                    ue_seed = (self._seed + idx + K + ki * self.num_interfering_ues + n)
                    _tdl_rng = np.random.default_rng(ue_seed)
                    h_ue_n = _generate_tdl_channel(
                        rng=_tdl_rng,
                        num_ofdm_sym=T,
                        num_rb=RB,
                        num_tx_ant=UE_ant,
                        num_rx_ant=BS_ant,
                        tau_rms_ns=self.tau_rms_ns,
                        subcarrier_spacing_hz=self.subcarrier_spacing,
                        doppler_hz=_dop,
                        rician_k_linear=_ric,
                        tdl_profile=self._tdl_profile,
                        tx_panel=self.ue_panel,
                        rx_panel=self.bs_panel,
                        xpd_db=self.xpd_db,
                    )
                    if serving_power > 1e-30:
                        h_ue_n = h_ue_n / np.sqrt(serving_power)
                    h_ue_n = (h_ue_n * rel_amp_n).astype(np.complex64)
                    ue_channels.append(h_ue_n)
                h_ul_intf_per_ue.append(
                    np.stack(ue_channels, axis=0)  # [N_ues, T, RB, UE_ant, BS_ant]
                )

        # --- SNR / SIR / SINR ------------------------------------------------
        p_serving_dbm = float(rx_power_dbm[serving_idx])
        snr_db = p_serving_dbm - noise_power_dbm

        if len(interferer_indices) > 0:
            p_interf_linear = np.sum(10.0 ** (rx_power_dbm[interferer_indices] / 10.0))
            p_interf_dbm = 10.0 * math.log10(max(p_interf_linear, 1e-30))
            sir_db = p_serving_dbm - p_interf_dbm
        else:
            sir_db = 49.9

        sinr_db = -10.0 * math.log10(10.0 ** (-snr_db / 10.0) + 10.0 ** (-sir_db / 10.0))

        _tx_power_offset = self.tx_power_dbm - self.ue_tx_power_dbm
        ul_snr_db_val = snr_db - _tx_power_offset
        ul_sinr_db_val = -10.0 * math.log10(
            10.0 ** (-ul_snr_db_val / 10.0) + 10.0 ** (-sir_db / 10.0)
        )

        snr_db = _clamp_db(snr_db)
        sir_db = _clamp_db(sir_db)
        sinr_db = _clamp_db(sinr_db)
        ul_snr_db_val = _clamp_db(ul_snr_db_val)
        ul_sinr_db_val = _clamp_db(ul_sinr_db_val)

        # --- Interference-aware channel estimation ----------------------------
        from msg_embedding.data.sources._interference_estimation import (
            estimate_channel_with_interference,
            estimate_paired_channels,
        )

        _wl_doppler = 3e8 / self.carrier_freq_hz
        _doppler_hz = (self.ue_speed_kmh / 3.6) / _wl_doppler
        if doppler_hz_override is not None:
            _doppler_hz = doppler_hz_override

        intf_cell_ids = (
            [getattr(sites[k], "pci", k) for k in interferer_indices]
            if interferer_indices
            else None
        )
        n_intf_ues = self.num_interfering_ues

        # SRS frequency-domain position — accumulate full hopping cycle
        from msg_embedding.ref_signals.srs import srs_accumulated_rb_indices as _srs_rb_fn

        tdd = self._tdd_pattern
        slot_idx = idx % tdd.period_slots
        symbol_map = tdd.symbol_map(slot_idx)
        dl_symbol_mask = np.array([s == "D" for s in symbol_map], dtype=bool)
        ul_symbol_mask = np.array([s == "U" for s in symbol_map], dtype=bool)
        ul_syms_for_srs = [i for i, m in enumerate(symbol_map) if m == "U"]
        _srs_sym = ul_syms_for_srs[-1] if ul_syms_for_srs else 0
        srs_rb_idx = _srs_rb_fn(self._srs_resource_cfg, idx, _srs_sym, RB)

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
                tau_rms_ns=self.tau_rms_ns,
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
                srs_resource_cfg=self._srs_resource_cfg,
                doppler_hz=_doppler_hz,
                ul_snr_dB=ul_snr_db_val,
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
            _srs_rb = None
            if effective_pilot == "srs_zc":
                _srs_rb = srs_rb_idx
                pilots = _generate_pilots_srs(
                    RB, serving_pci,
                    slot=slot_idx, symbol=_srs_sym,
                    group_hopping=self.srs_group_hopping,
                    sequence_hopping=self.srs_sequence_hopping,
                    K_TC=self.srs_comb,
                    srs_num_rb=len(_srs_rb),
                )
            else:
                pilots = _generate_pilots_csirs(RB, serving_pci)

            direction: str = "UL" if sample_link == "UL" else "DL"
            _sym_mask = ul_symbol_mask if direction == "UL" else dl_symbol_mask
            _est_snr = ul_snr_db_val if direction == "UL" else snr_db

            if self.channel_est_mode == "ls_hop_concat" and direction == "UL":
                from msg_embedding.data.sources._interference_estimation import (
                    estimate_channel_hop_concat,
                )
                est_result = estimate_channel_hop_concat(
                    h_serving_true=h_serving_true,
                    h_interferers=h_interferers,
                    interferer_cell_ids=intf_cell_ids,
                    direction="UL",
                    snr_dB=_est_snr,
                    rng=rng,
                    srs_resource_cfg=self._srs_resource_cfg,
                    current_slot=slot_idx,
                    srs_symbol=_srs_sym,
                    doppler_hz=_doppler_hz,
                    subcarrier_spacing_hz=self.subcarrier_spacing,
                    total_rb=RB,
                    serving_cell_id=serving_pci,
                    num_interfering_ues=n_intf_ues,
                    srs_group_hopping=self.srs_group_hopping,
                    srs_sequence_hopping=self.srs_sequence_hopping,
                    srs_K_TC=self.srs_comb,
                    h_interferers_ul_per_ue=h_ul_intf_per_ue,
                )
            else:
                _est_mode = "ls_linear" if (self.channel_est_mode == "ls_hop_concat" and direction == "DL") else self.channel_est_mode
                est_result = estimate_channel_with_interference(
                    h_serving_true=h_serving_true,
                    h_interferers=h_interferers,
                    pilots_serving=pilots,
                    interferer_cell_ids=intf_cell_ids,
                    direction=direction,  # type: ignore[arg-type]
                    snr_dB=_est_snr,
                    rng=rng,
                    est_mode=_est_mode,
                    tau_rms_ns=self.tau_rms_ns,
                    subcarrier_spacing=self.subcarrier_spacing,
                    serving_cell_id=serving_pci,
                    num_interfering_ues=n_intf_ues,
                    valid_symbol_mask=_sym_mask,
                    srs_rb_indices=_srs_rb,
                    srs_slot=slot_idx,
                    srs_symbol=_srs_sym if direction == "UL" else 0,
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
            prec = compute_dl_precoding(np.conj(h_serving_est), max_rank=_max_r)
            w_dl_out = prec.w_dl
            rank_out = prec.rank

        # Legacy interference signal (observed at receiver)
        interference_signal: np.ndarray | None = None
        if h_interferers is not None:
            interf_sum = np.sum(h_interferers, axis=0)  # [T, RB, BS, UE]
            interf_signal_raw = np.sum(interf_sum, axis=2)  # [T, RB, UE]
            interference_signal = interf_signal_raw.astype(np.complex64)

        # --- SSB measurements (multi-cell beam sweep) -------------------------
        ssb_rsrp: list[float] | None = None
        ssb_rsrq: list[float] | None = None
        ssb_sinr: list[float] | None = None
        ssb_best_beam: list[int] | None = None
        ssb_pcis_list: list[int] | None = None
        ssb_rsrp_true: list[float] | None = None
        ssb_sinr_true: list[float] | None = None

        if self.enable_ssb and K >= 1:
            try:
                from msg_embedding.phy_sim.ssb_measurement import SSBMeasurement

                ssb_meas = SSBMeasurement(
                    num_beams=self.num_ssb_beams,
                    num_bs_ant=BS_ant,
                    ref_power_offset_dBm=30.0,
                )
                all_pcis = [getattr(s, "pci", i) for i, s in enumerate(sites)]
                noise_power_lin = 10.0 ** (noise_power_dbm / 10.0) * 1e-3

                h_ssb = []
                for k in range(K):
                    rx_lin = 10.0 ** (rx_power_dbm[k] / 10.0) * 1e-3
                    h_ssb.append((h_all_norm[k] * np.sqrt(rx_lin)).astype(np.complex64))

                ssb_result = ssb_meas.measure(
                    h_per_cell=h_ssb,
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
                        h_per_cell=h_ssb,
                        pcis=all_pcis,
                        noise_power_lin=1e-30,
                    )
                    ssb_rsrp_true = ssb_result_ideal.rsrp_dBm.tolist()
                    ssb_sinr_true = ssb_result_ideal.ss_sinr_dB.tolist()
            except Exception as exc:
                logger.warning("SSB measurement failed: %s", exc)

        # --- SRS-based Pre-SINR ------------------------------------------------
        pre_sinr_db_out: float | None = None
        pre_sinr_per_rb_out: np.ndarray | None = None
        _h_ul_t = h_ul_true_out if h_ul_true_out is not None else (
            h_serving_true if sample_link == "UL" else None
        )
        _h_ul_e = h_ul_est_out if h_ul_est_out is not None else (
            h_serving_est if sample_link == "UL" else None
        )
        if _h_ul_t is not None and _h_ul_e is not None:
            _sig_per_rb = np.mean(np.abs(_h_ul_t) ** 2, axis=(0, 2, 3))
            _err = _h_ul_e - _h_ul_t
            _err_per_rb = np.mean(np.abs(_err) ** 2, axis=(0, 2, 3))
            _pre_sinr_linear = _sig_per_rb / (_err_per_rb + 1e-30)
            pre_sinr_per_rb_out = np.clip(
                10.0 * np.log10(_pre_sinr_linear + 1e-30), -50.0, 50.0
            ).astype(np.float32)
            _wb_sig = float(np.mean(_sig_per_rb))
            _wb_err = float(np.mean(_err_per_rb))
            pre_sinr_db_out = float(np.clip(
                10.0 * math.log10(_wb_sig / (_wb_err + 1e-30) + 1e-30), -50.0, 50.0
            ))

        # --- Assemble meta dict -----------------------------------------------
        meta = {
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
            "ue_tx_power_dbm": self.ue_tx_power_dbm,
            "ue_speed_kmh": self.ue_speed_kmh,
            "mobility_mode": self.mobility_mode,
            "ue_distribution": self.ue_distribution,
            "pilot_type": effective_pilot,
            "scenario": self.scenario,
            "channel_model": self.channel_model_name,
            "tdd_pattern": self.tdd_pattern_name,
            "srs_group_hopping": self.srs_group_hopping,
            "srs_sequence_hopping": self.srs_sequence_hopping,
            "srs_periodicity": self.srs_periodicity,
            "srs_b_hop": self.srs_b_hop,
            "srs_comb": self.srs_comb,
            "srs_c_srs": self.srs_c_srs,
            "srs_b_srs": self.srs_b_srs,
            "srs_n_rrc": self.srs_n_rrc,
            "srs_rb_indices": srs_rb_idx.tolist(),
            "num_ofdm_symbols": T,
            "num_rb": RB,
            "noise_figure_db": self.noise_figure_db,
            "noise_power_dbm": noise_power_dbm,
            "serving_cell_index": serving_idx,
            "serving_pci": serving_pci,
            "pathloss_dB": float(self.tx_power_dbm - rx_power_dbm[serving_idx]),
            "rx_power_serving_dbm": float(p_serving_dbm),
            "distance_3d_m": float(
                np.linalg.norm(np.asarray(sites[serving_idx].position) - ue_pos)
            ),
            "custom_positions": self.custom_site_positions is not None,
            "num_ues": self.num_ues,
            "fft_size": self.fft_size,
            "device": self.device,
            "sample_index": int(idx),
            "seed": self._seed,
            "mock": False,
            "sionna_rt_used": sionna_rt_used,
            "channel_generation_mode": "sionna_rt" if sionna_rt_used else "tdl_fallback",
            "is_los": bool(_is_los),
            "bs_panel": [self.bs_panel.n_h, self.bs_panel.n_v, self.bs_panel.n_p] if self.bs_panel else None,
            "ue_panel": [self.ue_panel.n_h, self.ue_panel.n_v, self.ue_panel.n_p] if self.ue_panel else None,
            "xpd_db": self.xpd_db,
            "dl_rank": rank_out,
            "interferer_precoding_applied": self.apply_interferer_precoding,
            "store_interferer_channels": self.store_interferer_channels,
        }
        if intf_ranks is not None:
            meta["interferer_ranks"] = intf_ranks
        if w_dl_out is not None:
            meta["w_dl_shape"] = list(w_dl_out.shape)
            meta["w_dl"] = w_dl_out

        return ChannelSample(
            h_serving_true=h_serving_true,
            h_serving_est=h_serving_est,
            h_interferers=h_interferers if self.store_interferer_channels else None,
            interference_signal=interference_signal if self.store_interferer_channels else None,
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
            ul_pre_sinr_dB=pre_sinr_db_out,
            ul_pre_sinr_per_rb=pre_sinr_per_rb_out,
            ul_snr_dB=ul_snr_db_val,
            ul_sinr_dB=ul_sinr_db_val,
            ssb_rsrp_true_dBm=ssb_rsrp_true,
            ssb_sinr_true_dB=ssb_sinr_true,
            w_dl=w_dl_out,
            dl_rank=rank_out,
            source="sionna_rt",
            sample_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc),
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Iterator interface
    # ------------------------------------------------------------------
    def iter_samples(self) -> Iterator[ChannelSample]:
        sites = self._build_topology()

        # -- Mobility: generate trajectory if mode is not static ---------------
        trajectory_positions: np.ndarray | None = None
        trajectory_dopplers: np.ndarray | None = None
        train_ue_positions: np.ndarray | None = None
        # F-005: static mode uses a single fixed position for all snapshots,
        # generated deterministically from self._seed so results are reproducible.
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
        else:
            from ._mobility import compute_doppler_from_trajectory, generate_trajectory

            traj_rng = np.random.default_rng(self._seed + 9999)

            mob_mode = self.mobility_mode
            mob_kwargs: dict = {}
            is_hsr = self.topology_layout == "linear" and mob_mode in ("linear", "track")
            if is_hsr:
                mob_mode = "track"
                unique_site_ids = sorted({s.site_id for s in sites})
                track_centerline = []
                for sid in unique_site_ids:
                    pos = next(s.position for s in sites if s.site_id == sid)
                    track_centerline.append([pos[0], 0.0])
                mob_kwargs["track_waypoints"] = np.array(track_centerline, dtype=np.float64)
                init_pos = np.array([track_centerline[0][0], 0.0, self.ue_height_m], dtype=np.float64)
            else:
                init_pos = self._place_ues(traj_rng, sites, 1)[0]

            cell_radius = self.isd_m / math.sqrt(3.0)
            num_rings = _sites_to_rings(self.num_cells)
            network_radius = max(cell_radius, self.isd_m * num_rings + cell_radius)

            trajectory_positions = generate_trajectory(
                rng=traj_rng,
                start_pos=init_pos,
                speed_kmh=self.ue_speed_kmh,
                num_steps=self.num_samples,
                dt_s=self.sample_interval_s,
                mode=mob_mode,
                boundary_radius_m=network_radius,
                boundary_center=np.zeros(2, dtype=np.float64),
                **mob_kwargs,
            )

            # HSR: generate train UE positions for all passengers
            train_ue_positions: np.ndarray | None = None
            if is_hsr and self.num_interfering_ues > 0:
                from ._mobility import generate_train_positions
                train_rng = np.random.default_rng(self._seed + 6666)
                train_ue_positions = generate_train_positions(
                    base_trajectory=trajectory_positions,
                    num_ues=self.num_interfering_ues,
                    rng=train_rng,
                )

            # Doppler: per-sample relative to nearest cell
            trajectory_dopplers = np.zeros(self.num_samples, dtype=np.float64)
            all_site_pos = np.array([s.position[:2] for s in sites], dtype=np.float64)
            for t_idx in range(self.num_samples):
                ue_xy = trajectory_positions[t_idx, :2]
                dists = np.linalg.norm(all_site_pos - ue_xy[None, :], axis=1)
                nearest_site_idx = int(np.argmin(dists))
                nearest_pos = np.asarray(sites[nearest_site_idx].position, dtype=np.float64)
                t_slice = max(0, t_idx - 1)
                sub_traj = trajectory_positions[t_slice:t_idx + 2]
                if len(sub_traj) >= 2:
                    d_before = np.linalg.norm(sub_traj[0, :2] - nearest_pos[:2])
                    d_after = np.linalg.norm(sub_traj[-1, :2] - nearest_pos[:2])
                    v_radial = (d_after - d_before) / (self.sample_interval_s * (len(sub_traj) - 1))
                    trajectory_dopplers[t_idx] = -v_radial / (3e8 / self.carrier_freq_hz)
                else:
                    trajectory_dopplers[t_idx] = 0.0

        self._last_rt_channels: tuple[list[np.ndarray], np.ndarray, float] | None = None

        for idx in range(self.num_samples):
            ue_pos_ov: np.ndarray | None = None
            doppler_ov: float | None = None
            if static_pos is not None:
                ue_pos_ov = static_pos
            elif trajectory_positions is not None:
                ue_pos_ov = trajectory_positions[idx]
            if trajectory_dopplers is not None:
                doppler_ov = float(np.abs(trajectory_dopplers[idx]))

            intf_pos_ov: np.ndarray | None = None
            if train_ue_positions is not None:
                intf_pos_ov = train_ue_positions[:, idx, :]

            if self.link == "BOTH":
                yield self._generate_one_sample(
                    idx,
                    sites,
                    paired=True,
                    ue_pos_override=ue_pos_ov,
                    doppler_hz_override=doppler_ov,
                    last_rt_channels=self._last_rt_channels,
                    train_intf_ue_positions=intf_pos_ov,
                )
            else:
                yield self._generate_one_sample(
                    idx,
                    sites,
                    ue_pos_override=ue_pos_ov,
                    doppler_hz_override=doppler_ov,
                    last_rt_channels=self._last_rt_channels,
                    train_intf_ue_positions=intf_pos_ov,
                )

    def describe(self) -> dict[str, Any]:
        return {
            "source": "sionna_rt",
            "available": _SIONNA_AVAILABLE,
            "sionna_available": _SIONNA_AVAILABLE,
            "use_real_sionna": self._use_real_sionna,
            "scenario": self.scenario,
            "num_cells": self.num_cells,
            "num_ues": self.num_ues,
            "expected_sample_count": self.num_samples,
            "pilot_type_dl": self.pilot_type_dl,
            "pilot_type_ul": self.pilot_type_ul,
            "pilot_type_dl": self.pilot_type_dl,
            "pilot_type_ul": self.pilot_type_ul,
            "channel_est_mode": self.channel_est_mode,
            "channel_model": self.channel_model_name,
            "tdd_pattern": self.tdd_pattern_name,
            "link": self.link,
            "device": self.device,
            "dimensions": {
                "T": self.num_ofdm_symbols,
                "RB": self._num_rb,
                "BS_ant": self.num_bs_ant,
                "UE_ant": self.num_ue_ant,
                "K": self.num_cells,
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
            "srs": {
                "group_hopping": self.srs_group_hopping,
                "sequence_hopping": self.srs_sequence_hopping,
                "periodicity": self.srs_periodicity,
                "b_hop": self.srs_b_hop,
            },
            "ssb": {
                "enable": self.enable_ssb,
                "num_beams": self.num_ssb_beams,
            },
            "notes": (
                "Sionna 2.0 limitation: scene.tx_array is globally shared; "
                "all BS use the same PlanarArray. "
                "When Sionna is not installed, TDL-based 38.901 channel "
                "generation is used as fallback."
            ),
        }


# ---------------------------------------------------------------------------
# Mock source (Sionna-free structural fallback)
# ---------------------------------------------------------------------------


@register_source
class SionnaRTMockSource(DataSource):
    """Numpy-only structural mock that mirrors :class:`SionnaRTSource`'s output.

    Useful for unit tests on CI hosts where Sionna is not installed.
    The generated samples carry ``meta['mock'] = True`` so downstream code
    can filter them out.  No physics is reproduced.
    """

    name = "sionna_rt_mock"

    # ------------------------------------------------------------------
    def validate_config(self) -> None:
        cfg = self.config
        self.num_samples: int = int(_dict_get(cfg, "num_samples", 1))

        T_default, RB_default, BS_default, UE_default = 2, 8, 4, 2
        self.T: int = int(_dict_get(cfg, "num_ofdm_symbols", T_default))
        self.RB: int = int(_dict_get(cfg, "num_rb", RB_default))

        # -- Cell / site count ------------------------------------------------
        _num_sites_raw = _dict_get(cfg, "num_sites", None)
        _num_cells_raw = _dict_get(cfg, "num_cells", None)
        if _num_sites_raw is not None:
            self.num_cells: int = int(_num_sites_raw)
        elif _num_cells_raw is not None:
            self.num_cells = int(_num_cells_raw)
        else:
            self.num_cells = 2

        # -- Topology params ---------------------------------------------------
        self.isd_m: float = float(_dict_get(cfg, "isd_m", 500.0))
        self.sectors_per_site: int = int(_dict_get(cfg, "sectors_per_site", 3))
        self.tx_height_m: float = float(_dict_get(cfg, "tx_height_m", 25.0))

        # -- Antenna arrays (split TX/RX; legacy kept for backward compat) ----
        _legacy_bs = int(_dict_get(cfg, "num_bs_ant", BS_default))
        _legacy_ue = int(_dict_get(cfg, "num_ue_ant", UE_default))
        self.num_bs_tx_ant: int = int(_dict_get(cfg, "num_bs_tx_ant", _legacy_bs))
        self.num_bs_rx_ant: int = int(_dict_get(cfg, "num_bs_rx_ant", _legacy_bs))
        self.num_ue_tx_ant: int = int(_dict_get(cfg, "num_ue_tx_ant", _legacy_ue))
        self.num_ue_rx_ant: int = int(_dict_get(cfg, "num_ue_rx_ant", _legacy_ue))
        self.BS: int = max(self.num_bs_tx_ant, self.num_bs_rx_ant)
        self.UE: int = max(self.num_ue_tx_ant, self.num_ue_rx_ant)

        # -- Power / mobility --------------------------------------------------
        self.tx_power_dbm: float = float(_dict_get(cfg, "tx_power_dbm", 43.0))
        self.ue_tx_power_dbm: float = float(_dict_get(cfg, "ue_tx_power_dbm", 23.0))
        self.ue_speed_kmh: float = float(_dict_get(cfg, "ue_speed_kmh", 3.0))
        self.ue_distribution: str = str(_dict_get(cfg, "ue_distribution", "uniform"))

        # -- Custom positions --------------------------------------------------
        self.custom_site_positions: list[dict[str, float]] | None = _dict_get(
            cfg, "custom_site_positions", None
        )
        self.custom_ue_positions: list[dict[str, float]] | None = _dict_get(
            cfg, "custom_ue_positions", None
        )

        self.serving_cell_id: int = int(_dict_get(cfg, "serving_cell_id", 0))
        self.channel_est_mode: str = str(_dict_get(cfg, "channel_est_mode", "ls_linear"))
        if self.channel_est_mode not in {"ideal", "ls_linear", "ls_mmse", "ls_hop_concat"}:
            raise ValueError(f"Unknown channel_est_mode {self.channel_est_mode!r}.")
        self.link: str = str(_dict_get(cfg, "link", "DL")).upper()
        if self.link not in {"UL", "DL", "BOTH"}:
            raise ValueError(f"link must be 'UL', 'DL', or 'both', got {self.link!r}.")
        self.scenario: str = str(_dict_get(cfg, "scenario", "munich"))
        _pf = str(_dict_get(cfg, "pilot_type", "csi_rs_gold"))
        self.pilot_type_dl: str = str(_dict_get(cfg, "pilot_type_dl", _pf))
        self.pilot_type_ul: str = str(_dict_get(cfg, "pilot_type_ul",
                                                 _pf if _pf == "srs_zc" else "srs_zc"))
        self.num_ues: int = int(_dict_get(cfg, "num_ues", 1))
        self.snr_dB: float = float(_dict_get(cfg, "snr_dB", _DEFAULT_SNR_DB))
        self.sir_dB: float = float(_dict_get(cfg, "sir_dB", _DEFAULT_SIR_DB))
        self.noise_power_dBm: float = float(_dict_get(cfg, "noise_power_dBm", _DEFAULT_NOISE_DBM))
        self._seed: int = int(_dict_get(cfg, "seed", 0))
        if not (0 <= self.serving_cell_id < self.num_cells):
            raise ValueError(
                f"serving_cell_id={self.serving_cell_id} out of range " f"[0, {self.num_cells})."
            )

        # -- New frontend params (channel model, TDD, SRS, SSB) ---------------
        self.channel_model_name: str = (
            str(_dict_get(cfg, "channel_model", "TDL-C")).upper().replace("_", "-")
        )
        self.tdd_pattern_name: str = str(_dict_get(cfg, "tdd_pattern", "DDDSU"))
        self.srs_group_hopping: bool = bool(_dict_get(cfg, "srs_group_hopping", False))
        self.srs_sequence_hopping: bool = bool(_dict_get(cfg, "srs_sequence_hopping", False))
        self.srs_periodicity: int = int(_dict_get(cfg, "srs_periodicity", 10))
        self.srs_b_hop: int = int(_dict_get(cfg, "srs_b_hop", 0))
        self.srs_comb: int = int(_dict_get(cfg, "srs_comb", 2))
        self.srs_c_srs: int = int(_dict_get(cfg, "srs_c_srs", 3))
        self.srs_b_srs: int = int(_dict_get(cfg, "srs_b_srs", 1))
        self.srs_n_rrc: int = int(_dict_get(cfg, "srs_n_rrc", 0))
        from msg_embedding.ref_signals.srs import SRSResourceConfig as _SRSResCfg

        self._srs_resource_cfg = _SRSResCfg(
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
        self.num_ssb_beams: int = int(_dict_get(cfg, "num_ssb_beams", 8))
        self.enable_ssb: bool = bool(_dict_get(cfg, "enable_ssb", True))

    # ------------------------------------------------------------------
    def _randn_c64(self, rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
        real = rng.standard_normal(shape, dtype=np.float32)
        imag = rng.standard_normal(shape, dtype=np.float32)
        return (real + 1j * imag).astype(np.complex64)

    def _generate_one_sample(
        self,
        idx: int,
        *,
        link_override: str | None = None,
        pilot_override: str | None = None,
        paired: bool = False,
    ) -> ChannelSample:
        rng = np.random.default_rng(self._seed + idx)
        shape = (self.T, self.RB, self.BS, self.UE)

        effective_link = link_override or self.link
        effective_pilot = pilot_override or self.pilot_type_dl

        h_true = self._randn_c64(rng, shape)
        noise_scale = 10.0 ** (-self.snr_dB / 20.0)
        h_est = (h_true + noise_scale * self._randn_c64(rng, shape)).astype(np.complex64)

        if self.num_cells > 1:
            interf_shape = (self.num_cells - 1, *shape)
            h_interferers: np.ndarray | None = self._randn_c64(rng, interf_shape)
        else:
            h_interferers = None

        sinr = float(
            10.0 * np.log10(1.0 / (10 ** (-self.snr_dB / 10.0) + 10 ** (-self.sir_dB / 10.0)))
        )
        sinr = float(np.clip(sinr, -49.9, 49.9))

        # Paired mode: generate UL/DL channels for the mock
        h_ul_true_out: np.ndarray | None = None
        h_ul_est_out: np.ndarray | None = None
        h_dl_true_out: np.ndarray | None = None
        h_dl_est_out: np.ndarray | None = None
        ul_sir_db_out: float | None = None
        dl_sir_db_out: float | None = None
        n_intf_ues_out: int | None = None
        link_pairing = "single"

        if paired:
            link_pairing = "paired"
            h_dl_true_out = h_true
            h_dl_est_out = h_est
            # TDD reciprocity → transpose back to contract [T, RB, BS, UE]
            h_ul_true_out = np.conj(h_true.transpose(0, 1, 3, 2)).transpose(
                0, 1, 3, 2
            ).astype(np.complex64)
            ul_shape = h_ul_true_out.shape  # [T, RB, BS, UE]
            h_ul_est_out = (
                h_ul_true_out + noise_scale * self._randn_c64(rng, ul_shape)
            ).astype(np.complex64)
            ul_sir_db_out = float(np.clip(self.sir_dB - 3.0, -49.9, 49.9))
            dl_sir_db_out = float(np.clip(self.sir_dB, -49.9, 49.9))
            n_intf_ues_out = max(1, self.num_ues - 1) if self.num_cells > 1 else None

        meta = {
            "num_cells": self.num_cells,
            "isd_m": self.isd_m,
            "sectors_per_site": self.sectors_per_site,
            "num_bs_tx_ant": self.num_bs_tx_ant,
            "num_bs_rx_ant": self.num_bs_rx_ant,
            "num_ue_tx_ant": self.num_ue_tx_ant,
            "num_ue_rx_ant": self.num_ue_rx_ant,
            "carrier_freq_hz": float(_dict_get(self.config, "carrier_freq_hz", 3.5e9)),
            "bandwidth_hz": float(_dict_get(self.config, "bandwidth_hz", 100e6)),
            "tx_power_dbm": self.tx_power_dbm,
            "ue_tx_power_dbm": self.ue_tx_power_dbm,
            "ue_speed_kmh": self.ue_speed_kmh,
            "ue_distribution": self.ue_distribution,
            "pilot_type": effective_pilot,
            "scenario": self.scenario,
            "channel_model": self.channel_model_name,
            "tdd_pattern": self.tdd_pattern_name,
            "srs_group_hopping": self.srs_group_hopping,
            "srs_sequence_hopping": self.srs_sequence_hopping,
            "srs_periodicity": self.srs_periodicity,
            "srs_b_hop": self.srs_b_hop,
            "srs_comb": self.srs_comb,
            "srs_c_srs": self.srs_c_srs,
            "srs_b_srs": self.srs_b_srs,
            "srs_n_rrc": self.srs_n_rrc,
            "custom_positions": self.custom_site_positions is not None,
            "sample_index": int(idx),
            "mock": True,
            "sionna_rt_used": False,
        }

        sample_link: str = effective_link if effective_link in ("UL", "DL") else "DL"
        _mock_snr = float(np.clip(self.snr_dB, -49.9, 49.9))
        _mock_tx_offset = self.tx_power_dbm - self.ue_tx_power_dbm
        _mock_ul_snr = float(np.clip(_mock_snr - _mock_tx_offset, -49.9, 49.9))
        return ChannelSample(
            h_serving_true=h_true,
            h_serving_est=h_est,
            h_interferers=h_interferers,
            noise_power_dBm=self.noise_power_dBm,
            snr_dB=_mock_snr,
            sir_dB=float(np.clip(self.sir_dB, -49.9, 49.9)),
            sinr_dB=sinr,
            link=sample_link,  # type: ignore[arg-type]
            channel_est_mode=self.channel_est_mode,  # type: ignore[arg-type]
            serving_cell_id=self.serving_cell_id,
            ue_position=np.asarray([0.0, 0.0, 1.5], dtype=np.float64),
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
            ul_snr_dB=_mock_ul_snr,
            ul_sinr_dB=float(np.clip(sinr - _mock_tx_offset, -49.9, 49.9)),
            w_dl=None,
            dl_rank=None,
            source="sionna_rt",
            sample_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc),
            meta=meta,
        )

    def iter_samples(self) -> Iterator[ChannelSample]:
        for idx in range(self.num_samples):
            if self.link == "BOTH":
                yield self._generate_one_sample(idx, paired=True)
            else:
                yield self._generate_one_sample(idx)

    def describe(self) -> dict[str, Any]:
        return {
            "source": "sionna_rt_mock",
            "available": _SIONNA_AVAILABLE,
            "scenario": self.scenario,
            "num_cells": self.num_cells,
            "expected_sample_count": self.num_samples,
            "pilot_type_dl": self.pilot_type_dl,
            "pilot_type_ul": self.pilot_type_ul,
            "pilot_type_dl": self.pilot_type_dl,
            "pilot_type_ul": self.pilot_type_ul,
            "channel_est_mode": self.channel_est_mode,
            "channel_model": self.channel_model_name,
            "tdd_pattern": self.tdd_pattern_name,
            "link": self.link,
            "dimensions": {
                "T": self.T,
                "RB": self.RB,
                "BS_ant": self.BS,
                "UE_ant": self.UE,
                "K": self.num_cells,
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
            "srs": {
                "group_hopping": self.srs_group_hopping,
                "sequence_hopping": self.srs_sequence_hopping,
                "periodicity": self.srs_periodicity,
                "b_hop": self.srs_b_hop,
            },
            "ssb": {
                "enable": self.enable_ssb,
                "num_beams": self.num_ssb_beams,
            },
            "tx_power_dbm": self.tx_power_dbm,
            "ue_speed_kmh": self.ue_speed_kmh,
            "ue_distribution": self.ue_distribution,
            "mock": True,
        }


__all__ = ["SionnaRTSource", "SionnaRTMockSource"]
