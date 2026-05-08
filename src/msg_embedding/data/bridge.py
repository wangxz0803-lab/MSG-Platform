"""Upgraded bridge: ``ChannelSample`` -> FeatureExtractor tokens + context stats.

This module is the Phase 1.7 successor to the legacy
``bridge_channel_to_pretrain.py`` script at the repo root. Differences vs. the
legacy bridge:

1. **Typed input**: consumes ``ChannelSample`` instances (pydantic contract),
   not raw ``.npy`` shards.
2. **Interference-aware context**: when ``h_interferers`` or
   ``interference_signal`` is present we compute SIR/SINR (linear),
   interferer spatial covariance top-4 eigendecomposition, MUSIC DoA peaks
   (with Capon fallback), and stash the result in ``norm_stats['interference']``
   rather than expanding the token sequence — this keeps the FeatureExtractor
   API (and its 16-slot layout) untouched.
3. **Self-contained PMI**: 38.214 Type I codebook generation and beam search
   are implemented without external CSV/VAM dependencies. When
   ``use_legacy_pmi=True``, we first try the CSV-backed path and fall back
   to the self-contained codebook (NOT SVD) if unavailable.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.linalg import svd as _scipy_svd

from msg_embedding.core.logging import get_logger

from .contract import ChannelSample

_LOGGER = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants mirrored from the legacy bridge / config.py
# ---------------------------------------------------------------------------

NORM_EPS: float = 1e-8
REF_POWER_OFFSET: float = -100.0
CQI_TABLE: tuple[float, ...] = (
    0.1523,
    0.2344,
    0.3770,
    0.6016,
    0.8770,
    1.1758,
    1.4766,
    1.9141,
    2.4063,
    2.7305,
    3.3223,
    3.9023,
    4.5234,
    5.1152,
    5.5547,
    5.9375,
)

# FeatureExtractor's fixed token layout expects a BS-dim of TX_ANT_NUM_MAX = 64.
# If the sample has fewer BS antennas we pad zeros; if it has more we truncate.
_TX_ANT_NUM_MAX: int = 64
_PDP_TAPS: int = 64
_CELL_RSRP_DIM: int = 16

# Interference bundle defaults (returned when sample has no interferer data).
_INTF_TOP_K: int = 4
_INTF_DOA_GRID: int = 181  # azimuth grid [-90, 90] deg, 1-deg resolution


# ---------------------------------------------------------------------------
# DFT beams — self-contained so we do not depend on ``SsbChanProcFunc`` here.
# ---------------------------------------------------------------------------


def _compute_dft_matrix(num_antennas: int, array_spacing: float = 0.5) -> np.ndarray:
    """Return a ``[N, N]`` DFT beam matrix using physical steering angles.

    Matches ``SSBBeamGenerator.generate_dft_beams`` from the original code:
    angles uniformly spaced in [-π/2, π/2), phase = exp(j·2π·d·sin(θ)·n)/√N.
    """
    n = int(num_antennas)
    beams = np.zeros((n, n), dtype=np.complex128)
    for beam_idx in range(n):
        theta = np.pi * (beam_idx / n - 0.5)
        kd = 2 * np.pi * array_spacing * np.sin(theta)
        for ant_idx in range(n):
            beams[ant_idx, beam_idx] = np.exp(1j * kd * ant_idx) / np.sqrt(n)
    return beams


# ---------------------------------------------------------------------------
# Per-channel scalar & vector helpers (mirror legacy bridge outputs)
# ---------------------------------------------------------------------------


def _compute_pdp(h: np.ndarray, n_taps: int = _PDP_TAPS) -> np.ndarray:
    """Return a length-``n_taps`` PDP in ``[0, 1]`` from ``h[T, RB, BS, UE]``.

    Matches original ``compute_pdp``: IFFT along RB axis on the full 4-D array,
    scale by √RB (Parseval), take power, average over T/BS/UE, normalise.
    """
    rb = h.shape[1]
    ht = np.fft.ifft(h, axis=1) * np.sqrt(rb)
    pdp = np.mean(np.abs(ht) ** 2, axis=(0, 2, 3)).real  # [RB]
    out = np.zeros(n_taps, dtype=np.float32)
    n = min(len(pdp), n_taps)
    out[:n] = pdp[:n]
    mx = float(out.max())
    if mx > NORM_EPS:
        out = out / mx
    return out.astype(np.float32)


def _compute_rsrp_srs(h: np.ndarray) -> np.ndarray:
    """Per-BS-antenna RSRP in dBm from ``h[T, RB, BS, UE]`` (length BS).

    mean(|h|^2) per BS antenna, averaged over T, RB, UE.
    """
    pwr = np.mean(np.abs(h) ** 2, axis=(0, 1, 3))  # [BS]
    rsrp = 10.0 * np.log10(pwr + NORM_EPS) + REF_POWER_OFFSET
    return np.clip(rsrp, -160.0, -60.0).astype(np.float32)


def _compute_rsrp_cb(h_avg: np.ndarray, dft_matrix: np.ndarray) -> np.ndarray:
    """Beam-domain RSRP in dBm.

    ``h_avg``: ``[BS, UE]`` — time-freq averaged channel.
    ``dft_matrix``: ``[BS, BS]`` — physical DFT beam matrix.
    Applies matched filter ``dft^H @ H_avg``, averages power over all UE.
    """
    h_beam = dft_matrix.conj().T @ h_avg  # [num_beams, UE]
    pwr = np.mean(np.abs(h_beam) ** 2, axis=1)  # avg UE → [num_beams]
    rsrp = 10.0 * np.log10(pwr + NORM_EPS) + REF_POWER_OFFSET
    return np.clip(rsrp, -160.0, -60.0).astype(np.float32)


def _srs_svd(h_avg: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """SVD-based SRS tokens: return 4 left singular vectors + top-4 singular values.

    ``h_avg``: ``[BS, UE]`` time-freq averaged channel.
    Returns ``(srs_list, sigma)`` where each srs is length-BS complex64.
    """
    u, s, _ = _scipy_svd(h_avg, full_matrices=False)
    bs = h_avg.shape[0]
    srs: list[np.ndarray] = []
    for i in range(4):
        if i < min(u.shape[1], len(s)):
            srs.append(u[:, i].astype(np.complex64))
        else:
            srs.append(np.zeros(bs, dtype=np.complex64))
    s4 = np.zeros(4, dtype=np.float64)
    s4[: min(4, len(s))] = s[: min(4, len(s))]
    return srs, s4


def _compute_spatial_covariance_iir(
    h: np.ndarray,
    alpha: float = 0.2,
) -> np.ndarray:
    """Spatial covariance ``R_hh`` with IIR filtering across time slots.

    For each time slot *t*, compute ``R_t = E_{rb,ue}[h · h^H]`` by averaging
    the outer product over all RB and UE indices.  Then apply first-order IIR:
    ``R_filt = alpha * R_t + (1-alpha) * R_filt``.

    ``h``: ``[T, RB, BS, UE]`` complex.
    Returns ``R_hh``: ``[BS, BS]`` complex128.
    """
    t_dim, rb, bs, ue = h.shape
    R_filtered: np.ndarray | None = None
    for t in range(t_dim):
        # h[t] is [RB, BS, UE] → reshape to [BS, RB*UE] so each column is one
        # spatial snapshot; R_t = (1/N) * H_col @ H_col^H
        H_col = h[t].transpose(1, 0, 2).reshape(bs, -1)  # [BS, RB*UE]
        R_t = (H_col @ H_col.conj().T) / (rb * ue)
        R_filtered = R_t.copy() if R_filtered is None else alpha * R_t + (1.0 - alpha) * R_filtered
    assert R_filtered is not None
    return R_filtered


def _srs_from_covariance(
    R_hh: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Extract SRS tokens from spatial covariance via eigendecomposition.

    ``R_hh``: ``[BS, BS]`` Hermitian PSD matrix.
    Returns ``(srs_list, eigvals_top4)`` — 4 eigenvectors (complex64) and the
    corresponding eigenvalues in descending order.
    """
    eigvals, eigvecs = np.linalg.eigh(R_hh)  # ascending order
    eigvals = eigvals[::-1].copy()
    eigvecs = eigvecs[:, ::-1].copy()

    bs = R_hh.shape[0]
    srs: list[np.ndarray] = []
    for i in range(4):
        if i < len(eigvals):
            srs.append(eigvecs[:, i].astype(np.complex64))
        else:
            srs.append(np.zeros(bs, dtype=np.complex64))
    s4 = np.zeros(4, dtype=np.float64)
    n = min(4, len(eigvals))
    s4[:n] = np.maximum(eigvals[:n].real, 0.0)
    return srs, s4


def _pmi_svd_fallback(h_avg: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    """SVD-based PMI fallback: use **U left singular vectors** as PMI.

    ``h_avg``: ``[BS, UE]`` — time-freq averaged channel.
    Returns ``(pmi_list, ri, cqi)`` where each PMI vector is [BS]-dimensional,
    matching the original ``bridge_channel_to_pretrain._pmi_svd_fallback``.
    """
    bs = h_avg.shape[0]
    u, s, _ = _scipy_svd(h_avg, full_matrices=False)
    ri = max(1, min(4, int(np.sum(s > s[0] * 0.1)))) if len(s) > 0 else 1
    pmi_list: list[np.ndarray] = []
    for i in range(4):
        if i < u.shape[1]:
            pmi_list.append(u[:, i].astype(np.complex64))
        else:
            pmi_list.append(np.zeros(bs, dtype=np.complex64))
    se = float(np.sum(np.log2(1.0 + s[:ri] ** 2 / 1e-10))) if len(s) > 0 else 0.0
    cqi = int(np.clip(int(np.argmin([abs(se - c) for c in CQI_TABLE])), 0, 15))
    return pmi_list, ri, cqi


def _generate_type_i_codebook(
    n1: int,
    o1: int,
    n2: int,
    o2: int,
    dual_pol: bool = True,
) -> np.ndarray:
    """Generate 38.214 Type I single-panel PMI codebook (self-contained).

    Builds oversampled DFT vectors for horizontal (N1, O1) and vertical
    (N2, O2) dimensions, Kronecker-products them, then extends to dual
    polarisation ``w = [v; phi*v]/sqrt(2)`` with 4 QPSK co-phase values.

    Returns codebook ``[ports, total_beams]`` complex128.
    ``ports`` = ``2*N1*N2`` for dual-pol, ``N1*N2`` for single-pol.
    """
    dft_h = np.zeros((n1, n1 * o1), dtype=np.complex128)
    for k in range(n1 * o1):
        for n in range(n1):
            dft_h[n, k] = np.exp(-1j * 2 * np.pi * n * k / (n1 * o1)) / np.sqrt(n1)

    dft_v = np.zeros((n2, n2 * o2), dtype=np.complex128)
    for k in range(n2 * o2):
        for n in range(n2):
            dft_v[n, k] = np.exp(-1j * 2 * np.pi * n * k / (n2 * o2)) / np.sqrt(n2)

    n_spatial = n1 * n2
    n_beams_spatial = n1 * o1 * n2 * o2
    spatial = np.zeros((n_spatial, n_beams_spatial), dtype=np.complex128)
    for i1 in range(n1 * o1):
        for i2 in range(n2 * o2):
            vec = np.kron(dft_v[:, i2], dft_h[:, i1])
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                vec = vec / norm
            spatial[:, i1 * (n2 * o2) + i2] = vec

    if dual_pol:
        co_phases = [np.exp(1j * np.pi * p / 2) for p in range(4)]
        n_total = n_beams_spatial * 4
        codebook = np.zeros((2 * n_spatial, n_total), dtype=np.complex128)
        for b in range(n_beams_spatial):
            v = spatial[:, b]
            for p, phi in enumerate(co_phases):
                codebook[:, b * 4 + p] = np.concatenate([v, phi * v]) / np.sqrt(2)
    else:
        codebook = spatial

    return codebook


def _pmi_type_i_codebook_search(
    h_avg: np.ndarray,
    antenna_layout: str = "8H4V",
) -> tuple[list[np.ndarray], int, int]:
    """38.214 Type I PMI codebook search on DL channel — self-contained.

    Generates the protocol DFT codebook for the given antenna layout,
    searches best beams by capacity maximisation, returns (pmi_list, ri, cqi).

    No external CSV or VAM dependencies — codebook is pure 38.214 math.
    The VAM port-compression step from the legacy code is skipped; the
    codebook is applied directly to the physical channel.

    ``h_avg``: ``[BS, UE]`` time-freq averaged DL channel.
    ``antenna_layout``: one of ``"8H4V"``, ``"16H2V"``, ``"4T4R"``.
    """
    bs, ue = h_avg.shape

    if antenna_layout == "8H4V":
        n_h, n_v = 8, 4
    elif antenna_layout == "16H2V":
        n_h, n_v = 16, 2
    elif antenna_layout == "4T4R":
        n_h, n_v = 2, 2
    else:
        n_h = int(np.sqrt(bs // 2)) if bs >= 4 else bs
        n_v = max(1, (bs // 2) // n_h) if bs >= 4 else 1

    n_spatial = n_h * n_v
    dual_pol = (bs == 2 * n_spatial)

    if not dual_pol and bs != n_spatial:
        n_spatial = bs
        n_h = bs
        n_v = 1
        dual_pol = False

    o1, o2 = 4, (4 if n_v > 1 else 1)

    if dual_pol:
        n1_cb = n_h
        n2_cb = n_v
    else:
        n1_cb = n_h
        n2_cb = n_v

    codebook = _generate_type_i_codebook(n1_cb, o1, n2_cb, o2, dual_pol=dual_pol)

    cb_ports = codebook.shape[0]
    h_use = h_avg[:cb_ports, :]

    n_beam = codebook.shape[1]
    noise_pwr = 1e-3

    # Vectorised beam gain: mean |w^H h|^2 over UE antennas
    proj = codebook.conj().T @ h_use  # [n_beam, UE]
    beam_gain = np.mean(np.abs(proj) ** 2, axis=1)  # [n_beam]

    ri_max = min(4, ue, n_beam)
    best_ri = 1
    best_cap = -1.0
    best_s: np.ndarray | None = None

    for ri_cand in range(1, ri_max + 1):
        top_beams = np.argsort(beam_gain)[-ri_cand:][::-1]
        W = codebook[:, top_beams]
        H_eff = W.conj().T @ h_use
        _, S, _ = _scipy_svd(H_eff, full_matrices=False)
        cap = float(np.sum(np.log2(1.0 + S ** 2 / noise_pwr)))

        if cap > best_cap:
            best_cap = cap
            best_ri = ri_cand
            best_s = S[:ri_cand].copy()

    top_beams_final = np.argsort(beam_gain)[-best_ri:][::-1]
    pmi_weights = codebook[:, top_beams_final]

    pmi_list: list[np.ndarray] = []
    for i in range(4):
        if i < best_ri and i < pmi_weights.shape[1]:
            pmi_list.append(pmi_weights[:, i].astype(np.complex64))
        else:
            pmi_list.append(np.zeros(cb_ports, dtype=np.complex64))

    se = float(np.sum(np.log2(1.0 + best_s ** 2 / noise_pwr))) if best_s is not None else 0.0
    cqi = int(np.clip(np.argmin([abs(se - c) for c in CQI_TABLE]), 0, 15))

    return pmi_list, best_ri, cqi


def _legacy_pmi_tokens(h_sim: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    """Try the legacy CSV-backed PMI generator; fall back to self-contained Type I codebook."""
    try:
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        import gol  # type: ignore  # noqa: WPS433

        gol._init()
        from CsiChanProcFunc import PmiCqiRiGenerator  # type: ignore  # noqa: WPS433

        ri, cqi, pmi_weights = PmiCqiRiGenerator(h_sim, "8H4V")
        bs = h_sim.shape[0]
        pmi_list: list[np.ndarray] = []
        for i in range(4):
            if i < ri and i < pmi_weights.shape[1]:
                pmi_list.append(pmi_weights[:, i].astype(np.complex64))
            else:
                pmi_list.append(np.zeros(bs, dtype=np.complex64))
        return pmi_list, int(ri), int(cqi)
    except Exception as exc:
        _LOGGER.info("legacy CSV PMI unavailable (%s); using self-contained Type I codebook", exc)
        h_avg = h_sim[:, :, :, 0].mean(axis=2)  # [BS, UE]
        return _pmi_type_i_codebook_search(h_avg, antenna_layout="8H4V")


# ---------------------------------------------------------------------------
# Interference-aware features
# ---------------------------------------------------------------------------


def _interference_defaults() -> dict[str, Any]:
    """Default interference bundle when the sample carries no interferer data."""
    return {
        "sir_linear": float("nan"),
        "sinr_linear": float("nan"),
        "intf_cov": None,
        "eigvals_top4": np.full(_INTF_TOP_K, np.nan, dtype=np.float32),
        "eigvecs_top4": None,
        "doa_peaks_deg": np.full(_INTF_TOP_K, np.nan, dtype=np.float32),
        "num_strong": 0,
        "spectrum_kind": "none",
    }


def _aggregate_interferer_covariance(h_interferers: np.ndarray) -> np.ndarray:
    """Aggregate ``[K-1, T, RB, BS, UE]`` -> ``[BS, BS]`` interference covariance.

    For each (interferer, T, RB, UE) we form an outer product ``h h^H`` and
    average across all indices.
    """
    k_m1, t, rb, bs, ue = h_interferers.shape
    cov = np.zeros((bs, bs), dtype=np.complex128)
    # Reshape into columns and compute cov via matmul — O(BS^2 * N) memory-efficient.
    h = h_interferers.transpose(3, 0, 1, 2, 4).reshape(bs, k_m1 * t * rb * ue)
    n = h.shape[1]
    if n == 0:
        return cov
    cov = (h @ h.conj().T) / float(n)
    return cov


def _music_spectrum(
    cov: np.ndarray,
    num_sources: int,
    angle_grid_deg: np.ndarray,
    array_spacing: float = 0.5,
) -> np.ndarray:
    """MUSIC pseudo-spectrum for a ULA with spacing ``array_spacing * lambda``."""
    bs = cov.shape[0]
    # Hermitian eigendecomposition; eigvals ascending.
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Noise subspace: all eigenvectors except the top ``num_sources`` (descending).
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    noise_dim = max(1, bs - num_sources)
    e_n = eigvecs[:, -noise_dim:]
    spec = np.zeros(angle_grid_deg.shape[0], dtype=np.float64)
    for i, theta_deg in enumerate(angle_grid_deg):
        theta = np.deg2rad(theta_deg)
        a = np.exp(1j * 2.0 * np.pi * array_spacing * np.sin(theta) * np.arange(bs)).astype(
            np.complex128
        )
        denom = float(np.abs(a.conj() @ e_n @ e_n.conj().T @ a))
        spec[i] = 1.0 / (denom + NORM_EPS)
    return spec


def _capon_spectrum(
    cov: np.ndarray,
    angle_grid_deg: np.ndarray,
    array_spacing: float = 0.5,
) -> np.ndarray:
    """Capon (MVDR) pseudo-spectrum — used when MUSIC's eigendecomp is degenerate."""
    bs = cov.shape[0]
    # Regularise for invertibility.
    cov_reg = cov + NORM_EPS * np.eye(bs, dtype=cov.dtype)
    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_reg)
    spec = np.zeros(angle_grid_deg.shape[0], dtype=np.float64)
    for i, theta_deg in enumerate(angle_grid_deg):
        theta = np.deg2rad(theta_deg)
        a = np.exp(1j * 2.0 * np.pi * array_spacing * np.sin(theta) * np.arange(bs)).astype(
            np.complex128
        )
        denom = float(np.abs(a.conj() @ cov_inv @ a))
        spec[i] = 1.0 / (denom + NORM_EPS)
    return spec


def _pick_peaks(spec: np.ndarray, angle_grid_deg: np.ndarray, top_k: int) -> np.ndarray:
    """Return the ``top_k`` strongest local maxima angles (deg); pad with NaN."""
    peaks: list[int] = []
    n = spec.shape[0]
    for i in range(n):
        left = spec[i - 1] if i > 0 else -np.inf
        right = spec[i + 1] if i < n - 1 else -np.inf
        if spec[i] >= left and spec[i] >= right:
            peaks.append(i)
    if not peaks:
        out = np.full(top_k, np.nan, dtype=np.float32)
        return out
    peaks.sort(key=lambda j: -spec[j])
    chosen = peaks[:top_k]
    out = np.full(top_k, np.nan, dtype=np.float32)
    for idx, j in enumerate(chosen):
        out[idx] = float(angle_grid_deg[j])
    return out


def compute_interference_features(sample: ChannelSample) -> dict[str, Any]:
    """Compute the interference-aware context bundle for a sample.

    Returns a dict with keys:

    * ``sir_linear`` / ``sinr_linear`` — scalar linear-domain ratios (NaN if absent).
    * ``intf_cov`` — ``[BS, BS]`` complex64 interference covariance, or None.
    * ``eigvals_top4`` — top-4 covariance eigenvalues (float32).
    * ``eigvecs_top4`` — ``[BS, 4]`` complex64 top-4 eigenvectors, or None.
    * ``doa_peaks_deg`` — top-4 DoA peaks in degrees (float32, NaN-padded).
    * ``num_strong`` — number of eigenvalues exceeding 10% of the top one.
    * ``spectrum_kind`` — ``"music"``, ``"capon"``, or ``"none"``.
    """
    out = _interference_defaults()

    # Scalar SIR/SINR — always populate from the contract if present.
    if sample.sir_dB is not None:
        out["sir_linear"] = float(10.0 ** (sample.sir_dB / 10.0))
    out["sinr_linear"] = float(10.0 ** (sample.sinr_dB / 10.0))

    if sample.h_interferers is None:
        return out

    cov = _aggregate_interferer_covariance(sample.h_interferers)
    out["intf_cov"] = cov.astype(np.complex64)

    # Hermitian eigendecomposition for top-4.
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return out
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    bs = cov.shape[0]
    k = min(_INTF_TOP_K, bs)
    top_vals = np.zeros(_INTF_TOP_K, dtype=np.float32)
    top_vals[:k] = eigvals[:k].astype(np.float32)
    out["eigvals_top4"] = top_vals
    vecs = np.zeros((bs, _INTF_TOP_K), dtype=np.complex64)
    vecs[:, :k] = eigvecs[:, :k].astype(np.complex64)
    out["eigvecs_top4"] = vecs

    top = float(eigvals[0]) if eigvals.size > 0 else 0.0
    if top > NORM_EPS:
        out["num_strong"] = int(np.sum(eigvals > 0.1 * top))
    else:
        out["num_strong"] = 0

    # DoA peaks via MUSIC; fall back to Capon if eigendecomp was fragile.
    angle_grid = np.linspace(-90.0, 90.0, _INTF_DOA_GRID)
    num_sources = max(1, min(_INTF_TOP_K, out["num_strong"]))
    try:
        spec = _music_spectrum(cov, num_sources, angle_grid)
        out["spectrum_kind"] = "music"
    except Exception as exc:  # pragma: no cover - numerically fragile paths
        _LOGGER.warning("MUSIC failed (%s); falling back to Capon", exc)
        spec = _capon_spectrum(cov, angle_grid)
        out["spectrum_kind"] = "capon"
    if not np.all(np.isfinite(spec)):
        spec = _capon_spectrum(cov, angle_grid)
        out["spectrum_kind"] = "capon"
    out["doa_peaks_deg"] = _pick_peaks(spec, angle_grid, _INTF_TOP_K)

    return out


# ---------------------------------------------------------------------------
# Feat dict assembly
# ---------------------------------------------------------------------------


def _pad_bs(vec: np.ndarray, target: int) -> np.ndarray:
    """Zero-pad or truncate a 1D complex vector to length ``target``."""
    vec = np.asarray(vec)
    if vec.shape[0] == target:
        return vec.astype(np.complex64, copy=False)
    if vec.shape[0] > target:
        return vec[:target].astype(np.complex64, copy=False)
    out = np.zeros(target, dtype=np.complex64)
    out[: vec.shape[0]] = vec.astype(np.complex64)
    return out


def _pad_real(vec: np.ndarray, target: int, fill: float = -160.0) -> np.ndarray:
    """Zero/fill-pad or truncate a 1D real vector to length ``target``."""
    vec = np.asarray(vec)
    if vec.shape[0] == target:
        return vec.astype(np.float32, copy=False)
    if vec.shape[0] > target:
        return vec[:target].astype(np.float32, copy=False)
    out = np.full(target, fill, dtype=np.float32)
    out[: vec.shape[0]] = vec.astype(np.float32)
    return out


def _build_feat_dict(
    sample: ChannelSample,
    use_legacy_pmi: bool,
    *,
    h_ul_override: np.ndarray | None = None,
    h_dl_override: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the 24-field feat dict (with batch dim 1) + context metadata.

    When ``sample.link_pairing == "paired"``, UL-derived tokens (PDP, SRS,
    DFT, RSRP) are computed from ``h_ul_est`` and DL-derived tokens (PMI,
    CQI) from ``h_dl_est``.  Use ``h_ul_override`` / ``h_dl_override`` to
    feed ground-truth channels for GT feature dicts.

    Produces exactly the 16 token fields + 8 gate fields expected by
    ``FeatureExtractor.forward()``:

    Tokens (16):
        ``pdp_crop``        [1, 64]  float32    — from UL channel
        ``srs1``..``srs4``  [1, 64]  complex64  — from UL channel
        ``pmi1``..``pmi4``  [1, 64]  complex64  — from DL channel
        ``dft1``..``dft4``  [1, 64]  complex64  — from UL channel
        ``rsrp_srs``        [1, 64]  float32    — from UL channel
        ``rsrp_cb``          [1, 64]  float32   — from UL channel
        ``cell_rsrp``       [1, 16]  float32    — from SSB

    Gates (8):
        ``srs_w1``..``srs_w4``  [1]  float32 — normalised singular values
        ``srs_sinr``            [1]  float32 — SRS SINR in dB
        ``srs_cb_sinr``         [1]  float32 — codebook SINR in dB
        ``cqi``                 [1]  int64   — channel quality index 0-15
    """
    # Select channels based on pairing mode
    if h_ul_override is not None:
        h_ul = h_ul_override
    elif sample.link_pairing == "paired" and sample.h_ul_est is not None:
        h_ul = sample.h_ul_est
    else:
        h_ul = sample.h_serving_est

    if h_dl_override is not None:
        h_dl = h_dl_override
    elif sample.link_pairing == "paired" and sample.h_dl_est is not None:
        h_dl = sample.h_dl_est
    else:
        h_dl = sample.h_serving_est

    # TDD reciprocity fix: h_ul may be stored as [T, RB, UE, BS] (transposed
    # from DL). Detect via h_serving_est which is always [T, RB, BS, UE], and
    # transpose h_ul/h_dl back to [T, RB, BS, UE] if needed.
    ref_bs = sample.h_serving_est.shape[2]
    if h_ul.shape[2] != ref_bs and h_ul.shape[3] == ref_bs:
        h_ul = h_ul.transpose(0, 1, 3, 2)
    if h_dl.shape[2] != ref_bs and h_dl.shape[3] == ref_bs:
        h_dl = h_dl.transpose(0, 1, 3, 2)

    h = h_ul  # primary channel for dimensional reference
    t, rb, bs, ue = h.shape

    feat: dict[str, Any] = {}

    # =====================================================================
    # Time-freq averages for UL and DL channels
    # =====================================================================
    h_ul_avg = np.mean(h_ul, axis=(0, 1))  # [BS, UE] — UL spatial
    h_dl_avg = np.mean(h_dl, axis=(0, 1))  # [BS, UE] — DL spatial

    # =====================================================================
    # Spatial covariance R_hh = E[h·h^H] with IIR temporal smoothing.
    # Computed from UL channel (BS receiver perspective).
    # =====================================================================
    R_hh = _compute_spatial_covariance_iir(h_ul, alpha=0.2)
    srs_cov_list, srs_cov_eigvals = _srs_from_covariance(R_hh)

    # DFT beam matrix — shared by DFT tokens and RSRP_CB.
    n_dft = min(bs, _TX_ANT_NUM_MAX)
    dft_matrix = _compute_dft_matrix(n_dft)

    # =====================================================================
    # Token 0: PDP  [1, 64] float32 — from UL channel
    # =====================================================================
    pdp = _compute_pdp(h_ul)
    feat["pdp_crop"] = torch.from_numpy(pdp).unsqueeze(0)

    # =====================================================================
    # Tokens 1-4: SRS  [1, 64] complex64 each
    # Eigenvectors of R_hh (spatial covariance with IIR smoothing).
    # =====================================================================
    for i in range(4):
        feat[f"srs{i + 1}"] = torch.from_numpy(_pad_bs(srs_cov_list[i], _TX_ANT_NUM_MAX)).unsqueeze(
            0
        )

    # Gate weights: normalised eigenvalues (sum to 1)
    ev_sum = float(srs_cov_eigvals.sum())
    ev_norm = srs_cov_eigvals / ev_sum if ev_sum > NORM_EPS else np.full(4, 0.25)
    for i in range(4):
        feat[f"srs_w{i + 1}"] = torch.tensor(
            [float(ev_norm[i])],
            dtype=torch.float32,
        )

    # =====================================================================
    # Tokens 5-8: PMI  [1, 64] complex64 each
    # =====================================================================
    if use_legacy_pmi:
        # Reshape DL channel to legacy layout [BS, UE, T*RB, 1] for CsiChanProcFunc.
        t_dl, rb_dl, bs_dl, ue_dl = h_dl.shape
        h_sim = h_dl.transpose(2, 3, 0, 1).reshape(bs_dl, ue_dl, t_dl * rb_dl, 1)
        pmi_list, _pmi_ri, pmi_cqi = _legacy_pmi_tokens(h_sim)
    else:
        # Self-contained 38.214 Type I codebook search on DL channel.
        pmi_list, _pmi_ri, pmi_cqi = _pmi_type_i_codebook_search(h_dl_avg)
    for i, vec in enumerate(pmi_list, start=1):
        feat[f"pmi{i}"] = torch.from_numpy(_pad_bs(vec, _TX_ANT_NUM_MAX)).unsqueeze(0)

    # =====================================================================
    # Tokens 9-12: DFT beams  [1, 64] complex64 each — from UL channel
    # Matched filter dft^H @ H_avg, sum energy over all UE, top-4 beams.
    # Returns dft_matrix columns (steering vectors).
    # =====================================================================
    beam_response = dft_matrix.conj().T @ h_ul_avg[:n_dft, :]  # [N, UE]
    beam_energy = np.sum(np.abs(beam_response) ** 2, axis=1)  # [N]

    n_beams = min(4, n_dft)
    top4_idx = np.argsort(beam_energy)[-n_beams:][::-1]
    for i in range(4):
        if i < n_beams:
            dft_vec = dft_matrix[:, top4_idx[i]].astype(np.complex64)  # column
        else:
            dft_vec = np.zeros(n_dft, dtype=np.complex64)
        dft_padded = np.zeros(_TX_ANT_NUM_MAX, dtype=np.complex64)
        dft_padded[:n_dft] = dft_vec[:n_dft]
        feat[f"dft{i + 1}"] = torch.from_numpy(dft_padded).unsqueeze(0)

    # =====================================================================
    # Token 13: rsrp_srs  [1, 64] float32 — per-antenna RSRP from UL
    # =====================================================================
    rsrp_srs_raw = _compute_rsrp_srs(h_ul)  # [BS] float32
    rsrp_srs_pad = np.full(_TX_ANT_NUM_MAX, -160.0, dtype=np.float32)
    n_rs = min(bs, _TX_ANT_NUM_MAX)
    rsrp_srs_pad[:n_rs] = rsrp_srs_raw[:n_rs]
    feat["rsrp_srs"] = torch.from_numpy(rsrp_srs_pad).unsqueeze(0)

    # =====================================================================
    # Token 14: rsrp_cb  [1, 64] float32 — beam-domain RSRP in dBm
    # dft^H @ H_avg, avg power over all UE, same DFT matrix as tokens 9-12.
    # =====================================================================
    rsrp_cb_raw = _compute_rsrp_cb(h_ul_avg[:n_dft, :], dft_matrix)  # [N]
    rsrp_cb_pad = np.full(_TX_ANT_NUM_MAX, -160.0, dtype=np.float32)
    n_cb = min(n_dft, _TX_ANT_NUM_MAX)
    rsrp_cb_pad[:n_cb] = rsrp_cb_raw[:n_cb]
    feat["rsrp_cb"] = torch.from_numpy(rsrp_cb_pad).unsqueeze(0)

    # =====================================================================
    # Token 15: cell_rsrp  [1, 16] float32
    # Serving cell + neighbor RSRP in dBm, padded with -160.
    # =====================================================================
    cell_rsrp = np.full(_CELL_RSRP_DIM, -160.0, dtype=np.float32)
    if sample.ssb_rsrp_dBm is not None and len(sample.ssb_rsrp_dBm) > 0:
        n_cells = min(len(sample.ssb_rsrp_dBm), _CELL_RSRP_DIM)
        for i in range(n_cells):
            cell_rsrp[i] = float(np.clip(sample.ssb_rsrp_dBm[i], -160.0, -60.0))
    feat["cell_rsrp"] = torch.from_numpy(cell_rsrp).unsqueeze(0)

    # =====================================================================
    # Gate: srs_sinr  [1] float32 — SRS SINR in dB
    # =====================================================================
    sinr_db = float(np.clip(sample.sinr_dB, -20.0, 20.0))
    feat["srs_sinr"] = torch.tensor([sinr_db], dtype=torch.float32)

    # =====================================================================
    # Gate: srs_cb_sinr  [1] float32 — codebook SINR in dB
    # =====================================================================
    feat["srs_cb_sinr"] = torch.tensor([sinr_db], dtype=torch.float32)

    # =====================================================================
    # Gate: cqi  [1] int64 — Channel Quality Index 0-15
    # CQI is derived from the PMI computation above (spectral efficiency
    # → CQI table lookup), matching the original bridge behaviour.
    # =====================================================================
    feat["cqi"] = torch.tensor([int(pmi_cqi)], dtype=torch.int64)

    # =====================================================================
    # Context / metadata
    # =====================================================================
    _dl_rank = sample.dl_rank if sample.dl_rank is not None else (sample.meta.get("dl_rank") if sample.meta else None)
    context: dict[str, Any] = {
        "bs_native": int(bs),
        "ue_native": int(ue),
        "t": int(t),
        "rb": int(rb),
        "used_legacy_pmi": use_legacy_pmi,
        "sample_id": sample.sample_id,
        "link_pairing": sample.link_pairing,
        "dl_rank": int(_dl_rank) if _dl_rank is not None else None,
    }

    return feat, context


def _cqi_from_sinr(sinr_db: float) -> int:
    """Map SINR dB -> spectral efficiency -> CQI index (0-15).

    Uses a Shannon-capacity-based mapping: SE = log2(1 + 10^(sinr_dB/10)),
    then linearly map SE to CQI range [0, 15] with max SE ~7.4 bps/Hz.
    """
    sinr_lin = 10.0 ** (float(sinr_db) / 10.0)
    se = float(np.log2(1.0 + sinr_lin))
    cqi = int(np.clip(np.round(se * 15.0 / 7.4), 0, 15))
    return cqi


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sample_to_features(
    sample: ChannelSample,
    feature_extractor: Any,
    use_legacy_pmi: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Convert a :class:`ChannelSample` into FeatureExtractor tokens + norm_stats.

    Parameters
    ----------
    sample
        Validated channel-sample record.
    feature_extractor
        An instance of :class:`tools.FeatureExtractor` (not imported here to
        avoid a hard dependency on the legacy root-level ``tools.py``).
    use_legacy_pmi
        If ``True`` (default), try the CSV-backed legacy PMI generator first,
        falling back to the self-contained 38.214 Type I codebook search if
        unavailable. If ``False``, use the self-contained codebook directly.

    Returns
    -------
    tokens : ``torch.Tensor``
        Shape ``[1, SEQ_LEN, TOKEN_DIM]`` — single-sample token sequence.
    norm_stats : ``dict``
        The ``FeatureExtractor`` normalisation bundle, augmented with:

        * ``'interference'`` — :func:`compute_interference_features` result.
        * ``'bridge_context'`` — source-of-truth metadata (bs_native, direct
          bypass flags, etc.).
        * ``'gt_tokens'`` — ground-truth tokens from ideal channels (only
          when ``link_pairing == "paired"``; ``None`` otherwise).
    """
    feat, context = _build_feat_dict(sample, use_legacy_pmi=use_legacy_pmi)
    intf = compute_interference_features(sample)

    with torch.no_grad():
        tokens, norm_stats = feature_extractor(feat)

    norm_stats["interference"] = intf
    norm_stats["bridge_context"] = context

    # GT tokens from ideal channels when paired
    gt_tokens: torch.Tensor | None = None
    if (
        sample.link_pairing == "paired"
        and sample.h_ul_true is not None
        and sample.h_dl_true is not None
    ):
        feat_gt, _ = _build_feat_dict(
            sample,
            use_legacy_pmi=use_legacy_pmi,
            h_ul_override=sample.h_ul_true,
            h_dl_override=sample.h_dl_true,
        )
        with torch.no_grad():
            gt_tokens, _ = feature_extractor(feat_gt)
    norm_stats["gt_tokens"] = gt_tokens

    return tokens, norm_stats


def batch_samples_to_features(
    samples: Iterable[ChannelSample],
    feature_extractor: Any,
    batch_size: int = 32,
    use_legacy_pmi: bool = True,
) -> Iterator[tuple[torch.Tensor, dict[str, Any]]]:
    """Yield ``(tokens[B, SEQ_LEN, TOKEN_DIM], norm_stats)`` chunks of size <= ``batch_size``.

    Each ``norm_stats['interference']`` / ``['bridge_context']`` becomes a list
    of per-sample dicts (length B) so downstream code can route each sample
    individually. All other entries are the stacked FeatureExtractor outputs.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    buffer: list[ChannelSample] = []

    def _flush(buf: list[ChannelSample]) -> tuple[torch.Tensor, dict[str, Any]]:
        per_feat: list[dict[str, Any]] = []
        per_context: list[dict[str, Any]] = []
        per_intf: list[dict[str, Any]] = []
        for s in buf:
            fd, ctx = _build_feat_dict(s, use_legacy_pmi=use_legacy_pmi)
            per_feat.append(fd)
            per_context.append(ctx)
            per_intf.append(compute_interference_features(s))

        # Stack per-key across the batch (each tensor in per_feat has batch-dim 1).
        stacked: dict[str, torch.Tensor] = {}
        for key in per_feat[0]:
            stacked[key] = torch.cat([d[key] for d in per_feat], dim=0)
        with torch.no_grad():
            tokens, norm_stats = feature_extractor(stacked)
        norm_stats["interference"] = per_intf
        norm_stats["bridge_context"] = per_context
        return tokens, norm_stats

    for sample in samples:
        buffer.append(sample)
        if len(buffer) >= batch_size:
            yield _flush(buffer)
            buffer = []
    if buffer:
        yield _flush(buffer)


__all__ = [
    "sample_to_features",
    "batch_samples_to_features",
    "compute_interference_features",
]
