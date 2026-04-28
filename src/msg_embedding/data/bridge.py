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
3. **Legacy fallback**: when ``use_legacy_pmi=True``, we reuse the legacy
   ``bridge_channel_to_pretrain`` PMI/SRS computation so feature coverage
   matches the in-tree training runs.
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


def _compute_dft_matrix(num_antennas: int) -> np.ndarray:
    """Return a ``[N, N]`` complex128 DFT beam matrix (normalised)."""
    n = int(num_antennas)
    return np.fft.fft(np.eye(n)) / np.sqrt(n)


# ---------------------------------------------------------------------------
# Per-channel scalar & vector helpers (mirror legacy bridge outputs)
# ---------------------------------------------------------------------------


def _compute_pdp(h: np.ndarray, n_taps: int = _PDP_TAPS) -> np.ndarray:
    """Return a length-``n_taps`` PDP in ``[0, 1]`` from ``h[T, RB, BS, UE]``.

    Computation: average over time, take first UE antenna → [RB, BS],
    IFFT along freq axis → power → average over BS → normalise to [0,1].
    """
    # Average over time: [T, RB, BS, UE] -> [RB, BS, UE]
    h_avg_freq = np.mean(h, axis=0)
    # Take first UE antenna: [RB, BS]
    h_flat = h_avg_freq[:, :, 0]
    # IFFT along freq axis (axis=0, the RB axis) -> power delay profile
    pdp = np.abs(np.fft.ifft(h_flat, axis=0)) ** 2  # [RB, BS]
    # Average over BS antennas: [RB]
    pdp = np.mean(pdp, axis=1)
    # Normalise to [0, 1]
    mx = float(pdp.max())
    if mx > NORM_EPS:
        pdp = pdp / mx
    # Crop / zero-pad to n_taps
    out = np.zeros(n_taps, dtype=np.float32)
    cut = min(n_taps, pdp.shape[0])
    out[:cut] = pdp[:cut].real.astype(np.float32)
    return out


def _compute_rsrp_srs(h: np.ndarray) -> np.ndarray:
    """Per-BS-antenna RSRP in dBm from ``h[T, RB, BS, UE]`` (length BS).

    mean(|h|^2) per BS antenna, averaged over T, RB, UE.
    """
    pwr = np.mean(np.abs(h) ** 2, axis=(0, 1, 3))  # [BS]
    rsrp = 10.0 * np.log10(pwr + 1e-30) + REF_POWER_OFFSET
    return rsrp.astype(np.float32)


def _compute_rsrp_cb(beam_power: np.ndarray) -> np.ndarray:
    """Beam-domain RSRP in dBm from pre-computed beam power array.

    ``beam_power`` is ``[N]`` — linear beam-domain power from DFT token step.
    """
    rsrp = 10.0 * np.log10(beam_power + 1e-30) + REF_POWER_OFFSET
    return rsrp.astype(np.float32)


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


def _pmi_svd_fallback(h_avg: np.ndarray, vh: np.ndarray) -> list[np.ndarray]:
    """SVD-based PMI fallback: use right singular vectors ``Vh`` rows as PMI.

    ``h_avg``: ``[BS, UE]`` — only used for dimension.
    ``vh``:    ``[min(BS,UE), UE]`` — right singular vectors from SVD of h_avg.
    Returns 4 PMI vectors, each zero-padded/truncated to ``_TX_ANT_NUM_MAX``.
    """
    pmi_list: list[np.ndarray] = []
    for i in range(4):
        if i < vh.shape[0]:
            pmi_vec = vh[i, :].astype(np.complex64)  # [UE] or [min(BS,UE)]
        else:
            pmi_vec = np.zeros(vh.shape[1] if vh.ndim > 1 else 1, dtype=np.complex64)
        pmi_list.append(pmi_vec)
    return pmi_list


def _pmi_dft_codebook_search(
    h_avg: np.ndarray,
    oversampling: int = 4,
) -> list[np.ndarray]:
    """DFT codebook PMI: Type-I single-panel search (3GPP 38.214 §5.2.2.2.1).

    Builds an oversampled DFT codebook of size ``N_tx * oversampling``, then
    selects the top-4 beams by projection power onto ``h_avg[:, 0]``.

    Returns 4 PMI vectors (complex64), each of length ``N_tx``.
    """
    n_tx = h_avg.shape[0]
    n_beams = n_tx * oversampling
    codebook = np.zeros((n_beams, n_tx), dtype=np.complex128)
    for b in range(n_beams):
        codebook[b, :] = np.exp(1j * 2 * np.pi * b * np.arange(n_tx) / n_beams) / np.sqrt(n_tx)

    h_col = h_avg[:, 0]  # [N_tx] — first UE antenna
    proj_power = np.abs(codebook @ h_col) ** 2  # [n_beams]
    top4_idx = np.argsort(proj_power)[::-1][:4]

    pmi_list: list[np.ndarray] = []
    for i in range(4):
        if i < len(top4_idx):
            pmi_list.append(codebook[top4_idx[i], :].astype(np.complex64))
        else:
            pmi_list.append(np.zeros(n_tx, dtype=np.complex64))
    return pmi_list


def _legacy_pmi_tokens(h_sim: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    """Try the legacy CSV-backed PMI generator; silently fall back to SVD on error."""
    try:
        # The legacy code lives at the repo root — we only need it when the
        # caller opts into ``use_legacy_pmi`` and a direct PMI is absent.
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
    except Exception as exc:  # pragma: no cover - fallback path is data-dependent
        _LOGGER.warning("legacy PMI unavailable (%s); using SVD fallback", exc)
        # Fall back to SVD-based PMI using h_sim layout
        bs = h_sim.shape[0]
        h_avg = h_sim[:, :, :, 0].mean(axis=2)  # [BS, UE]
        _, s, vh = _scipy_svd(h_avg, full_matrices=False)
        pmi_list = _pmi_svd_fallback(h_avg, vh)
        if len(s) == 0:
            ri = 1
            cqi_val = 0
        else:
            ri = max(1, min(4, int(np.sum(s > s[0] * 0.1))))
            se = float(np.sum(np.log2(1.0 + s[:ri] ** 2 / 1e-10)))
            cqi_val = int(np.clip(int(np.argmin([abs(se - c) for c in CQI_TABLE])), 0, 15))
        return pmi_list, ri, cqi_val


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

    # SVD of DL average — needed for PMI fallback (Vh right singular vectors).
    _, sigma, vh_svd = _scipy_svd(h_dl_avg, full_matrices=False)

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
        pmi_list, _, _ = _legacy_pmi_tokens(h_sim)
    else:
        # DFT codebook search (Type-I single-panel, 3GPP 38.214) on DL channel
        pmi_list = _pmi_dft_codebook_search(h_dl_avg)
    for i, vec in enumerate(pmi_list, start=1):
        feat[f"pmi{i}"] = torch.from_numpy(_pad_bs(vec, _TX_ANT_NUM_MAX)).unsqueeze(0)

    # =====================================================================
    # Tokens 9-12: DFT beams  [1, 64] complex64 each — from UL channel
    # Top-4 energy DFT beams of UL h_avg.
    # =====================================================================
    n_dft = min(bs, _TX_ANT_NUM_MAX)
    dft_matrix = _compute_dft_matrix(n_dft)  # [N, N]
    beam_response = dft_matrix @ h_ul_avg[:n_dft, 0]  # [N] beam-domain channel
    beam_power = np.abs(beam_response) ** 2  # [N] linear power per beam

    n_beams = min(4, n_dft)
    top4_idx = np.argsort(beam_power)[::-1][:n_beams]
    for i in range(4):
        if i < n_beams:
            dft_vec = dft_matrix[top4_idx[i], :].astype(np.complex64)  # [N] beam row
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
    # =====================================================================
    rsrp_cb_raw = _compute_rsrp_cb(beam_power)  # [N] float32
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
        rsrp_sorted = sorted(sample.ssb_rsrp_dBm, reverse=True)
        n_cells = min(len(rsrp_sorted), _CELL_RSRP_DIM)
        for i in range(n_cells):
            cell_rsrp[i] = float(rsrp_sorted[i])
    else:
        cell_rsrp[0] = -110.0  # default serving cell
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
    # When precoding weights are available, compute effective per-layer
    # SINR from the precoded channel for a more accurate CQI.
    # =====================================================================
    _w_dl = sample.w_dl if sample.w_dl is not None else (sample.meta.get("w_dl") if sample.meta else None)
    if _w_dl is not None and isinstance(_w_dl, np.ndarray) and _w_dl.ndim == 3:
        from msg_embedding.phy_sim.precoding import apply_precoding
        try:
            h_eff = apply_precoding(h_dl, _w_dl)  # [T, RB, rank, UE_ant]
            eff_power = float(np.mean(np.abs(h_eff) ** 2))
            noise_lin = 10.0 ** (-float(sample.snr_dB) / 10.0)
            eff_sinr_lin = eff_power / max(noise_lin, 1e-30)
            if sample.sir_dB is not None:
                intf_lin = 10.0 ** (-float(sample.sir_dB) / 10.0)
                eff_sinr_lin = eff_power / max(noise_lin + intf_lin, 1e-30)
            eff_sinr_db = float(10.0 * np.log10(max(eff_sinr_lin, 1e-15)))
            eff_sinr_db = float(np.clip(eff_sinr_db, -50.0, 50.0))
            cqi = _cqi_from_sinr(eff_sinr_db)
        except Exception:
            cqi = _cqi_from_sinr(sample.sinr_dB)
    else:
        cqi = _cqi_from_sinr(sample.sinr_dB)
    feat["cqi"] = torch.tensor([int(cqi)], dtype=torch.int64)

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
        If ``True`` (default), fall back to the legacy PMI generator (with
        its SVD fallback). Setting this to ``False`` skips the CSV-dependent
        legacy path and always uses SVD.

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
