"""DL precoding weight computation from UL channel estimates (SRS-based beamforming).

TDD reciprocity: H_DL = conj(H_UL^T) (antenna-dim transpose).
SVD(H_DL) = U S Vh → W_DL = U[:, :rank] maximizes effective channel gain.

All channels follow the contract convention: [T, RB, BS_ant, UE_ant].
For UL, h[bs, ue] = channel from UE_ue (tx) to BS_bs (rx).
For DL, h[bs, ue] = channel from BS_bs (tx) to UE_ue (rx).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import svd as _scipy_svd


@dataclass(frozen=True)
class PrecodingResult:
    """Result of DL precoding weight computation."""

    w_dl: np.ndarray
    """[RB, BS_ant, rank] complex64 — per-RB precoding weight matrix."""

    rank: int
    """Transmission rank, 1 ≤ rank ≤ min(BS_ant, UE_ant)."""

    singular_values: np.ndarray
    """[RB, min(BS_ant, UE_ant)] float32 — per-RB singular values (descending)."""


def _ul_to_dl(h_ul: np.ndarray) -> np.ndarray:
    """Convert UL channel (contract convention) to DL via TDD reciprocity.

    Both stored as [..., BS_ant, UE_ant] in contract convention.
    h_ul_contract[bs, ue] = gain from UE_ue → BS_bs (physical UL).
    h_dl_contract[bs, ue] = gain from BS_bs → UE_ue (physical DL).
    Reciprocity: h_dl_contract = conj(h_ul_contract) (no transpose in contract).
    """
    return np.conj(h_ul)


def compute_dl_precoding(
    h_ul_est: np.ndarray,
    *,
    max_rank: int = 4,
    rank_threshold: float = 0.1,
    average_time: bool = True,
) -> PrecodingResult:
    """Compute per-RB DL precoding weights from UL estimated channel via SVD.

    Parameters
    ----------
    h_ul_est:
        ``[T, RB, BS_ant, UE_ant]`` complex — UL channel in contract convention.
    max_rank:
        Maximum allowed transmission rank.
    rank_threshold:
        Singular values below ``rank_threshold * sigma_max`` are dropped.
    average_time:
        If True, average across time before SVD. If False, use last slot.
    """
    h = np.asarray(h_ul_est)
    if h.ndim != 4:
        raise ValueError(f"h_ul_est must be 4-D [T, RB, BS_ant, UE_ant], got shape {h.shape}")

    T, RB, BS_ant, UE_ant = h.shape

    h_avg = h.mean(axis=0) if average_time else h[-1]  # [RB, BS_ant, UE_ant]
    h_dl = _ul_to_dl(h_avg)  # [RB, BS_ant, UE_ant] in DL convention

    min_dim = min(BS_ant, UE_ant)
    max_possible_rank = min(min_dim, max_rank)

    all_sv = np.zeros((RB, min_dim), dtype=np.float32)
    rank_per_rb = np.zeros(RB, dtype=np.int32)
    w_all = np.zeros((RB, BS_ant, max_possible_rank), dtype=np.complex64)

    for rb in range(RB):
        # SVD(H_DL) = U S Vh; W = U[:, :rank]
        u, s, _ = _scipy_svd(h_dl[rb], full_matrices=False)
        all_sv[rb, :len(s)] = s.astype(np.float32)

        n_sig = int(np.sum(s > s[0] * rank_threshold)) if s[0] > 0 else 1
        rank_per_rb[rb] = min(max(n_sig, 1), max_possible_rank)

        n_cols = min(max_possible_rank, u.shape[1])
        w_all[rb, :, :n_cols] = u[:, :n_cols].astype(np.complex64)

    rank = int(np.clip(np.median(rank_per_rb), 1, max_possible_rank))
    w_dl = w_all[:, :, :rank].copy()

    # Unit-norm each column
    norms = np.linalg.norm(w_dl, axis=1, keepdims=True)  # [RB, 1, rank]
    w_dl = np.where(norms > 1e-10, w_dl / norms, w_dl).astype(np.complex64)

    return PrecodingResult(w_dl=w_dl, rank=rank, singular_values=all_sv)


def compute_dl_precoding_wideband(
    h_ul_est: np.ndarray,
    *,
    max_rank: int = 4,
    rank_threshold: float = 0.1,
) -> PrecodingResult:
    """Wideband DL precoding: single W across all RBs via eigendecomposition."""
    h = np.asarray(h_ul_est)
    if h.ndim != 4:
        raise ValueError(f"h_ul_est must be 4-D, got shape {h.shape}")

    T, RB, BS_ant, UE_ant = h.shape
    min_dim = min(BS_ant, UE_ant)
    max_possible_rank = min(min_dim, max_rank)

    h_avg = h.mean(axis=0)  # [RB, BS_ant, UE_ant]
    h_dl = _ul_to_dl(h_avg)  # [RB, BS_ant, UE_ant]

    # Wideband covariance R = (1/RB) * Σ_rb H_dl[rb] @ H_dl[rb]^H → [BS_ant, BS_ant]
    R = np.zeros((BS_ant, BS_ant), dtype=np.complex128)
    for rb in range(RB):
        R += h_dl[rb] @ h_dl[rb].conj().T
    R /= RB

    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = eigvals[::-1].copy()
    eigvecs = eigvecs[:, ::-1].copy()

    n_sig = int(np.sum(eigvals > eigvals[0] * rank_threshold)) if eigvals[0] > 0 else 1
    rank = min(max(n_sig, 1), max_possible_rank)

    w_wb = eigvecs[:, :rank].astype(np.complex64)
    w_dl = np.broadcast_to(w_wb[np.newaxis, :, :], (RB, BS_ant, rank)).copy()

    all_sv = np.zeros((RB, min_dim), dtype=np.float32)
    for rb in range(RB):
        _, s, _ = _scipy_svd(h_dl[rb], full_matrices=False)
        all_sv[rb, :len(s)] = s.astype(np.float32)

    return PrecodingResult(w_dl=w_dl, rank=rank, singular_values=all_sv)


def apply_precoding(
    h_channel: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """Apply precoding weights to get effective channel.

    Parameters
    ----------
    h_channel:
        ``[T, RB, BS_ant, UE_ant]`` complex — raw DL channel (contract convention).
    w:
        ``[RB, BS_ant, rank]`` complex — precoding weight matrix.

    Returns
    -------
    h_eff:
        ``[T, RB, rank, UE_ant]`` complex64 — effective channel.
        ``h_eff[t, rb] = W[rb]^H @ H_DL[t, rb]``
    """
    T, RB_h, BS_ant, UE_ant = h_channel.shape
    RB_w, BS_w, rank = w.shape
    assert RB_h == RB_w and BS_ant == BS_w

    # h_eff[t, rb, r, ue] = Σ_bs conj(w[rb, bs, r]) · h[t, rb, bs, ue]
    h_eff = np.einsum("fbr,tfbu->tfru", np.conj(w), h_channel)
    return h_eff.astype(np.complex64)
