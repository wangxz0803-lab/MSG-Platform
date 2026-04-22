"""Linear MMSE refinement of pilot-based channel estimates.

Given a least-squares estimate :math:`\\hat H_{\\mathrm{LS}} = H + n / x`
on the reference-signal subcarriers, the linear MMSE estimator that uses
the prior covariance :math:`R_{hh}` is (Kay, *Fundamentals of Statistical
Signal Processing: Estimation Theory*, §12.3):

.. math::

    \\hat H_{\\mathrm{MMSE}} = R_{hh} \\,
    \\big(R_{hh} + \\tfrac{1}{\\mathrm{SNR}} I\\big)^{-1}
    \\hat H_{\\mathrm{LS}}.

For 5G NR with reference signals defined by 3GPP TS 38.211 §§6.4 / 7.4,
``R_hh`` is the frequency-domain autocorrelation of the channel across the
RS subcarriers, which for an exponential power-delay profile (PDP)
:math:`P(\\tau) = (1/\\tau_{\\mathrm{rms}}) e^{-\\tau/\\tau_{\\mathrm{rms}}}`
admits the closed form

.. math::

    R[k, l] = \\frac{1}{1 + j\\, 2\\pi\\, (k-l)\\,\\Delta_f\\, \\tau_{\\mathrm{rms}}}.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

__all__ = [
    "mmse_estimate",
    "exponential_pdp_covariance",
]


def _validate_snr(snr_linear: float) -> float:
    if not np.isfinite(snr_linear) or snr_linear <= 0:
        raise ValueError(f"snr_linear must be positive and finite, got {snr_linear!r}.")
    return float(snr_linear)


def exponential_pdp_covariance(
    N_rs: int,
    tau_rms: float,
    delta_f: float,
    *,
    dtype: Literal["complex64", "complex128"] | np.dtype = "complex128",
) -> np.ndarray:
    """Frequency-domain autocovariance for a one-sided exponential PDP.

    The one-sided exponential PDP has impulse response statistics such that
    its Fourier transform on a uniform subcarrier grid is

    .. math::

        R[k, l] = \\mathbb{E}[H_k H_l^*] = \\frac{1}{1 + j\\, 2\\pi\\,
        (k-l)\\,\\Delta_f\\, \\tau_{\\mathrm{rms}}}.

    The power is normalised to ``R[k, k] = 1``; callers that wish to use a
    different channel-power prior may scale the output themselves.

    Parameters
    ----------
    N_rs:
        Number of reference-signal subcarriers in the covariance matrix.
    tau_rms:
        RMS delay spread in seconds.
    delta_f:
        Subcarrier spacing between consecutive RS subcarriers in Hz.  Note
        that if the pilots are every ``m``-th subcarrier, ``delta_f`` should
        be ``m * SCS``.
    dtype:
        Complex dtype of the returned matrix.

    Returns
    -------
    np.ndarray
        Hermitian PSD covariance matrix of shape ``[N_rs, N_rs]``.
    """
    if N_rs <= 0:
        raise ValueError("N_rs must be positive.")
    if tau_rms < 0 or not np.isfinite(tau_rms):
        raise ValueError("tau_rms must be non-negative and finite.")
    if not np.isfinite(delta_f):
        raise ValueError("delta_f must be finite.")

    dt = np.dtype(dtype)
    if dt.kind != "c":
        raise ValueError("dtype must be complex64 or complex128.")

    k = np.arange(N_rs)
    diff = k[:, None] - k[None, :]
    denom = 1.0 + 1j * 2.0 * np.pi * diff * float(delta_f) * float(tau_rms)
    R = 1.0 / denom
    # Enforce exact Hermitian symmetry against floating error.
    R = 0.5 * (R + np.conjugate(R.T))
    return R.astype(dt, copy=False)


def mmse_estimate(
    H_ls: np.ndarray,
    R_hh: np.ndarray,
    snr_linear: float,
    *,
    dtype: Literal["complex64", "complex128"] | np.dtype | None = None,
) -> np.ndarray:
    """Linear MMSE refinement of an LS channel estimate.

    Parameters
    ----------
    H_ls:
        LS channel estimate with shape ``[N_rs, ...]`` (trailing axes may be
        receive antennas, batches, ...).
    R_hh:
        Prior channel autocovariance on the RS subcarriers, shape
        ``[N_rs, N_rs]``.  Must be Hermitian positive semi-definite.
    snr_linear:
        Post-equalisation pilot SNR in linear scale (not dB).
    dtype:
        Optional output dtype (``complex64`` / ``complex128``).  Defaults to
        the input dtype of ``H_ls``.

    Returns
    -------
    np.ndarray
        MMSE channel estimate, same shape as ``H_ls``.

    Notes
    -----
    As ``snr_linear -> inf`` the noise term vanishes and
    ``W = R_hh (R_hh)^-1 = I``, i.e. MMSE collapses to the LS input.
    As ``snr_linear -> 0`` the estimator shrinks towards zero.
    """
    H = np.asarray(H_ls)
    R = np.asarray(R_hh)
    snr = _validate_snr(snr_linear)

    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"R_hh must be square 2-D, got shape {R.shape}.")
    if H.shape[0] != R.shape[0]:
        raise ValueError(f"Leading axis of H_ls ({H.shape[0]}) must match R_hh dim ({R.shape[0]}).")

    out_dtype = np.dtype(dtype) if dtype is not None else H.dtype
    if out_dtype.kind != "c":
        out_dtype = np.dtype(np.complex128)

    # Use high precision for the solve then cast.
    R64 = R.astype(np.complex128, copy=False)
    H64 = H.astype(np.complex128, copy=False)
    N = R64.shape[0]
    A = R64 + (1.0 / snr) * np.eye(N, dtype=np.complex128)

    # Solve A^T X^T = H^T  ->  X = A^-1 H (columns), then W H = R @ X.
    # Flatten trailing axes for a single batched solve.
    flat = H64.reshape(N, -1)
    X = np.linalg.solve(A, flat)
    H_mmse_flat = R64 @ X

    out = H_mmse_flat.reshape(H64.shape)
    return out.astype(out_dtype, copy=False)
