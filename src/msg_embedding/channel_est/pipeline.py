"""Unified three-tier channel-estimation entry point.

Data collection materialises all three estimates (``ideal``, ``ls_linear``,
``ls_mmse``) so the training side can pick one via configuration.  The
reference-signal layout follows 3GPP TS 38.211 §§6.4 / 7.4.

Modes
-----
``ideal``
    Bypass every estimator and return the genie channel ``h_true`` directly.
``ls_linear``
    Least-squares estimate (:mod:`.ls`) followed by separable linear
    frequency/time interpolation (:mod:`.interpolate`).
``ls_mmse``
    LS on the RS, linear-MMSE refinement on the RS (:mod:`.mmse`) using an
    exponential-PDP prior, then separable linear interpolation.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .interpolate import interp_2d
from .ls import ls_estimate
from .mmse import exponential_pdp_covariance, mmse_estimate

__all__ = ["estimate_channel", "EstimationMode"]

EstimationMode = Literal["ideal", "ls_linear", "ls_mmse"]


def _ensure_complex(
    arr: np.ndarray, dtype: Literal["complex64", "complex128"] | np.dtype | None
) -> np.ndarray:
    """Promote ``arr`` to a complex dtype; honour user override if given."""
    if dtype is not None:
        return arr.astype(np.dtype(dtype), copy=False)
    if arr.dtype.kind == "c":
        return arr
    return arr.astype(np.complex128, copy=False)


def _gather_rs_grid(H_rs_flat: np.ndarray, n_freq: int, n_time: int) -> np.ndarray:
    """Reshape ``[n_freq * n_time, ...]`` into ``[n_freq, n_time, ...]``."""
    expected = n_freq * n_time
    if H_rs_flat.shape[0] != expected:
        raise ValueError(
            f"Expected {expected} RS samples (n_freq={n_freq} * n_time={n_time}), "
            f"got {H_rs_flat.shape[0]}."
        )
    return H_rs_flat.reshape((n_freq, n_time) + H_rs_flat.shape[1:])


def estimate_channel(
    Y_rs: np.ndarray,
    X_rs: np.ndarray,
    rs_positions_freq: np.ndarray,
    rs_positions_time: np.ndarray,
    N_sc: int,
    N_sym: int,
    mode: EstimationMode,
    *,
    h_true: np.ndarray | None = None,
    pdp_prior: dict[str, Any] | None = None,
    snr_db: float | None = None,
    dtype: Literal["complex64", "complex128"] | np.dtype | None = None,
) -> np.ndarray:
    """Return the full-grid channel estimate for the requested mode.

    Parameters
    ----------
    Y_rs:
        Received RS samples, shape ``[N_freq_rs * N_time_rs, N_rx]`` where
        the outer axis is ordered ``(freq slow, time fast)`` or equivalently
        flattened ``[N_freq_rs, N_time_rs]``.  A flat ``[N_rs]`` vector is
        also accepted (``N_rx`` inferred as 1).
    X_rs:
        Corresponding transmitted pilots, shape ``[N_freq_rs * N_time_rs]``
        or ``[N_freq_rs * N_time_rs, 1]``.
    rs_positions_freq:
        Subcarrier indices of the frequency RS, shape ``[N_freq_rs]``.
    rs_positions_time:
        OFDM symbol indices of the time RS, shape ``[N_time_rs]``.
    N_sc, N_sym:
        Target full-grid size.
    mode:
        One of ``{"ideal", "ls_linear", "ls_mmse"}``.
    h_true:
        Required for ``mode == "ideal"``; shape ``[N_sc, N_sym, N_rx]``.
    pdp_prior:
        Required for ``mode == "ls_mmse"``.  Dictionary with keys
        ``tau_rms`` (s) and ``delta_f`` (Hz of the pilot grid).
    snr_db:
        Required for ``mode == "ls_mmse"``.  Pilot SNR in dB.
    dtype:
        Optional complex output dtype.

    Returns
    -------
    np.ndarray
        Full-grid estimate with shape ``[N_sc, N_sym, N_rx]``.
    """
    n_freq = int(np.asarray(rs_positions_freq).shape[0])
    n_time = int(np.asarray(rs_positions_time).shape[0])

    if mode == "ideal":
        if h_true is None:
            raise ValueError("mode='ideal' requires h_true.")
        ht = np.asarray(h_true)
        if ht.ndim == 2:
            ht = ht[..., None]
        if ht.shape[:2] != (N_sc, N_sym):
            raise ValueError(
                f"h_true shape {ht.shape} must start with (N_sc={N_sc}, N_sym={N_sym})."
            )
        return _ensure_complex(ht, dtype)

    # Shared LS step — valid for ls_linear and ls_mmse.
    Y = np.asarray(Y_rs)
    if Y.ndim == 1:
        Y = Y[:, None]
    H_ls_flat = ls_estimate(Y, np.asarray(X_rs), dtype=dtype)  # [N_rs, N_rx]

    if mode == "ls_linear":
        H_ls_grid = _gather_rs_grid(H_ls_flat, n_freq, n_time)
        return interp_2d(
            H_ls_grid,
            rs_positions_freq,
            rs_positions_time,
            N_sc,
            N_sym,
            method="linear",
        )

    if mode == "ls_mmse":
        if pdp_prior is None or "tau_rms" not in pdp_prior or "delta_f" not in pdp_prior:
            raise ValueError("mode='ls_mmse' requires pdp_prior={'tau_rms': ..., 'delta_f': ...}.")
        if snr_db is None:
            raise ValueError("mode='ls_mmse' requires snr_db.")
        snr_linear = float(10.0 ** (float(snr_db) / 10.0))

        H_ls_grid = _gather_rs_grid(H_ls_flat, n_freq, n_time)  # [Nf, Nt, Nrx]
        # MMSE along the frequency axis (Nf) for each time column & rx.
        Nf, Nt = H_ls_grid.shape[:2]
        R_hh = exponential_pdp_covariance(
            Nf,
            float(pdp_prior["tau_rms"]),
            float(pdp_prior["delta_f"]),
            dtype=(dtype if dtype is not None else "complex128"),
        )
        # mmse_estimate supports batched trailing axes.
        H_mmse_grid = mmse_estimate(H_ls_grid, R_hh, snr_linear, dtype=dtype)

        # Optional MMSE along the time axis would require a Doppler prior;
        # fall back to linear time interpolation which is the standard
        # industry practice (cf. Sesia et al., *LTE — The UMTS Long Term
        # Evolution*, §9.2).
        return interp_2d(
            H_mmse_grid,
            rs_positions_freq,
            rs_positions_time,
            N_sc,
            N_sym,
            method="linear",
        )

    raise ValueError(f"Unknown mode {mode!r}; expected one of ideal / ls_linear / ls_mmse.")
