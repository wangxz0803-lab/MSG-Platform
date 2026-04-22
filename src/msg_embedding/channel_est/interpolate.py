"""Frequency- and time-domain interpolation for pilot-based channel estimation.

After the least-squares step in :mod:`.ls` we have estimates only on the
reference-signal (RS) subcarriers / OFDM symbols defined by 3GPP
TS 38.211 §§6.4, 7.4.  This module lifts those estimates to the full
resource grid using standard 1-D and separable 2-D interpolation over the
real and imaginary parts (see Kay, *Fundamentals of Statistical Signal
Processing: Estimation Theory*, §13.3 on linear reconstruction from regular
samples).

Three methods are supported:

* ``linear`` — :class:`scipy.interpolate.interp1d` with ``kind='linear'``.
* ``cubic``  — :class:`scipy.interpolate.interp1d` with ``kind='cubic'``.
* ``spline`` — :class:`scipy.interpolate.CubicSpline` (natural BC).

For values outside the RS support the nearest RS value is used (constant
extrapolation); this matches the behaviour of NR channel estimation in the
edge resource blocks.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.interpolate import CubicSpline, interp1d

__all__ = ["interp_frequency", "interp_time", "interp_2d"]

InterpMethod = Literal["linear", "cubic", "spline"]


def _build_axis_interp(
    x_known: np.ndarray, y_known: np.ndarray, method: InterpMethod
) -> tuple[object, float, float]:
    """Build a callable interpolator for one real 1-D axis, with edge clamping.

    Returns ``(fn, x_min, x_max)``; ``fn(x)`` evaluates at arbitrary samples
    and callers must clamp ``x`` into ``[x_min, x_max]`` themselves to get
    nearest-value extrapolation.
    """
    n = int(x_known.shape[0])
    if n == 0:
        raise ValueError("At least one reference-signal position is required.")
    if n == 1:
        val = float(y_known[0])
        x0 = float(x_known[0])
        return (
            lambda x, v=val: np.full_like(np.asarray(x, dtype=float), v),
            x0,
            x0,
        )

    if method == "linear":
        fn = interp1d(
            x_known,
            y_known,
            kind="linear",
            bounds_error=False,
            fill_value=(float(y_known[0]), float(y_known[-1])),
            assume_sorted=True,
        )
    elif method == "cubic":
        kind = "cubic" if n >= 4 else "linear"
        fn = interp1d(
            x_known,
            y_known,
            kind=kind,
            bounds_error=False,
            fill_value=(float(y_known[0]), float(y_known[-1])),
            assume_sorted=True,
        )
    elif method == "spline":
        if n >= 2:
            fn = CubicSpline(x_known, y_known, bc_type="natural", extrapolate=True)
        else:  # pragma: no cover — already handled above.
            fn = interp1d(x_known, y_known, kind="linear")
    else:
        raise ValueError(f"Unknown interpolation method {method!r}.")
    return fn, float(x_known[0]), float(x_known[-1])


def _interp_1d_complex(
    values: np.ndarray,
    pos_known: np.ndarray,
    pos_full: np.ndarray,
    method: InterpMethod,
) -> np.ndarray:
    """Complex 1-D interpolation by independent real / imag reconstruction.

    ``values`` has shape ``[N_known, *trailing]``; returns ``[N_full, *trailing]``.
    """
    values = np.asarray(values)
    pos_known = np.asarray(pos_known, dtype=np.float64).ravel()
    pos_full = np.asarray(pos_full, dtype=np.float64).ravel()

    if pos_known.shape[0] != values.shape[0]:
        raise ValueError(
            f"pos_known length {pos_known.shape[0]} must match values leading axis "
            f"{values.shape[0]}."
        )
    # Require monotonic increasing positions (required by SciPy spline).
    if not np.all(np.diff(pos_known) > 0):
        order = np.argsort(pos_known)
        pos_known = pos_known[order]
        values = values[order]

    out_shape = (pos_full.shape[0],) + values.shape[1:]
    out_dtype = np.complex64 if values.dtype == np.complex64 else np.complex128
    out = np.empty(out_shape, dtype=out_dtype)

    # Flatten trailing axes so we can loop over columns cheaply.
    flat = values.reshape(values.shape[0], -1)
    flat_out = out.reshape(pos_full.shape[0], -1)

    x_clamped = np.clip(pos_full, pos_known[0], pos_known[-1])

    for j in range(flat.shape[1]):
        re_fn, _, _ = _build_axis_interp(pos_known, flat[:, j].real.astype(np.float64), method)
        im_fn, _, _ = _build_axis_interp(pos_known, flat[:, j].imag.astype(np.float64), method)
        re = np.asarray(re_fn(x_clamped))
        im = np.asarray(im_fn(x_clamped))
        flat_out[:, j] = (re + 1j * im).astype(out_dtype, copy=False)

    return out


def interp_frequency(
    H_rs: np.ndarray,
    rs_positions: np.ndarray,
    N_sc: int,
    method: InterpMethod = "linear",
) -> np.ndarray:
    """Interpolate per-RS-subcarrier estimates onto the full frequency grid.

    Parameters
    ----------
    H_rs:
        Channel estimates on RS subcarriers, shape ``[N_rs, ...]``.
    rs_positions:
        Integer subcarrier indices of the RS samples, shape ``[N_rs]``.
    N_sc:
        Total number of subcarriers to fill, ``0 <= k < N_sc``.
    method:
        Interpolation method; see module docstring.

    Returns
    -------
    np.ndarray
        Interpolated channel, shape ``[N_sc, ...]``.
    """
    if N_sc <= 0:
        raise ValueError("N_sc must be positive.")
    full = np.arange(N_sc, dtype=np.float64)
    return _interp_1d_complex(np.asarray(H_rs), np.asarray(rs_positions), full, method)


def interp_time(
    H_symbols: np.ndarray,
    sym_positions: np.ndarray,
    N_sym: int,
    method: InterpMethod = "linear",
) -> np.ndarray:
    """Interpolate pilot-symbol channel estimates onto every OFDM symbol.

    Parameters
    ----------
    H_symbols:
        Channel estimates at pilot OFDM symbols, shape ``[N_pilot_sym, ...]``.
    sym_positions:
        Integer OFDM symbol indices of the pilots, shape ``[N_pilot_sym]``.
    N_sym:
        Total number of OFDM symbols in the slot/frame.
    method:
        Interpolation method; see module docstring.

    Returns
    -------
    np.ndarray
        Interpolated channel, shape ``[N_sym, ...]``.
    """
    if N_sym <= 0:
        raise ValueError("N_sym must be positive.")
    full = np.arange(N_sym, dtype=np.float64)
    return _interp_1d_complex(np.asarray(H_symbols), np.asarray(sym_positions), full, method)


def interp_2d(
    H_rs_grid: np.ndarray,
    freq_pos: np.ndarray,
    time_pos: np.ndarray,
    N_sc: int,
    N_sym: int,
    method: InterpMethod = "linear",
) -> np.ndarray:
    """Separable 2-D interpolation: frequency first, then time.

    Parameters
    ----------
    H_rs_grid:
        Channel on the RS lattice with shape ``[N_freq_rs, N_time_rs, ...]``.
    freq_pos:
        Subcarrier indices of the frequency RS, shape ``[N_freq_rs]``.
    time_pos:
        OFDM symbol indices of the time RS, shape ``[N_time_rs]``.
    N_sc, N_sym:
        Target grid size.
    method:
        Interpolation method used for both axes.

    Returns
    -------
    np.ndarray
        Full-grid channel, shape ``[N_sc, N_sym, ...]``.

    Notes
    -----
    Separable reconstruction is optimal only when the underlying 2-D random
    field is separable (Kay §13.3); for practical NR geometries this is a
    close-enough approximation that is cheap compared to a full MMSE 2-D
    filter.
    """
    H_rs_grid = np.asarray(H_rs_grid)
    if H_rs_grid.ndim < 2:
        raise ValueError("H_rs_grid must have at least 2 dimensions (freq, time).")

    # Step 1: interpolate along frequency for each RS time column.
    freq_full = interp_frequency(H_rs_grid, freq_pos, N_sc, method=method)
    # freq_full shape: [N_sc, N_time_rs, ...]; move time axis to front.
    freq_full_ax = np.moveaxis(freq_full, 1, 0)  # [N_time_rs, N_sc, ...]

    # Step 2: interpolate along time for each subcarrier.
    full = interp_time(freq_full_ax, time_pos, N_sym, method=method)  # [N_sym, N_sc, ...]
    return np.moveaxis(full, 0, 1)  # [N_sc, N_sym, ...]
