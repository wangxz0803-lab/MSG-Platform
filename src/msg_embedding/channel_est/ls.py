"""Least-Squares (LS) channel estimation on reference-signal positions.

The LS estimator for a single-tap observation ``y = h * x + n`` is the
classical deconvolution

.. math::

    \\hat h_{\\mathrm{LS}} = \\frac{y \\, x^*}{|x|^2}

which is the minimum-variance unbiased estimator when no prior on ``h`` is
used and ``n`` is zero-mean (see Kay, *Fundamentals of Statistical Signal
Processing: Estimation Theory*, ch. 8).  In 5G NR the reference signals
(DMRS / SRS / CSI-RS) described in 3GPP TS 38.211 §§6.4, 7.4 occupy a known
subset of resource elements; LS is applied on those elements and the full
grid is then interpolated (see :mod:`.interpolate`).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

__all__ = ["ls_estimate"]

_DEFAULT_EPS: float = 1e-12


def ls_estimate(
    Y_rs: np.ndarray,
    X_rs: np.ndarray,
    *,
    eps: float = _DEFAULT_EPS,
    dtype: Literal["complex64", "complex128"] | np.dtype | None = None,
) -> np.ndarray:
    """Least-squares channel estimate at the reference-signal subcarriers.

    Given the received OFDM-domain samples ``Y_rs`` on the reference-signal
    resource elements and the transmitted pilots ``X_rs``, returns

    .. math::

        \\hat H_{\\mathrm{LS}}[k, r] = \\frac{Y_{\\mathrm{RS}}[k, r]\\,
        X_{\\mathrm{RS}}^*[k]}{|X_{\\mathrm{RS}}[k]|^2 + \\varepsilon}

    where ``k`` indexes the ``N_rs`` reference-signal positions and ``r`` the
    receive antennas.

    Parameters
    ----------
    Y_rs:
        Received samples on the reference-signal positions.  Shape
        ``[N_rs, N_rx]`` or batched ``[B, N_rs, N_rx]``; a flat vector of
        length ``N_rs`` is also accepted (``N_rx`` inferred as 1).
    X_rs:
        Transmitted reference-signal symbols, shape ``[N_rs]`` or
        ``[B, N_rs]``.  Broadcast against ``Y_rs`` over ``N_rx``.
    eps:
        Small regulariser added to ``|X_rs|^2`` so zero pilots do not blow up
        the estimate.  Defaults to ``1e-12``.
    dtype:
        Optional output dtype (``complex64`` / ``complex128``).  If ``None``
        the result follows NumPy's complex promotion of the inputs.

    Returns
    -------
    np.ndarray
        ``\\hat H`` at the reference-signal positions, same batch shape as
        ``Y_rs`` and trailing ``[N_rs, N_rx]``.

    Notes
    -----
    The variance of the estimate for unit-modulus pilots is
    ``Var(\\hat H) = sigma^2 / |X|^2``; see Kay §8.5.
    """
    Y = np.asarray(Y_rs)
    X = np.asarray(X_rs)

    if Y.ndim == 1:
        Y = Y[:, None]

    # Align X's trailing axis with Y's N_rx axis.  We always append a
    # trailing length-1 axis unless X already matches Y exactly; then we
    # rely on NumPy's leading-axis broadcasting for any missing batch dims.
    X_bcast = X if X.ndim == Y.ndim else X[..., None]
    # Sanity check: after trailing-axis alignment, X must broadcast against Y.
    try:
        np.broadcast_shapes(X_bcast.shape, Y.shape)
    except ValueError as exc:
        raise ValueError(
            f"X_rs shape {X.shape} not broadcastable to Y_rs shape {Y.shape}."
        ) from exc

    # Promote to a common complex dtype to avoid silently downcasting real
    # pilots to complex64 when Y is complex128.
    out_dtype: np.dtype
    if dtype is not None:
        out_dtype = np.dtype(dtype)
    else:
        out_dtype = np.result_type(Y.dtype, X.dtype, np.complex64)
        if out_dtype.kind != "c":
            out_dtype = np.dtype(np.complex128)

    Yc = Y.astype(out_dtype, copy=False)
    Xc = X_bcast.astype(out_dtype, copy=False)

    denom = (Xc.real * Xc.real + Xc.imag * Xc.imag).astype(
        np.float64 if out_dtype == np.complex128 else np.float32, copy=False
    )
    # Regularise only where the pilot is effectively zero: this keeps the
    # estimate exact for normal (non-zero) pilots while avoiding blow-up at
    # punctured / DC subcarriers.
    denom_safe = np.where(denom > eps, denom, eps)
    return (Yc * np.conjugate(Xc) / denom_safe).astype(out_dtype, copy=False)
