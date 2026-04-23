"""Shared interference-aware channel estimation for all data source adapters.

Constructs received signals that physically model multi-user (UL) or
multi-cell (DL) interference, then delegates to the unified
:func:`channel_est.estimate_channel` pipeline for LS/MMSE estimation.

UL model (BS receiver):
    Y = h_serving · x_SRS_serving + Σ_k h_intf_k · x_SRS_k + noise

DL model (UE receiver):
    Y = h_serving · x_CSIRS_serving + Σ_k h_intf_k · x_CSIRS_k + noise
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

Direction = Literal["UL", "DL"]


@dataclass(frozen=True)
class InterferenceEstResult:
    """Result of interference-aware channel estimation."""

    h_est: np.ndarray
    sir_dB: float | None


class PairedChannelResult(TypedDict):
    h_ul_true: np.ndarray
    h_ul_est: np.ndarray
    h_dl_true: np.ndarray
    h_dl_est: np.ndarray
    ul_sir_dB: float | None
    dl_sir_dB: float | None
    num_interfering_ues: int | None


def _generate_interferer_pilots_srs(
    num_rb: int,
    cell_id: int,
    ue_index: int,
) -> np.ndarray:
    """Generate SRS pilots for an interfering UE.

    Each UE uses a different ZC root index derived from (cell_id, ue_index)
    to model realistic SRS code-domain separation.
    Returns shape ``[num_rb]`` complex128.
    """
    from msg_embedding.ref_signals.zc import zadoff_chu

    Nzc = num_rb
    if Nzc < 2:
        return np.ones(max(num_rb, 1), dtype=np.complex128)

    nzc_prime = Nzc
    while nzc_prime >= 2:
        if _is_prime(nzc_prime):
            break
        nzc_prime -= 1
    if nzc_prime < 2:
        nzc_prime = 2

    u = max(1, ((cell_id + ue_index + 1) * 7 + 3) % nzc_prime)
    if u == 0:
        u = 1

    seq = zadoff_chu(u, nzc_prime)
    if len(seq) < num_rb:
        seq = np.concatenate([seq, np.ones(num_rb - len(seq), dtype=np.complex128)])
    else:
        seq = seq[:num_rb]
    return seq


def _generate_interferer_pilots_csirs(
    num_rb: int,
    cell_id: int,
) -> np.ndarray:
    """Generate CSI-RS pilots for an interfering cell.

    Uses the standard CSI-RS sequence seeded by the interferer's cell_id.
    Returns shape ``[num_rb]`` complex128.
    """
    from msg_embedding.ref_signals.csi_rs import csi_rs_sequence

    n_rb = max(1, num_rb)
    seq = csi_rs_sequence(
        n_ID=cell_id % 1024,
        slot=0,
        symbol=0,
        density=1.0,
        n_rb=n_rb,
    )
    if len(seq) < num_rb:
        seq = np.concatenate([seq, np.ones(num_rb - len(seq), dtype=np.complex128)])
    else:
        seq = seq[:num_rb]
    return seq


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    r = int(math.isqrt(n))
    return all(n % i != 0 for i in range(3, r + 1, 2))


def _build_pilot_grid(
    h_true: np.ndarray,
    RB: int,
    T: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Compute pilot RS grid positions matching NR DMRS density.

    Returns (rs_freq, rs_time, n_freq, n_time).
    """
    pilot_rb_spacing = 1
    pilot_sym_spacing = max(1, min(4, T))
    rs_freq = np.arange(0, RB, pilot_rb_spacing, dtype=np.int64)
    rs_time = np.arange(0, T, pilot_sym_spacing, dtype=np.int64)

    if len(rs_freq) == 0:
        rs_freq = np.array([0], dtype=np.int64)
    if len(rs_time) == 0:
        rs_time = np.array([0], dtype=np.int64)

    return rs_freq, rs_time, len(rs_freq), len(rs_time)


def estimate_channel_with_interference(
    h_serving_true: np.ndarray,
    h_interferers: np.ndarray | None,
    pilots_serving: np.ndarray,
    interferer_cell_ids: list[int] | None,
    direction: Direction,
    snr_dB: float,
    rng: np.random.Generator,
    *,
    est_mode: str = "ls_linear",
    tau_rms_ns: float = 100.0,
    subcarrier_spacing: int | float = 30000,
    serving_cell_id: int = 0,
    num_interfering_ues: int = 3,
) -> InterferenceEstResult:
    """Estimate the serving channel with physically modelled interference.

    Parameters
    ----------
    h_serving_true:
        ``[T, RB, Ant_tx, Ant_rx]`` complex64 ideal serving channel.
    h_interferers:
        ``[K-1, T, RB, Ant_tx, Ant_rx]`` complex64 interferer channels.
        ``None`` if no interferers (falls back to noise-only estimation).
    pilots_serving:
        ``[RB]`` complex128 serving cell's reference signal sequence.
    interferer_cell_ids:
        Physical cell IDs for each interferer (length K-1).
        For UL direction, these are used to derive per-UE ZC roots.
        For DL direction, these seed interferer CSI-RS sequences.
    direction:
        ``"UL"`` — BS receives SRS from multiple UEs.
        ``"DL"`` — UE receives CSI-RS from multiple cells.
    snr_dB:
        Signal-to-noise ratio in dB.
    rng:
        Numpy random generator for noise.
    est_mode:
        Channel estimation mode: ``"ideal"``, ``"ls_linear"``, ``"ls_mmse"``.
    tau_rms_ns:
        RMS delay spread in nanoseconds (for MMSE prior).
    subcarrier_spacing:
        Subcarrier spacing in Hz.
    serving_cell_id:
        PCI of the serving cell (for pilot generation of interfering UEs in UL).
    num_interfering_ues:
        Number of interfering UEs per cell (UL only). Each gets a distinct
        ZC root. Ignored for DL.

    Returns
    -------
    InterferenceEstResult
        Contains ``h_est`` (estimated channel with interference residual)
        and ``sir_dB`` (signal-to-interference ratio, None if no interferers).
    """
    from msg_embedding.channel_est import estimate_channel

    T, RB, Ant_tx, Ant_rx = h_serving_true.shape

    if est_mode == "ideal":
        h_freq_time = h_serving_true.transpose(1, 0, 2, 3).reshape(RB, T, Ant_tx * Ant_rx)
        est = estimate_channel(
            Y_rs=np.zeros((RB * T, Ant_tx * Ant_rx), dtype=np.complex64),
            X_rs=np.ones(RB * T, dtype=np.complex64),
            rs_positions_freq=np.arange(RB, dtype=np.int64),
            rs_positions_time=np.arange(T, dtype=np.int64),
            N_sc=RB,
            N_sym=T,
            mode="ideal",
            h_true=h_freq_time,
            dtype="complex64",
        )
        return InterferenceEstResult(
            h_est=est.reshape(RB, T, Ant_tx, Ant_rx).transpose(1, 0, 2, 3).astype(np.complex64),
            sir_dB=None,
        )

    # -- Build pilot grid (NR DMRS density) --
    rs_freq, rs_time, n_freq, n_time = _build_pilot_grid(h_serving_true, RB, T)

    # -- Extract serving channel at pilot positions --
    # h_serving_true: [T, RB, Ant_tx, Ant_rx]
    h_at_pilots = h_serving_true[np.ix_(rs_time, rs_freq)]  # [n_time, n_freq, Ant_tx, Ant_rx]
    h_at_pilots = h_at_pilots.transpose(1, 0, 2, 3)  # [n_freq, n_time, Ant_tx, Ant_rx]

    # Serving pilot sequence at RS frequencies
    pilot_seq = pilots_serving[rs_freq]  # [n_freq]

    # -- Thermal noise --
    noise_std = 10.0 ** (-snr_dB / 20.0) / math.sqrt(2.0)
    noise = noise_std * (
        rng.standard_normal(h_at_pilots.shape)
        + 1j * rng.standard_normal(h_at_pilots.shape)
    )

    # -- Interference construction --
    # Start with serving signal: H_serving * X_serving + noise
    # Y_pilot = H_serving * X_serving (before adding interference + noise)
    X_serving = pilot_seq[:, None, None, None]  # [n_freq, 1, 1, 1]
    Y_serving = h_at_pilots * X_serving  # [n_freq, n_time, Ant_tx, Ant_rx]

    sir_dB: float | None = None
    interference_total = np.zeros_like(h_at_pilots, dtype=np.complex128)

    if h_interferers is not None and len(h_interferers) > 0:
        K_minus_1 = h_interferers.shape[0]

        if direction == "UL":
            # UL: BS receives SRS from multiple UEs in the same cell.
            # Each interfering UE transmits SRS with a different ZC root.
            # h_interferers: [K-1, T, RB, BS_ant, UE_ant]
            # We model num_interfering_ues UEs per interferer cell.
            n_intf_total = 0
            for cell_k in range(K_minus_1):
                h_intf_k = h_interferers[cell_k]  # [T, RB, BS_ant, UE_ant]
                h_intf_at_pilots = h_intf_k[np.ix_(rs_time, rs_freq)]
                h_intf_at_pilots = h_intf_at_pilots.transpose(1, 0, 2, 3)

                intf_cell_id = (
                    interferer_cell_ids[cell_k]
                    if interferer_cell_ids is not None and cell_k < len(interferer_cell_ids)
                    else serving_cell_id + cell_k + 1
                )

                for ue_idx in range(num_interfering_ues):
                    intf_pilots = _generate_interferer_pilots_srs(
                        RB, intf_cell_id, ue_idx,
                    )
                    X_intf = intf_pilots[rs_freq][:, None, None, None]
                    interference_total += h_intf_at_pilots * X_intf
                    n_intf_total += 1

            logger.debug(
                "ul_interference_injected",
                num_interferer_cells=K_minus_1,
                num_ues_per_cell=num_interfering_ues,
                total_interfering_signals=n_intf_total,
            )

        else:
            # DL: UE receives CSI-RS from multiple cells.
            # Each interferer cell transmits CSI-RS with its own Gold-PRBS sequence.
            # h_interferers: [K-1, T, RB, BS_ant, UE_ant]
            for cell_k in range(K_minus_1):
                h_intf_k = h_interferers[cell_k]
                h_intf_at_pilots = h_intf_k[np.ix_(rs_time, rs_freq)]
                h_intf_at_pilots = h_intf_at_pilots.transpose(1, 0, 2, 3)

                intf_cell_id = (
                    interferer_cell_ids[cell_k]
                    if interferer_cell_ids is not None and cell_k < len(interferer_cell_ids)
                    else serving_cell_id + cell_k + 1
                )

                intf_pilots = _generate_interferer_pilots_csirs(RB, intf_cell_id)
                X_intf = intf_pilots[rs_freq][:, None, None, None]
                interference_total += h_intf_at_pilots * X_intf

            logger.debug(
                "dl_interference_injected",
                num_interferer_cells=K_minus_1,
            )

        # Compute SIR: P_serving / P_interference
        p_serving = float(np.mean(np.abs(Y_serving) ** 2))
        p_intf = float(np.mean(np.abs(interference_total) ** 2))
        if p_intf > 1e-30:
            sir_dB = float(10.0 * math.log10(max(p_serving / p_intf, 1e-15)))
            sir_dB = max(-50.0, min(50.0, sir_dB))
        else:
            sir_dB = 49.9

    # -- Compose received signal: Y = H_s·X_s + Σ H_k·X_k + noise --
    Y_total = Y_serving + interference_total + noise.astype(np.complex128)

    # -- Flatten for the estimation pipeline --
    # Y_total: [n_freq, n_time, Ant_tx, Ant_rx]
    Y_flat = Y_total.reshape(n_freq * n_time, Ant_tx * Ant_rx)

    # Pilot sequence tiled for the flat layout (freq-slow, time-fast)
    X_flat = np.repeat(pilot_seq, n_time).astype(np.complex128)

    est = estimate_channel(
        Y_rs=Y_flat.astype(np.complex64),
        X_rs=X_flat.astype(np.complex64),
        rs_positions_freq=rs_freq,
        rs_positions_time=rs_time,
        N_sc=RB,
        N_sym=T,
        mode=est_mode,  # type: ignore[arg-type]
        h_true=h_serving_true.transpose(1, 0, 2, 3).reshape(RB, T, Ant_tx * Ant_rx),
        pdp_prior={
            "tau_rms": tau_rms_ns * 1e-9,
            "delta_f": 12 * subcarrier_spacing * 1,
        },
        snr_db=snr_dB,
        dtype="complex64",
    )

    h_est = est.reshape(RB, T, Ant_tx, Ant_rx).transpose(1, 0, 2, 3).astype(np.complex64)

    return InterferenceEstResult(h_est=h_est, sir_dB=sir_dB)


def estimate_paired_channels(
    h_dl_true: np.ndarray,
    h_interferers_dl: np.ndarray | None,
    serving_cell_id: int,
    interferer_cell_ids: list[int] | None,
    snr_dB: float,
    rng: np.random.Generator,
    *,
    est_mode: str = "ls_linear",
    tau_rms_ns: float = 100.0,
    subcarrier_spacing: int | float = 30000,
    num_interfering_ues: int = 3,
    reciprocity_noise_scale: float = 0.01,
    num_rb: int | None = None,
) -> PairedChannelResult:
    """Generate paired UL+DL estimated channels from a single DL ground truth.

    Uses TDD reciprocity: H_UL = conj(H_DL^T) with small perturbation.
    Both UL and DL channels are estimated with direction-appropriate
    interference injection.

    Parameters
    ----------
    h_dl_true:
        ``[T, RB, BS_ant, UE_ant]`` complex64 ideal DL channel.
    h_interferers_dl:
        ``[K-1, T, RB, BS_ant, UE_ant]`` complex64 DL interferer channels.
    serving_cell_id:
        PCI of the serving cell.
    interferer_cell_ids:
        PCIs of interferer cells (length K-1).
    snr_dB, rng, est_mode, tau_rms_ns, subcarrier_spacing:
        Passed through to :func:`estimate_channel_with_interference`.
    num_interfering_ues:
        Number of interfering UEs per cell for UL estimation.
    reciprocity_noise_scale:
        Scale of the small independent noise added to UL channel for
        TDD reciprocity imperfections.
    num_rb:
        Number of RBs (inferred from h_dl_true if None).

    Returns
    -------
    dict with keys:
        h_ul_true, h_ul_est, h_dl_true, h_dl_est,
        ul_sir_dB, dl_sir_dB, num_interfering_ues
    """
    T, RB, BS_ant, UE_ant = h_dl_true.shape
    if num_rb is None:
        num_rb = RB

    # -- TDD reciprocity: H_UL = conj(H_DL^T) --
    h_ul_true = np.conj(h_dl_true.transpose(0, 1, 3, 2))
    rng_ul = np.random.Generator(rng.bit_generator.jumped())  # type: ignore[attr-defined]
    h_ul_true = (
        h_ul_true
        + reciprocity_noise_scale
        * (
            rng_ul.standard_normal(h_ul_true.shape)
            + 1j * rng_ul.standard_normal(h_ul_true.shape)
        )
        / math.sqrt(2.0)
    ).astype(np.complex64)

    # Interferers in UL domain (conjugate transpose of DL interferers)
    h_interferers_ul: np.ndarray | None = None
    if h_interferers_dl is not None and len(h_interferers_dl) > 0:
        h_interferers_ul = np.conj(
            h_interferers_dl.transpose(0, 1, 2, 4, 3)
        ).astype(np.complex64)

    # -- Generate serving-cell reference signals --
    from msg_embedding.data.sources.internal_sim import (
        _generate_pilots_csirs,
        _generate_pilots_srs,
    )

    pilots_srs = _generate_pilots_srs(num_rb, serving_cell_id)
    pilots_csirs = _generate_pilots_csirs(num_rb, serving_cell_id)

    # -- UL estimation (BS receives SRS with multi-UE interference) --
    rng_ul_est = np.random.Generator(rng.bit_generator.jumped())  # type: ignore[attr-defined]
    ul_result = estimate_channel_with_interference(
        h_serving_true=h_ul_true,
        h_interferers=h_interferers_ul,
        pilots_serving=pilots_srs,
        interferer_cell_ids=interferer_cell_ids,
        direction="UL",
        snr_dB=snr_dB,
        rng=rng_ul_est,
        est_mode=est_mode,
        tau_rms_ns=tau_rms_ns,
        subcarrier_spacing=subcarrier_spacing,
        serving_cell_id=serving_cell_id,
        num_interfering_ues=num_interfering_ues,
    )

    # -- DL estimation (UE receives CSI-RS with multi-cell interference) --
    rng_dl_est = np.random.Generator(rng.bit_generator.jumped())  # type: ignore[attr-defined]
    dl_result = estimate_channel_with_interference(
        h_serving_true=h_dl_true,
        h_interferers=h_interferers_dl,
        pilots_serving=pilots_csirs,
        interferer_cell_ids=interferer_cell_ids,
        direction="DL",
        snr_dB=snr_dB,
        rng=rng_dl_est,
        est_mode=est_mode,
        tau_rms_ns=tau_rms_ns,
        subcarrier_spacing=subcarrier_spacing,
        serving_cell_id=serving_cell_id,
    )

    return {
        "h_ul_true": h_ul_true,
        "h_ul_est": ul_result.h_est,
        "h_dl_true": h_dl_true.astype(np.complex64),
        "h_dl_est": dl_result.h_est,
        "ul_sir_dB": ul_result.sir_dB,
        "dl_sir_dB": dl_result.sir_dB,
        "num_interfering_ues": num_interfering_ues if h_interferers_ul is not None else None,
    }
