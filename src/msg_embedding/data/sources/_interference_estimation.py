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
    *,
    slot: int = 0,
    symbol: int = 0,
    group_hopping: bool = False,
    sequence_hopping: bool = False,
    K_TC: int = 2,
    srs_num_rb: int | None = None,
) -> np.ndarray:
    """Generate SRS pilots for an interfering UE per 3GPP TS 38.211 §6.4.1.4.

    Uses the standard SRS sequence with ``n_SRS_ID = cell_id`` and per-UE
    cyclic shift ``n_cs = ue_index % 8`` to model intra-cell orthogonality
    and inter-cell interference randomisation.
    Returns shape ``[effective_rb]`` complex128.
    """
    from msg_embedding.ref_signals.srs import srs_sequence

    effective_rb = srs_num_rb if srs_num_rb is not None else num_rb
    if effective_rb < 1:
        return np.ones(max(num_rb, 1), dtype=np.complex128)

    Msc = max(effective_rb * 12 // K_TC, 6)
    n_SRS_ID = cell_id % 1024
    n_cs = ue_index % (K_TC * 4)  # cyclic shift range per K_TC

    seq = srs_sequence(
        n_SRS_ID=n_SRS_ID,
        K_TC=K_TC,
        n_cs=n_cs,
        N_ap=1,
        Msc=Msc,
        slot=slot,
        symbol=symbol,
        n_ap_index=0,
        group_hopping=group_hopping,
        sequence_hopping=sequence_hopping,
    )
    sc_per_rb = 12 // K_TC
    if sc_per_rb > 1 and len(seq) > effective_rb:
        seq = seq[::sc_per_rb][:effective_rb]
    if len(seq) < effective_rb:
        pad = np.ones(effective_rb - len(seq), dtype=np.complex128)
        seq = np.concatenate([seq, pad])
    else:
        seq = seq[:effective_rb]
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



def _build_pilot_grid(
    h_true: np.ndarray,
    RB: int,
    T: int,
    valid_symbol_mask: np.ndarray | None = None,
    srs_rb_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Compute pilot RS grid positions matching NR reference signal density.

    If ``srs_rb_indices`` is provided, pilots are placed only on those RBs
    (SRS frequency hopping). Otherwise, pilots cover the full band.

    If ``valid_symbol_mask`` is provided (TDD-aware), pilots are only placed
    on symbols where the mask is True.

    Returns (rs_freq, rs_time, n_freq, n_time).
    """
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
    valid_symbol_mask: np.ndarray | None = None,
    srs_rb_indices: np.ndarray | None = None,
    srs_slot: int = 0,
    srs_symbol: int = 0,
    srs_group_hopping: bool = False,
    srs_sequence_hopping: bool = False,
    srs_K_TC: int = 2,
    srs_num_rb: int | None = None,
    h_interferers_ul_per_ue: list[np.ndarray] | None = None,
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

    # -- Build pilot grid (NR RS density, TDD-aware, SRS freq-hopping) --
    rs_freq, rs_time, n_freq, n_time = _build_pilot_grid(
        h_serving_true, RB, T,
        valid_symbol_mask=valid_symbol_mask,
        srs_rb_indices=srs_rb_indices,
    )

    # -- Extract serving channel at pilot positions --
    # h_serving_true: [T, RB, Ant_tx, Ant_rx]
    h_at_pilots = h_serving_true[np.ix_(rs_time, rs_freq)]  # [n_time, n_freq, Ant_tx, Ant_rx]
    h_at_pilots = h_at_pilots.transpose(1, 0, 2, 3)  # [n_freq, n_time, Ant_tx, Ant_rx]

    # Serving pilot sequence at RS frequencies.
    # When srs_rb_indices is used, pilots_serving already has length == n_freq
    # (generated for SRS bandwidth only), so use it directly.
    if srs_rb_indices is not None and len(pilots_serving) == n_freq:
        pilot_seq = pilots_serving  # [n_freq]
    else:
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
            # UL: BS receives SRS from UEs in neighboring (inter-cell) cells.
            # Each interfering UE has its own independent channel when
            # h_interferers_ul_per_ue is provided; otherwise all UEs in a
            # cell share the cell-level channel (legacy fallback).
            # The number of active UEs per neighbor is randomized to model
            # scheduling uncertainty (0 to num_interfering_ues).
            _has_per_ue = (
                h_interferers_ul_per_ue is not None
                and len(h_interferers_ul_per_ue) >= K_minus_1
            )
            n_intf_total = 0
            for cell_k in range(K_minus_1):
                intf_cell_id = (
                    interferer_cell_ids[cell_k]
                    if interferer_cell_ids is not None and cell_k < len(interferer_cell_ids)
                    else serving_cell_id + cell_k + 1
                )

                if _has_per_ue:
                    n_ues_available = h_interferers_ul_per_ue[cell_k].shape[0]
                else:
                    n_ues_available = num_interfering_ues

                n_active_ues = int(rng.integers(0, n_ues_available + 1))
                active_ue_indices = (
                    rng.choice(n_ues_available, size=n_active_ues, replace=False)
                    if n_active_ues > 0
                    else np.array([], dtype=np.intp)
                )

                for ue_idx in active_ue_indices:
                    if _has_per_ue:
                        h_ue = h_interferers_ul_per_ue[cell_k][ue_idx]
                    else:
                        h_ue = h_interferers[cell_k]
                    h_ue_at_pilots = h_ue[np.ix_(rs_time, rs_freq)]
                    h_ue_at_pilots = h_ue_at_pilots.transpose(1, 0, 2, 3)

                    intf_pilots = _generate_interferer_pilots_srs(
                        RB, intf_cell_id, int(ue_idx),
                        slot=srs_slot,
                        symbol=srs_symbol,
                        group_hopping=srs_group_hopping,
                        sequence_hopping=srs_sequence_hopping,
                        K_TC=srs_K_TC,
                        srs_num_rb=srs_num_rb,
                    )
                    X_intf = intf_pilots[rs_freq][:, None, None, None]
                    interference_total += h_ue_at_pilots * X_intf
                    n_intf_total += 1

            logger.debug(
                "ul_interference_injected",
                num_interferer_cells=K_minus_1,
                max_ues_per_cell=num_interfering_ues,
                total_interfering_signals=n_intf_total,
            )

        else:
            # DL: UE receives CSI-RS from neighboring (inter-cell) cells.
            # Intra-cell CSI-RS ports are orthogonal (CDM/FDM/TDM);
            # interference comes from neighbors whose CSI-RS collides in
            # time-freq. A random subset of neighbors is active per sample.
            n_active_cells = int(rng.integers(0, K_minus_1 + 1))
            active_cell_indices = (
                rng.choice(K_minus_1, size=n_active_cells, replace=False)
                if n_active_cells > 0
                else np.array([], dtype=np.intp)
            )

            for cell_k in active_cell_indices:
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
                num_interferer_cells_available=K_minus_1,
                num_active_cells=n_active_cells,
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
    srs_group_hopping: bool = False,
    srs_sequence_hopping: bool = False,
    srs_comb: int = 2,
    slot_idx: int = 0,
    h_interferers_ul_per_ue: list[np.ndarray] | None = None,
    dl_symbol_mask: np.ndarray | None = None,
    ul_symbol_mask: np.ndarray | None = None,
    srs_rb_indices: np.ndarray | None = None,
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

    # SRS pilot on first available UL symbol
    _ul_syms = np.where(ul_symbol_mask)[0] if ul_symbol_mask is not None else [0]
    _srs_symbol = int(_ul_syms[-1]) if len(_ul_syms) > 0 else 0
    _srs_num_rb = len(srs_rb_indices) if srs_rb_indices is not None else None
    pilots_srs = _generate_pilots_srs(
        num_rb, serving_cell_id,
        slot=slot_idx, symbol=_srs_symbol,
        group_hopping=srs_group_hopping,
        sequence_hopping=srs_sequence_hopping,
        K_TC=srs_comb,
        srs_num_rb=_srs_num_rb,
    )
    # CSI-RS pilot on first available DL symbol
    _dl_syms = np.where(dl_symbol_mask)[0] if dl_symbol_mask is not None else [0]
    _csirs_symbol = int(_dl_syms[0]) if len(_dl_syms) > 0 else 0
    pilots_csirs = _generate_pilots_csirs(num_rb, serving_cell_id, slot=slot_idx, symbol=_csirs_symbol)

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
        valid_symbol_mask=ul_symbol_mask,
        srs_rb_indices=srs_rb_indices,
        srs_slot=slot_idx,
        srs_symbol=_srs_symbol,
        srs_group_hopping=srs_group_hopping,
        srs_sequence_hopping=srs_sequence_hopping,
        srs_K_TC=srs_comb,
        srs_num_rb=_srs_num_rb,
        h_interferers_ul_per_ue=h_interferers_ul_per_ue,
    )

    # -- DL estimation (UE receives CSI-RS with multi-cell interference) --
    # DL uses CSI-RS which covers full band — no srs_rb_indices
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
        valid_symbol_mask=dl_symbol_mask,
    )

    # Transpose UL back to contract shape [T, RB, BS_ant, UE_ant].
    # Internally UL is [T, RB, UE_ant, BS_ant] (physical), but contract
    # stores everything uniformly as [T, RB, BS_ant, UE_ant].
    h_ul_true_out = h_ul_true.transpose(0, 1, 3, 2).astype(np.complex64)
    h_ul_est_out = ul_result.h_est.transpose(0, 1, 3, 2).astype(np.complex64)

    return {
        "h_ul_true": h_ul_true_out,
        "h_ul_est": h_ul_est_out,
        "h_dl_true": h_dl_true.astype(np.complex64),
        "h_dl_est": dl_result.h_est,
        "ul_sir_dB": ul_result.sir_dB,
        "dl_sir_dB": dl_result.sir_dB,
        "num_interfering_ues": num_interfering_ues if h_interferers_ul is not None else None,
    }
