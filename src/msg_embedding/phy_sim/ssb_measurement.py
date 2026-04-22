"""Multi-cell SSB beam measurement pipeline per 3GPP TS 38.213/38.215.

Simulates the UE-side SSB measurement procedure:
1. Each cell transmits SSB blocks (PSS + SSS + PBCH-DMRS) using DFT beams
2. SSB signals propagate through the multi-cell channel H
3. UE receives the composite signal (serving + interferers + noise)
4. UE performs beam sweep, measures per-cell per-beam RSRP
5. Reports: per-cell best-beam RSRP, RSRQ, SS-SINR

SSB RE mapping per 38.211 §7.4.3:
  - SSB occupies 4 OFDM symbols × 240 subcarriers (20 RBs)
  - Symbol 0: PSS (56..182, 127 subcarriers, center of 240)
  - Symbol 1: PBCH-DMRS + PBCH data (full 240 SCs)
  - Symbol 2: SSS (56..182, 127 subcarriers)
  - Symbol 3: PBCH-DMRS + PBCH data (full 240 SCs)

Usage::

    meas = SSBMeasurement(num_beams=8, num_bs_ant=4)
    results = meas.measure(
        h_per_cell=[h_cell0, h_cell1, ...],  # each [T, RB, BS, UE]
        pcis=[0, 3, 6],
        noise_power_lin=1e-10,
    )
    # results.rsrp[k] = best-beam RSRP for cell k in dBm
    # results.rsrq[k] = RSRQ for cell k in dB
    # results.ss_sinr[k] = SS-SINR for cell k in dB
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SSBMeasurement",
    "SSBMeasurementResult",
    "generate_ssb_block",
]


@dataclass
class SSBMeasurementResult:
    """Per-cell SSB measurement results as reported by UE."""

    num_cells: int
    rsrp_dBm: np.ndarray  # [K] best-beam RSRP per cell
    rsrq_dB: np.ndarray  # [K] RSRQ per cell
    ss_sinr_dB: np.ndarray  # [K] SS-SINR per cell
    best_beam_idx: np.ndarray  # [K] best beam index per cell
    all_beam_rsrp: np.ndarray  # [K, num_beams] RSRP for all beams per cell
    pcis: list[int]  # PCI per cell

    def to_dict(self) -> dict:
        return {
            "num_cells": self.num_cells,
            "rsrp_dBm": self.rsrp_dBm.tolist(),
            "rsrq_dB": self.rsrq_dB.tolist(),
            "ss_sinr_dB": self.ss_sinr_dB.tolist(),
            "best_beam_idx": self.best_beam_idx.tolist(),
            "pcis": self.pcis,
        }


# SSB subcarrier mapping constants
SSB_NUM_SC = 240  # 20 RBs
PSS_SC_OFFSET = 56  # PSS starts at SC 56 within SSB band
PSS_LEN = 127
SSS_SC_OFFSET = 56
SSS_LEN = 127
SSB_SYMBOLS = 4  # SSB spans 4 OFDM symbols


def _generate_dft_beams(num_ant: int, num_beams: int) -> np.ndarray:
    """DFT codebook [num_ant, num_beams] for SSB beam sweep."""
    beams = np.zeros((num_ant, num_beams), dtype=np.complex128)
    for b in range(num_beams):
        theta = np.pi * (b / num_beams - 0.5)
        kd = 2.0 * np.pi * 0.5 * np.sin(theta)
        for a in range(num_ant):
            beams[a, b] = np.exp(1j * kd * a) / np.sqrt(num_ant)
    return beams


def generate_ssb_block(
    pci: int,
    i_ssb: int = 0,
    num_sc: int = SSB_NUM_SC,
) -> np.ndarray:
    """Generate one SSB block as a [4, num_sc] complex array.

    Maps PSS (symbol 0), PBCH+DMRS (symbol 1), SSS (symbol 2),
    PBCH+DMRS (symbol 3) onto their correct subcarrier positions.
    """
    from ..ref_signals.ssb import pbch_dmrs, pss, sss

    n_id_2 = pci % 3
    pci // 3

    block = np.zeros((SSB_SYMBOLS, num_sc), dtype=np.complex128)

    # Symbol 0: PSS
    pss_seq = pss(n_id_2)
    block[0, PSS_SC_OFFSET : PSS_SC_OFFSET + PSS_LEN] = pss_seq

    # Symbol 2: SSS
    sss_seq = sss(pci)
    block[2, SSS_SC_OFFSET : SSS_SC_OFFSET + SSS_LEN] = sss_seq

    # Symbols 1 & 3: PBCH-DMRS (every 4th SC per 38.211, 3 DMRS per RB)
    dmrs = pbch_dmrs(pci, i_ssb, length=num_sc // 4)
    for sym_idx in (1, 3):
        dmrs_pos = np.arange(0, num_sc, 4)
        block[sym_idx, dmrs_pos[: len(dmrs)]] = dmrs[: len(dmrs_pos)]
        # Fill non-DMRS positions with unit-power QPSK (PBCH data placeholder)
        data_pos = np.setdiff1d(np.arange(num_sc), dmrs_pos)
        rng = np.random.default_rng(pci * 100 + i_ssb * 10 + sym_idx)
        qpsk = (1.0 / np.sqrt(2)) * (
            (2 * rng.integers(0, 2, size=len(data_pos)) - 1)
            + 1j * (2 * rng.integers(0, 2, size=len(data_pos)) - 1)
        )
        block[sym_idx, data_pos] = qpsk

    return block


class SSBMeasurement:
    """Simulate UE-side SSB beam measurement across K cells.

    Parameters
    ----------
    num_beams : int
        Number of DFT beams for SSB beam sweep (typically 4, 8, or 16).
    num_bs_ant : int
        Number of BS antennas (must match channel H dimension).
    ref_power_offset_dBm : float
        Reference power offset for RSRP calculation.
    """

    def __init__(
        self,
        num_beams: int = 8,
        num_bs_ant: int = 4,
        ref_power_offset_dBm: float = -100.0,
    ):
        self.num_beams = num_beams
        self.num_bs_ant = num_bs_ant
        self.ref_power_offset = ref_power_offset_dBm
        self.beam_codebook = _generate_dft_beams(num_bs_ant, num_beams)

    @staticmethod
    def _ssb_block_to_rb(ssb_block: np.ndarray, n_rb: int) -> np.ndarray:
        """Down-convert SSB block from subcarrier to RB resolution.

        SSB spans 20 RBs (240 SCs).  The SSB band is placed starting at
        the center of the system bandwidth.  Returns [4, n_rb] with SSB
        power mapped to the correct RBs and zero elsewhere.
        """
        ssb_n_rb = SSB_NUM_SC // 12  # 20 RBs
        ssb_rb_start = max(0, (n_rb - ssb_n_rb) // 2)
        ssb_rb_end = min(n_rb, ssb_rb_start + ssb_n_rb)

        out = np.zeros((SSB_SYMBOLS, n_rb), dtype=np.complex128)
        for sym in range(SSB_SYMBOLS):
            for rb_idx in range(ssb_rb_start, ssb_rb_end):
                local_rb = rb_idx - ssb_rb_start
                sc_start = local_rb * 12
                sc_end = min(sc_start + 12, SSB_NUM_SC)
                if sc_start < SSB_NUM_SC:
                    avg_val = np.mean(ssb_block[sym, sc_start:sc_end])
                    out[sym, rb_idx] = avg_val
        return out

    def _apply_beam_and_channel(
        self,
        ssb_rb: np.ndarray,
        h_cell: np.ndarray,
        beam_idx: int,
    ) -> np.ndarray:
        """Transmit SSB through channel with a specific beam.

        ssb_rb: [4, n_rb] — SSB signal at RB resolution
        h_cell: [T, RB, BS, UE] — channel from this cell
        beam_idx: which DFT beam to use for transmission

        Returns: received signal [4, RB, UE]
        """
        beam = self.beam_codebook[:, beam_idx]  # [BS]
        _, rb, bs, ue = h_cell.shape

        t_avail = h_cell.shape[0]
        received = np.zeros((SSB_SYMBOLS, rb, ue), dtype=np.complex128)

        for sym in range(SSB_SYMBOLS):
            t_idx = sym % t_avail
            h_rb = h_cell[t_idx, :, :, :]  # [RB, BS, UE]
            h_eff = np.einsum("rbu,b->ru", h_rb, beam)  # [RB, UE]
            received[sym] = h_eff * ssb_rb[sym, :, np.newaxis]

        return received

    def _measure_rsrp_one_beam(
        self,
        received: np.ndarray,
        ssb_rb_mask: np.ndarray,
    ) -> float:
        """RSRP from received SSB signal (SS-RSRP per 38.215 §5.1.1).

        Measures average power over RBs that carry SSS (symbol 2).
        received: [4, RB, UE]
        ssb_rb_mask: [RB] boolean — which RBs carry SSB content
        """
        sss_signal = received[2, ssb_rb_mask, :]  # [active_RBs, UE]
        if sss_signal.size == 0:
            return -160.0
        rsrp_lin = np.mean(np.abs(sss_signal) ** 2)
        rsrp_dBm = 10.0 * np.log10(rsrp_lin + 1e-30) + self.ref_power_offset
        return float(rsrp_dBm)

    def measure(
        self,
        h_per_cell: list[np.ndarray],
        pcis: list[int],
        noise_power_lin: float = 1e-10,
        i_ssb: int = 0,
    ) -> SSBMeasurementResult:
        """Run full SSB measurement procedure.

        Parameters
        ----------
        h_per_cell : list of ndarray
            Channel matrix per cell, each [T, RB, BS, UE].
        pcis : list of int
            Physical Cell ID per cell.
        noise_power_lin : float
            Noise power in linear scale.
        i_ssb : int
            SSB candidate index.

        Returns
        -------
        SSBMeasurementResult with per-cell measurements.
        """
        K = len(h_per_cell)
        assert len(pcis) == K

        n_rb = h_per_cell[0].shape[1]

        # Generate SSB blocks per cell and convert to RB resolution
        ssb_blocks = [generate_ssb_block(pci, i_ssb) for pci in pcis]
        ssb_rb_blocks = [self._ssb_block_to_rb(blk, n_rb) for blk in ssb_blocks]

        # Mask of RBs carrying SSB content (SSS in symbol 2)
        ssb_n_rb = SSB_NUM_SC // 12
        ssb_rb_start = max(0, (n_rb - ssb_n_rb) // 2)
        ssb_rb_end = min(n_rb, ssb_rb_start + ssb_n_rb)
        ssb_rb_mask = np.zeros(n_rb, dtype=bool)
        ssb_rb_mask[ssb_rb_start:ssb_rb_end] = True

        # Beam sweep: measure RSRP for each (cell, beam) combination
        all_rsrp = np.full((K, self.num_beams), -160.0, dtype=np.float64)

        for k in range(K):
            h_k = h_per_cell[k]
            bs_ant = h_k.shape[2]
            if bs_ant != self.num_bs_ant:
                codebook = _generate_dft_beams(bs_ant, self.num_beams)
            else:
                codebook = self.beam_codebook

            saved_codebook = self.beam_codebook
            self.beam_codebook = codebook

            for b in range(self.num_beams):
                received = self._apply_beam_and_channel(ssb_rb_blocks[k], h_k, b)
                all_rsrp[k, b] = self._measure_rsrp_one_beam(received, ssb_rb_mask)

            self.beam_codebook = saved_codebook

        # Best beam per cell
        best_beam = np.argmax(all_rsrp, axis=1)
        best_rsrp = np.array([all_rsrp[k, best_beam[k]] for k in range(K)])

        # RSRQ per cell (38.215 §5.1.3):
        # RSRQ = N × RSRP / RSSI, where RSSI includes all cells + noise
        # N = number of RBs used for RSSI measurement
        n_rb = max(1, min(h_per_cell[0].shape[1], SSB_NUM_SC // 12))
        # Total received power (all cells) at best beam of serving cell
        total_rssi_lin = np.zeros(K, dtype=np.float64)
        for k in range(K):
            rsrp_lin = 10.0 ** ((best_rsrp[k] - self.ref_power_offset) / 10.0)
            total_rssi_lin[k] = rsrp_lin
        total_power = total_rssi_lin.sum() + noise_power_lin * n_rb
        rsrq_dB = np.zeros(K, dtype=np.float64)
        for k in range(K):
            rsrp_lin = 10.0 ** ((best_rsrp[k] - self.ref_power_offset) / 10.0)
            rsrq_lin = n_rb * rsrp_lin / (total_power + 1e-30)
            rsrq_dB[k] = 10.0 * np.log10(rsrq_lin + 1e-30)

        # SS-SINR per cell (38.215 §5.1.25):
        # SS-SINR = RSRP_serving / (interference + noise)
        ss_sinr_dB = np.zeros(K, dtype=np.float64)
        for k in range(K):
            rsrp_lin_k = 10.0 ** ((best_rsrp[k] - self.ref_power_offset) / 10.0)
            interference = total_power - rsrp_lin_k
            if interference <= 0:
                interference = noise_power_lin * n_rb
            sinr_lin = rsrp_lin_k / (interference + 1e-30)
            ss_sinr_dB[k] = float(np.clip(10.0 * np.log10(sinr_lin + 1e-30), -20.0, 40.0))

        return SSBMeasurementResult(
            num_cells=K,
            rsrp_dBm=np.clip(best_rsrp, -160.0, -40.0).astype(np.float32),
            rsrq_dB=np.clip(rsrq_dB, -40.0, 0.0).astype(np.float32),
            ss_sinr_dB=ss_sinr_dB.astype(np.float32),
            best_beam_idx=best_beam.astype(np.int32),
            all_beam_rsrp=all_rsrp.astype(np.float32),
            pcis=pcis,
        )
