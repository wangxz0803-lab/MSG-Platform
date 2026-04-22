"""Sounding Reference Signal (SRS) per 3GPP TS 38.211 §6.4.1.4.

Covers:
    * Transmission combs K_TC ∈ {2, 4, 8} (§6.4.1.4.2).
    * Cyclic shift α_i for antenna ports 1000 + i (up to 8 shifts per comb).
    * C_SRS / B_SRS bandwidth configuration table (Table 6.4.1.4.3-1, a small
      subset is hard-coded — the full table is ~64 rows and not required for
      sequence generation itself).
    * Sequence-group / sequence hopping (§6.4.1.4.2) driven by the NR Gold PRBS.
    * Base-sequence selection: long (ZC) for Msc ≥ 36, short (CG tables) for
      Msc ∈ {6, 12, 18, 24}.

The main API ``srs_sequence`` returns the complex ``r̃_{u,v}^{α}(n)`` of
length ``Msc`` with the per-port cyclic shift already applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gold import pseudo_random
from .zc import _next_prime_below, r_uv_long, r_uv_short

__all__ = [
    "SRSBandwidthRow",
    "SRS_BW_TABLE",
    "SRS_PERIODICITY_TABLE",
    "SRSResourceConfig",
    "srs_cyclic_shift",
    "srs_base_sequence",
    "srs_sequence",
    "srs_group_number",
    "srs_freq_position",
]


# -----------------------------------------------------------------------------
# 38.211 Table 6.4.1.4.3-1 — subset: (C_SRS, B_SRS=0..3 → m_SRS, N)
# Each row: (m_SRS_0, N_0, m_SRS_1, N_1, m_SRS_2, N_2, m_SRS_3, N_3).
# Only a handful of C_SRS rows are included here; the table is large but this
# subset is sufficient for unit-test coverage of the bandwidth tree.
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SRSBandwidthRow:
    """One row of 38.211 Table 6.4.1.4.3-1."""

    c_srs: int
    m_srs: tuple[int, int, int, int]
    n: tuple[int, int, int, int]


SRS_BW_TABLE: tuple[SRSBandwidthRow, ...] = (
    SRSBandwidthRow(0, (4, 4, 4, 4), (1, 1, 1, 1)),
    SRSBandwidthRow(1, (8, 4, 4, 4), (1, 2, 1, 1)),
    SRSBandwidthRow(2, (12, 4, 4, 4), (1, 3, 1, 1)),
    SRSBandwidthRow(3, (16, 4, 4, 4), (1, 4, 1, 1)),
    SRSBandwidthRow(4, (16, 8, 4, 4), (1, 2, 2, 1)),
    SRSBandwidthRow(5, (20, 4, 4, 4), (1, 5, 1, 1)),
    SRSBandwidthRow(6, (24, 4, 4, 4), (1, 6, 1, 1)),
    SRSBandwidthRow(7, (24, 12, 4, 4), (1, 2, 3, 1)),
    SRSBandwidthRow(8, (28, 4, 4, 4), (1, 7, 1, 1)),
    SRSBandwidthRow(9, (32, 16, 8, 4), (1, 2, 2, 2)),
    SRSBandwidthRow(10, (36, 12, 4, 4), (1, 3, 3, 1)),
    SRSBandwidthRow(11, (40, 20, 4, 4), (1, 2, 5, 1)),
    SRSBandwidthRow(12, (48, 16, 8, 4), (1, 3, 2, 2)),
    SRSBandwidthRow(13, (48, 24, 12, 4), (1, 2, 2, 3)),
    SRSBandwidthRow(14, (52, 4, 4, 4), (1, 13, 1, 1)),
    SRSBandwidthRow(15, (56, 28, 4, 4), (1, 2, 7, 1)),
    SRSBandwidthRow(16, (60, 20, 4, 4), (1, 3, 5, 1)),
    SRSBandwidthRow(17, (64, 32, 16, 4), (1, 2, 2, 4)),
)


def srs_cyclic_shift(n_cs: int, n_ap_index: int, N_ap: int, K_TC: int) -> float:
    """Compute α_i for port 1000 + i  per 38.211 §6.4.1.4.2 eq (6.4.1.4.2-2).

        α_i = 2·π · (n_cs + (n_cs^max · (p_i - 1000)) / N_ap) / n_cs^max  (mod 2π)

    where n_cs^max = 8 for K_TC ∈ {2, 4} and n_cs^max = 12 for K_TC = 8.
    """
    if K_TC not in (2, 4, 8):
        raise ValueError("K_TC must be in {2, 4, 8}")
    n_cs_max = 12 if K_TC == 8 else 8
    if not (0 <= n_cs < n_cs_max):
        raise ValueError(f"n_cs must be in [0, {n_cs_max})")
    if N_ap not in (1, 2, 4):
        raise ValueError("N_ap must be in {1, 2, 4}")
    if not (0 <= n_ap_index < N_ap):
        raise ValueError(f"n_ap_index must be in [0, {N_ap})")
    n_cs_i = (n_cs + (n_cs_max * n_ap_index) // N_ap) % n_cs_max
    return 2.0 * np.pi * n_cs_i / n_cs_max


def srs_group_number(
    n_SRS_ID: int, slot: int, symbol: int, group_hopping: bool, sequence_hopping: bool
) -> tuple[int, int]:
    """Compute (u, v) for SRS per 38.211 §6.4.1.4.2 / §5.2.2.

    * No hopping:  u = n_SRS_ID mod 30,  v = 0.
    * Group hopping: u = (f_gh + n_SRS_ID) mod 30 with f_gh from the PRBS.
    * Sequence hopping: v = c(n_s·N_symb_slot + l)  if Msc ≥ 72 (see caller).
    """
    if group_hopping and sequence_hopping:
        raise ValueError("group_hopping and sequence_hopping are mutually exclusive")

    if not group_hopping:
        f_gh = 0
    else:
        # f_gh = sum_{i=0..7} c(8·(N_symb_slot·n_s + l) + i) · 2^i   mod 30
        N_symb_slot = 14
        offset = 8 * (N_symb_slot * int(slot) + int(symbol))
        # Need at least offset + 8 PRBS bits; seed with n_SRS_ID / 30.
        c_init = (n_SRS_ID // 30) & 0x7FFFFFFF
        c = pseudo_random(c_init, offset + 8)
        f_gh = 0
        for i in range(8):
            f_gh = (f_gh + int(c[offset + i]) * (1 << i)) % 30
    u = (f_gh + n_SRS_ID) % 30

    if not sequence_hopping:
        v = 0
    else:
        N_symb_slot = 14
        c_init = n_SRS_ID & 0x7FFFFFFF
        idx = N_symb_slot * int(slot) + int(symbol)
        c = pseudo_random(c_init, idx + 1)
        v = int(c[idx])
    return u, v


def srs_base_sequence(u: int, v: int, Msc: int) -> np.ndarray:
    """Return r̄_{u,v}(n) of length ``Msc`` per 38.211 §5.2.2."""
    if Msc in (6, 12, 18, 24):
        return r_uv_short(u, v, Msc)
    if Msc < 36:
        raise ValueError(f"Msc {Msc} not supported (need Msc ≥ 36 for long sequence)")
    Nzc = _next_prime_below(Msc)
    r = r_uv_long(u, v, Nzc)
    if Msc == Nzc:
        return r
    # Cyclic extension to Msc: r̄(n) = x_q(n mod Nzc).
    n = np.arange(Msc)
    return r[n % Nzc]


def srs_sequence(
    n_SRS_ID: int,
    K_TC: int,
    n_cs: int,
    N_ap: int,
    Msc: int,
    slot: int,
    symbol: int,
    n_ap_index: int = 0,
    group_hopping: bool = False,
    sequence_hopping: bool = False,
    u_override: int | None = None,
    v_override: int | None = None,
) -> np.ndarray:
    """Return r̃_{u,v}^{α_i}(n) of length ``Msc`` per 38.211 §6.4.1.4.2.

    The output is

        r̃(n) = exp(j · α_i · n) · r̄_{u,v}(n),   n = 0..Msc-1

    with α_i the port-specific cyclic shift and (u, v) the SRS group/sequence
    indices from :func:`srs_group_number`.

    Parameters
    ----------
    n_SRS_ID:
        SRS scrambling identity (0..1023).
    K_TC:
        Transmission comb in {2, 4, 8}.
    n_cs:
        Cyclic shift index (0..7 for K_TC∈{2,4}, 0..11 for K_TC=8).
    N_ap:
        Number of SRS ports (1, 2 or 4).
    Msc:
        Sub-carrier count per symbol (must equal m_SRS·N_sc/K_TC for the
        configured bandwidth row — not enforced here).
    slot, symbol:
        Slot within the frame and OFDM symbol within the slot.
    n_ap_index:
        0..N_ap-1, picks the per-port cyclic shift.
    group_hopping, sequence_hopping:
        Mutually exclusive. Default: both off (ID hopping disabled).
    u_override, v_override:
        If provided, force (u, v) and skip the hopping computation (used for
        unit testing against known ZC / CG vectors).
    """
    if u_override is None or v_override is None:
        u, v = srs_group_number(n_SRS_ID, slot, symbol, group_hopping, sequence_hopping)
    else:
        u, v = int(u_override), int(v_override)
    r_bar = srs_base_sequence(u, v, Msc)
    alpha = srs_cyclic_shift(n_cs, n_ap_index, N_ap, K_TC)
    n = np.arange(Msc, dtype=np.int64)
    return (np.exp(1j * alpha * n) * r_bar).astype(np.complex128)


# ---------------------------------------------------------------------------
# 38.211 Table 6.4.1.4.4-1 — SRS periodicity and slot offset
# T_SRS (slots) and the valid offset range [0, T_SRS-1].
# ---------------------------------------------------------------------------
SRS_PERIODICITY_TABLE: tuple[int, ...] = (
    1,
    2,
    4,
    5,
    8,
    10,
    16,
    20,
    32,
    40,
    64,
    80,
    160,
    320,
    640,
    1280,
    2560,
)


@dataclass(frozen=True)
class SRSResourceConfig:
    """Configuration for one SRS resource per 38.211 §6.4.1.4.

    Captures the parameters needed for frequency-domain position (hopping)
    and time-domain periodicity.
    """

    C_SRS: int  # Bandwidth configuration index (row in Table 6.4.1.4.3-1)
    B_SRS: int  # Bandwidth index 0..3 (column level in the BW tree)
    K_TC: int  # Transmission comb {2, 4, 8}
    n_RRC: int  # RRC-configured frequency-domain starting position
    b_hop: int  # Frequency hopping parameter: ≥ B_SRS means no hopping
    n_SRS_ID: int  # SRS scrambling identity (0..1023)
    T_SRS: int  # Periodicity in slots (from SRS_PERIODICITY_TABLE)
    T_offset: int  # Slot offset within period [0, T_SRS-1]
    N_ap: int = 1  # Number of SRS antenna ports {1, 2, 4}
    n_cs: int = 0  # Cyclic shift index
    group_hopping: bool = False
    sequence_hopping: bool = False
    R: int = 1  # Repetition factor {1, 2, 4}

    def __post_init__(self):
        if self.T_SRS not in SRS_PERIODICITY_TABLE:
            raise ValueError(f"T_SRS={self.T_SRS} not in SRS_PERIODICITY_TABLE")
        if not (0 <= self.T_offset < self.T_SRS):
            raise ValueError(f"T_offset must be in [0, {self.T_SRS})")
        if self.K_TC not in (2, 4, 8):
            raise ValueError("K_TC must be in {2, 4, 8}")
        if self.b_hop > 3:
            raise ValueError("b_hop must be in [0, 3]")
        if self.B_SRS > 3:
            raise ValueError("B_SRS must be in [0, 3]")

    def is_slot_active(self, slot: int) -> bool:
        """Check if SRS is transmitted in this slot per the periodicity."""
        return (slot - self.T_offset) % self.T_SRS == 0

    @property
    def hopping_enabled(self) -> bool:
        return self.b_hop < self.B_SRS


def _get_bw_row(C_SRS: int) -> SRSBandwidthRow:
    """Look up bandwidth config row by C_SRS index."""
    for row in SRS_BW_TABLE:
        if row.c_srs == C_SRS:
            return row
    raise ValueError(f"C_SRS={C_SRS} not found in SRS_BW_TABLE (have 0..{SRS_BW_TABLE[-1].c_srs})")


def srs_freq_position(
    cfg: SRSResourceConfig,
    slot: int,
    symbol: int,
) -> tuple[int, int]:
    """Compute SRS frequency-domain starting position per 38.211 §6.4.1.4.3.

    Returns (n_b, k_0_bar) where:
      - n_b is the frequency position index at bandwidth level B_SRS
      - k_0_bar is the starting subcarrier (before comb offset)

    The frequency position determines which part of the bandwidth is used
    for the current SRS transmission, implementing frequency hopping when
    b_hop < B_SRS.

    Per 38.211 eq (6.4.1.4.3-1):
        k_0^(p_i) = k_0_bar + sum_{b=0}^{B_SRS} K_TC · m_SRS,b · n_b
    where k_0_bar = n_shift · N_sc^RB / 2 + k_TC_offset.
    """
    row = _get_bw_row(cfg.C_SRS)
    B_SRS = cfg.B_SRS
    b_hop = cfg.b_hop

    # n_shift (RRC configured starting position, mapped to subcarrier grid)
    N_sc_RB = 12
    k_0_bar = cfg.n_RRC * N_sc_RB

    if not cfg.hopping_enabled:
        # No hopping: n_b = n_RRC mod N_b for all b
        n_b_list = []
        for b in range(B_SRS + 1):
            N_b = row.n[b]
            n_b_list.append(cfg.n_RRC % max(1, N_b))
        # k_0 = k_0_bar + sum K_TC * m_SRS_b * n_b
        k_0 = k_0_bar
        for b in range(B_SRS + 1):
            k_0 += cfg.K_TC * row.m_srs[b] * n_b_list[b]
        return n_b_list[B_SRS] if n_b_list else 0, k_0
    else:
        # Frequency hopping per §6.4.1.4.3
        # N_symb_slot = 14, compute n_SRS counter for hopping
        N_symb_slot = 14
        if cfg.T_SRS > 0:
            n_SRS = ((slot - cfg.T_offset) // cfg.T_SRS) * cfg.R + (symbol // N_symb_slot)
        else:
            n_SRS = 0

        n_b_list = []
        for b in range(B_SRS + 1):
            N_b = row.n[b]
            if N_b <= 1:
                n_b_list.append(0)
                continue

            if b <= b_hop:
                # Below or at hopping level: fixed position
                n_b_list.append(cfg.n_RRC % N_b)
            else:
                # Above hopping level: hopping pattern
                # F_b(n_SRS) per 38.211 §6.4.1.4.3
                # Product of N values from b_hop+1 to b-1
                prod_N = 1
                for bp in range(b_hop + 1, b):
                    prod_N *= row.n[bp]

                if b == b_hop + 1:
                    F_b = (n_SRS // prod_N) % N_b
                else:
                    F_b_prev = 0
                    prod_N_prev = 1
                    for bp in range(b_hop + 1, b):
                        prod_N_prev *= row.n[bp]

                    F_b = ((n_SRS // prod_N) + F_b_prev) % N_b

                n_b_val = (F_b + cfg.n_RRC % N_b) % N_b
                n_b_list.append(n_b_val)

        k_0 = k_0_bar
        for b in range(B_SRS + 1):
            k_0 += cfg.K_TC * row.m_srs[b] * n_b_list[b]

        return n_b_list[B_SRS] if n_b_list else 0, k_0
