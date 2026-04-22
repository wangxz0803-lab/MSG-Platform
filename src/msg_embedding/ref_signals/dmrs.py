"""PDSCH/PUSCH DMRS generation per 3GPP TS 38.211 §7.4.1.1 / §6.4.1.1.

Implements the QPSK DMRS sequence used by both PDSCH (§7.4.1.1.1) and
CP-OFDM-based PUSCH (§6.4.1.1.1.1):

    r(m) = (1/√2)·[1 - 2·c(2m)] + j·(1/√2)·[1 - 2·c(2m+1)]

with a shared Gold PRBS ``c(·)`` seeded from the slot, symbol, scrambling ID and
DMRS scrambling bit ``n_SCID``. Both DMRS Type 1 (Δf = 2, 6 freq-domain pairs
per RB) and Type 2 (Δf = 3, 4 freq-domain pairs per RB) are supported, along
with the CDM-group / port maps (Tables 7.4.1.1.2-1 / 7.4.1.1.2-2).

The optional ``additional_symbols`` parameter allows repeating the sequence for
scenarios with 2 / 3 / 4 DMRS symbols per slot (mapping type A front-loaded +
additional DMRS per 38.211 Table 7.4.1.1.2-3/4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gold import pseudo_random

__all__ = [
    "DMRSConfig",
    "dmrs_sequence",
    "dmrs_re_map",
    "TYPE1_CDM_GROUPS",
    "TYPE2_CDM_GROUPS",
    "TYPE1_WF",
    "TYPE1_WT",
    "TYPE2_WF",
    "TYPE2_WT",
]

# -----------------------------------------------------------------------------
# DMRS Type 1 — Table 7.4.1.1.2-1
#   CDM group λ ∈ {0, 1}; Δ = λ; w_f from [+1,+1]/[+1,-1]; w_t from [+1,+1]/[+1,-1].
# Port 1000..1007 mapped as:
#   p=1000: λ=0, w_f=[+1,+1], w_t=[+1,+1]
#   p=1001: λ=0, w_f=[+1,-1], w_t=[+1,+1]
#   p=1002: λ=1, w_f=[+1,+1], w_t=[+1,+1]
#   p=1003: λ=1, w_f=[+1,-1], w_t=[+1,+1]
#   p=1004: λ=0, w_f=[+1,+1], w_t=[+1,-1]
#   p=1005: λ=0, w_f=[+1,-1], w_t=[+1,-1]
#   p=1006: λ=1, w_f=[+1,+1], w_t=[+1,-1]
#   p=1007: λ=1, w_f=[+1,-1], w_t=[+1,-1]
# -----------------------------------------------------------------------------
TYPE1_CDM_GROUPS: dict[int, int] = {
    1000: 0,
    1001: 0,
    1002: 1,
    1003: 1,
    1004: 0,
    1005: 0,
    1006: 1,
    1007: 1,
}
TYPE1_WF: dict[int, tuple[int, int]] = {
    1000: (1, 1),
    1001: (1, -1),
    1002: (1, 1),
    1003: (1, -1),
    1004: (1, 1),
    1005: (1, -1),
    1006: (1, 1),
    1007: (1, -1),
}
TYPE1_WT: dict[int, tuple[int, int]] = {
    1000: (1, 1),
    1001: (1, 1),
    1002: (1, 1),
    1003: (1, 1),
    1004: (1, -1),
    1005: (1, -1),
    1006: (1, -1),
    1007: (1, -1),
}

# -----------------------------------------------------------------------------
# DMRS Type 2 — Table 7.4.1.1.2-2
#   CDM group λ ∈ {0, 1, 2}; Δ = 2λ.
# Port 1000..1011 mapped as:
#   p=1000/1001: λ=0, w_f=[+1,+1]/[+1,-1], w_t=[+1,+1]
#   p=1002/1003: λ=1, w_f=[+1,+1]/[+1,-1], w_t=[+1,+1]
#   p=1004/1005: λ=2, w_f=[+1,+1]/[+1,-1], w_t=[+1,+1]
#   p=1006..1011: same as 1000..1005 but w_t=[+1,-1]
# -----------------------------------------------------------------------------
TYPE2_CDM_GROUPS: dict[int, int] = {
    1000: 0,
    1001: 0,
    1002: 1,
    1003: 1,
    1004: 2,
    1005: 2,
    1006: 0,
    1007: 0,
    1008: 1,
    1009: 1,
    1010: 2,
    1011: 2,
}
TYPE2_WF: dict[int, tuple[int, int]] = {
    1000: (1, 1),
    1001: (1, -1),
    1002: (1, 1),
    1003: (1, -1),
    1004: (1, 1),
    1005: (1, -1),
    1006: (1, 1),
    1007: (1, -1),
    1008: (1, 1),
    1009: (1, -1),
    1010: (1, 1),
    1011: (1, -1),
}
TYPE2_WT: dict[int, tuple[int, int]] = {
    1000: (1, 1),
    1001: (1, 1),
    1002: (1, 1),
    1003: (1, 1),
    1004: (1, 1),
    1005: (1, 1),
    1006: (1, -1),
    1007: (1, -1),
    1008: (1, -1),
    1009: (1, -1),
    1010: (1, -1),
    1011: (1, -1),
}


@dataclass
class DMRSConfig:
    """Configuration block for PDSCH/PUSCH DMRS generation.

    Attributes
    ----------
    dmrs_type:
        1 or 2 (defines comb spacing and CDM-group count).
    n_scid:
        Scrambling ID bit in {0, 1} (controls which of the two SCIDs is used).
    n_id:
        31-bit scrambling identifier (N_ID^0_SCID or N_ID^1_SCID).
    n_rb:
        Number of PRBs in the allocation.
    slot:
        Slot index within the frame (0..N_slot_frame-1).
    symbol:
        OFDM symbol index carrying the DMRS (0..13 for normal CP).
    n_scid_bit:
        Alias for ``n_scid`` — kept for readability at call sites.
    """

    dmrs_type: int = 1
    n_scid: int = 0
    n_id: int = 0
    n_rb: int = 1
    slot: int = 0
    symbol: int = 2
    n_scid_bit: int | None = None

    def __post_init__(self) -> None:
        if self.dmrs_type not in (1, 2):
            raise ValueError("dmrs_type must be 1 or 2")
        if self.n_scid not in (0, 1):
            raise ValueError("n_scid must be 0 or 1")
        if not (0 <= self.n_id < (1 << 16)):
            raise ValueError("n_id must fit in 16 bits per 38.211 §7.4.1.1.1")
        if self.n_rb <= 0:
            raise ValueError("n_rb must be positive")
        if self.slot < 0 or self.symbol < 0:
            raise ValueError("slot/symbol must be non-negative")
        if self.n_scid_bit is None:
            self.n_scid_bit = self.n_scid


def _dmrs_c_init(slot: int, symbol: int, n_id: int, n_scid: int) -> int:
    """38.211 §7.4.1.1.1:

    c_init = (2^17·(N_symb_slot·n_s,f^μ + l + 1)·(2·N_ID + 1)
              + 2^17·floor(n_SCID/2) + 2·N_ID + n_SCID) mod 2^31
    """
    N_symb_slot = 14  # Normal CP.
    term1 = (1 << 17) * (N_symb_slot * int(slot) + int(symbol) + 1) * (2 * int(n_id) + 1)
    term2 = (1 << 17) * (int(n_scid) // 2)
    term3 = 2 * int(n_id) + int(n_scid)
    return (term1 + term2 + term3) & 0x7FFFFFFF


def dmrs_sequence(config: DMRSConfig, port: int = 1000) -> np.ndarray:
    """Return complex-symbol DMRS r(m) weighted by the port's w_f cover code.

    The number of DMRS tones per RB is 6 for Type 1 and 4 for Type 2, giving
    ``n_dmrs = 6·n_rb`` (Type 1) or ``4·n_rb`` (Type 2) output symbols.

    Parameters
    ----------
    config:
        :class:`DMRSConfig` instance.
    port:
        Antenna port index (1000..1007 for Type 1, 1000..1011 for Type 2).
    """
    if config.dmrs_type == 1:
        if port not in TYPE1_CDM_GROUPS:
            raise ValueError(f"port {port} not valid for DMRS Type 1")
        w_f = TYPE1_WF[port]
        n_sc_per_rb = 6
    else:
        if port not in TYPE2_CDM_GROUPS:
            raise ValueError(f"port {port} not valid for DMRS Type 2")
        w_f = TYPE2_WF[port]
        n_sc_per_rb = 4
    n_dmrs = n_sc_per_rb * config.n_rb

    c_init = _dmrs_c_init(config.slot, config.symbol, config.n_id, int(config.n_scid_bit))
    c = pseudo_random(c_init, 2 * n_dmrs)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    re = inv_sqrt2 * (1.0 - 2.0 * c[0::2].astype(np.float64))
    im = inv_sqrt2 * (1.0 - 2.0 * c[1::2].astype(np.float64))
    r = (re + 1j * im).astype(np.complex128)

    # Apply w_f cover code: alternate +1, w_f[1] across the two REs of each pair.
    # The ``k'`` index of 38.211 alternates 0/1 within a CDM pair in frequency.
    cover = np.empty(n_dmrs, dtype=np.float64)
    cover[0::2] = w_f[0]
    cover[1::2] = w_f[1]
    return r * cover


def dmrs_re_map(config: DMRSConfig, port: int = 1000) -> np.ndarray:
    """Return the sub-carrier indices k (within the allocation) carrying DMRS.

    Type 1: k = 4·n + 2·k' + Δ,  k' ∈ {0,1},  n = 0..3·n_rb-1  — total 6·n_rb REs
    Type 2: k = 6·n + k' + Δ,    k' ∈ {0,1},  n = 0..2·n_rb-1  — total 4·n_rb REs

    where Δ = λ (Type 1) or Δ = 2λ (Type 2) and λ is the CDM group of ``port``.
    """
    if config.dmrs_type == 1:
        lam = TYPE1_CDM_GROUPS[port]
        delta = lam
        n = np.arange(3 * config.n_rb)
        ks = np.concatenate([4 * n + 0 + delta, 4 * n + 2 + delta])
    else:
        lam = TYPE2_CDM_GROUPS[port]
        delta = 2 * lam
        n = np.arange(2 * config.n_rb)
        ks = np.concatenate([6 * n + 0 + delta, 6 * n + 1 + delta])
    return np.sort(ks).astype(np.int64)
