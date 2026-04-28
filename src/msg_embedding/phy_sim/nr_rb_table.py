"""3GPP TS 38.101-1/2 Table 5.3.2-1: NR channel bandwidth and RB mapping.

Provides a lookup from (bandwidth_MHz, subcarrier_spacing_kHz) to the
standard number of resource blocks (N_RB).  Falls back to the formula
``floor(bandwidth / (12 * SCS))`` for non-standard combinations.
"""

from __future__ import annotations

__all__ = [
    "nr_rb_lookup",
    "nr_valid_bandwidths",
    "nr_valid_scs",
    "NR_RB_TABLE_FR1",
    "NR_RB_TABLE_FR2",
]

# FR1 (sub-7.125 GHz): TS 38.101-1 Table 5.3.2-1
# Key: (bandwidth_MHz, scs_kHz) → N_RB
NR_RB_TABLE_FR1: dict[tuple[int, int], int] = {
    # SCS 15 kHz
    (5, 15): 25,
    (10, 15): 52,
    (15, 15): 79,
    (20, 15): 106,
    (25, 15): 133,
    (30, 15): 160,
    (40, 15): 216,
    (50, 15): 270,
    # SCS 30 kHz
    (5, 30): 11,
    (10, 30): 24,
    (15, 30): 38,
    (20, 30): 51,
    (25, 30): 65,
    (30, 30): 78,
    (40, 30): 106,
    (50, 30): 133,
    (60, 30): 162,
    (70, 30): 189,
    (80, 30): 217,
    (90, 30): 245,
    (100, 30): 273,
    # SCS 60 kHz
    (10, 60): 11,
    (15, 60): 18,
    (20, 60): 24,
    (25, 60): 31,
    (30, 60): 38,
    (40, 60): 51,
    (50, 60): 65,
    (60, 60): 79,
    (70, 60): 93,
    (80, 60): 107,
    (90, 60): 121,
    (100, 60): 135,
}

# FR2 (24.25–52.6 GHz): TS 38.101-2 Table 5.3.2-1
NR_RB_TABLE_FR2: dict[tuple[int, int], int] = {
    # SCS 60 kHz
    (50, 60): 66,
    (100, 60): 132,
    (200, 60): 264,
    # SCS 120 kHz
    (50, 120): 32,
    (100, 120): 66,
    (200, 120): 132,
    (400, 120): 264,
}

_ALL_TABLES = {**NR_RB_TABLE_FR1, **NR_RB_TABLE_FR2}


def nr_rb_lookup(bandwidth_hz: float, scs_hz: float) -> int:
    """Return the standard NR RB count for the given bandwidth and SCS.

    Tries exact match in the 3GPP table first.  If the combination is
    non-standard, falls back to ``floor(bandwidth / (12 * SCS))`` and
    clamps to [1, 275] (FR1 max).
    """
    bw_mhz = round(bandwidth_hz / 1e6)
    scs_khz = round(scs_hz / 1e3)

    n_rb = _ALL_TABLES.get((bw_mhz, scs_khz))
    if n_rb is not None:
        return n_rb

    # Fallback for non-standard combos (e.g. custom bandwidths)
    n_rb_calc = max(1, int(bandwidth_hz / (12 * scs_hz)))
    return min(n_rb_calc, 275)


def nr_valid_bandwidths(scs_khz: int) -> list[int]:
    """Return sorted list of valid NR bandwidths (MHz) for a given SCS."""
    bws = sorted({bw for (bw, scs) in _ALL_TABLES if scs == scs_khz})
    return bws


def nr_valid_scs() -> list[int]:
    """Return all SCS values (kHz) present in the NR tables."""
    return sorted({scs for (_, scs) in _ALL_TABLES})
