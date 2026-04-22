"""5G NR TDD slot pattern configuration per 3GPP TS 38.213 §11.1.

Each slot in a TDD frame is designated as:
  D = Downlink, U = Uplink, S = Special (flexible/guard).

The special slot is further divided into DL symbols, guard period, and UL
symbols (e.g., 10D+2G+2U within 14 OFDM symbols).

Usage::

    pattern = get_tdd_pattern("DDDSU")
    for slot_idx in range(pattern.period_slots):
        direction = pattern.slot_type(slot_idx)  # 'D', 'U', or 'S'
        sym_map = pattern.symbol_map(slot_idx)    # per-symbol 'D'/'U'/'G' list
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "TDDPattern",
    "get_tdd_pattern",
    "list_tdd_patterns",
    "STANDARD_PATTERNS",
]

SlotType = Literal["D", "U", "S"]
SymbolType = Literal["D", "U", "G"]


@dataclass(frozen=True)
class SpecialSlotConfig:
    """Symbol-level split of a special slot (14 OFDM symbols total)."""

    dl_symbols: int = 10
    guard_symbols: int = 2
    ul_symbols: int = 2

    def __post_init__(self) -> None:
        total = self.dl_symbols + self.guard_symbols + self.ul_symbols
        if total != 14:
            raise ValueError(
                f"Special slot symbols must sum to 14, got {total} "
                f"({self.dl_symbols}D+{self.guard_symbols}G+{self.ul_symbols}U)"
            )

    def symbol_map(self) -> list[SymbolType]:
        return ["D"] * self.dl_symbols + ["G"] * self.guard_symbols + ["U"] * self.ul_symbols


@dataclass(frozen=True)
class TDDPattern:
    """A TDD slot pattern over one period."""

    name: str
    slots: str  # e.g., "DDDSU" — each char is D/U/S
    special: SpecialSlotConfig = field(default_factory=SpecialSlotConfig)
    periodicity_ms: float = 5.0  # pattern repetition period
    subcarrier_spacing_kHz: int = 30  # determines slots per subframe

    @property
    def period_slots(self) -> int:
        return len(self.slots)

    @property
    def num_dl_slots(self) -> int:
        return self.slots.count("D")

    @property
    def num_ul_slots(self) -> int:
        return self.slots.count("U")

    @property
    def num_special_slots(self) -> int:
        return self.slots.count("S")

    @property
    def dl_ul_ratio(self) -> str:
        """Human-readable DL:UL ratio including special slot contributions."""
        dl_syms = self.num_dl_slots * 14
        ul_syms = self.num_ul_slots * 14
        for c in self.slots:
            if c == "S":
                dl_syms += self.special.dl_symbols
                ul_syms += self.special.ul_symbols
        return f"{dl_syms}:{ul_syms}"

    def slot_type(self, slot_idx: int) -> SlotType:
        return self.slots[slot_idx % self.period_slots]  # type: ignore[return-value]

    def symbol_map(self, slot_idx: int) -> list[SymbolType]:
        """Per-OFDM-symbol direction map for a given slot index."""
        st = self.slot_type(slot_idx)
        if st == "D":
            return ["D"] * 14
        elif st == "U":
            return ["U"] * 14
        else:
            return self.special.symbol_map()

    def is_dl_symbol(self, slot_idx: int, symbol_idx: int) -> bool:
        return self.symbol_map(slot_idx)[symbol_idx] == "D"

    def is_ul_symbol(self, slot_idx: int, symbol_idx: int) -> bool:
        return self.symbol_map(slot_idx)[symbol_idx] == "U"

    def dl_symbol_count(self) -> int:
        """Total DL OFDM symbols per pattern period."""
        return sum(1 for s in range(self.period_slots) for sym in self.symbol_map(s) if sym == "D")

    def ul_symbol_count(self) -> int:
        """Total UL OFDM symbols per pattern period."""
        return sum(1 for s in range(self.period_slots) for sym in self.symbol_map(s) if sym == "U")


# Standard 5G NR TDD patterns used in practice
STANDARD_PATTERNS: dict[str, TDDPattern] = {
    "DDDSU": TDDPattern(
        name="DDDSU",
        slots="DDDSU",
        special=SpecialSlotConfig(dl_symbols=10, guard_symbols=2, ul_symbols=2),
        periodicity_ms=5.0,
    ),
    "DDSUU": TDDPattern(
        name="DDSUU",
        slots="DDSUU",
        special=SpecialSlotConfig(dl_symbols=10, guard_symbols=2, ul_symbols=2),
        periodicity_ms=5.0,
    ),
    "DDDDDDDSUU": TDDPattern(
        name="DDDDDDDSUU",
        slots="DDDDDDDSUU",
        special=SpecialSlotConfig(dl_symbols=6, guard_symbols=4, ul_symbols=4),
        periodicity_ms=10.0,
    ),
    "DDDSUDDSUU": TDDPattern(
        name="DDDSUDDSUU",
        slots="DDDSUDDSUU",
        special=SpecialSlotConfig(dl_symbols=10, guard_symbols=2, ul_symbols=2),
        periodicity_ms=10.0,
    ),
    "DSUUD": TDDPattern(
        name="DSUUD",
        slots="DSUUD",
        special=SpecialSlotConfig(dl_symbols=6, guard_symbols=4, ul_symbols=4),
        periodicity_ms=5.0,
    ),
    "DDDDDDDDD_UL": TDDPattern(
        name="all_DL",
        slots="DDDDD",
        periodicity_ms=5.0,
    ),
    "UUUUUUUUU_DL": TDDPattern(
        name="all_UL",
        slots="UUUUU",
        periodicity_ms=5.0,
    ),
}


def get_tdd_pattern(name: str) -> TDDPattern:
    """Look up a TDD pattern by name."""
    if name in STANDARD_PATTERNS:
        return STANDARD_PATTERNS[name]
    # Allow custom pattern strings like "DDDSU"
    name_upper = name.upper()
    if all(c in "DUS" for c in name_upper):
        return TDDPattern(name=name_upper, slots=name_upper)
    raise ValueError(
        f"Unknown TDD pattern {name!r}; available: {list(STANDARD_PATTERNS.keys())} "
        "or pass a custom string of D/U/S characters."
    )


def list_tdd_patterns() -> list[str]:
    return list(STANDARD_PATTERNS.keys())
