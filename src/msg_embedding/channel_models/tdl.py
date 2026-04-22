"""3GPP 38.901 Tapped Delay Line (TDL) channel model profiles.

Tables reproduced from TS 38.901 v17.1.0 Tables 7.7.2-1 through 7.7.2-5.
Each profile defines a set of taps with normalised delays (to be scaled by
the scenario's RMS delay spread) and relative powers in dB.

TDL-A, TDL-B, TDL-C: NLOS profiles (Rayleigh fading on all taps).
TDL-D, TDL-E: LOS profiles (first tap is Rician with K-factor).

Usage::

    profile = get_tdl_profile("TDL-C")
    delays_s = profile.delays_norm * tau_rms_s
    powers_lin = 10 ** (profile.powers_dB / 10)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "TDLProfile",
    "get_tdl_profile",
    "list_tdl_profiles",
    "TDL_A",
    "TDL_B",
    "TDL_C",
    "TDL_D",
    "TDL_E",
]


@dataclass(frozen=True)
class TDLProfile:
    """A single TDL channel model profile."""

    name: str
    delays_norm: np.ndarray  # normalised delays (multiply by tau_rms)
    powers_dB: np.ndarray  # relative power per tap in dB
    k_factor_dB: float | None = None  # Rician K-factor for LOS tap (TDL-D/E only)
    los_tap_index: int = 0  # which tap is the LOS component
    is_los: bool = False

    @property
    def num_taps(self) -> int:
        return len(self.delays_norm)

    def delays_seconds(self, tau_rms_s: float) -> np.ndarray:
        return self.delays_norm * tau_rms_s

    def powers_linear(self) -> np.ndarray:
        return 10.0 ** (self.powers_dB / 10.0)

    def powers_normalized(self) -> np.ndarray:
        """Powers normalised so that total power = 1."""
        p = self.powers_linear()
        return p / p.sum()


# TS 38.901 Table 7.7.2-1: TDL-A (NLOS, 23 taps)
TDL_A = TDLProfile(
    name="TDL-A",
    delays_norm=np.array(
        [
            0.0000,
            0.3819,
            0.4025,
            0.5868,
            0.4610,
            0.5375,
            0.6708,
            0.5750,
            0.7618,
            1.5375,
            1.8978,
            2.2242,
            2.1718,
            2.4942,
            2.5119,
            3.0582,
            4.0810,
            4.4579,
            4.5695,
            4.7966,
            5.0066,
            5.3043,
            9.6586,
        ]
    ),
    powers_dB=np.array(
        [
            -13.4,
            0.0,
            -2.2,
            -4.0,
            -6.0,
            -8.2,
            -9.9,
            -10.5,
            -7.5,
            -15.9,
            -6.6,
            -16.7,
            -12.4,
            -15.2,
            -10.8,
            -11.3,
            -12.7,
            -16.2,
            -18.3,
            -18.9,
            -16.6,
            -19.9,
            -29.7,
        ]
    ),
    is_los=False,
)

# TS 38.901 Table 7.7.2-2: TDL-B (NLOS, 23 taps)
TDL_B = TDLProfile(
    name="TDL-B",
    delays_norm=np.array(
        [
            0.0000,
            0.1072,
            0.2155,
            0.2095,
            0.2870,
            0.2986,
            0.3752,
            0.5055,
            0.3681,
            0.3697,
            0.5700,
            0.5283,
            1.1021,
            1.2756,
            1.5474,
            1.7842,
            2.0169,
            2.8294,
            3.0219,
            3.6187,
            4.1067,
            4.2790,
            4.7834,
        ]
    ),
    powers_dB=np.array(
        [
            0.0,
            -2.2,
            -4.0,
            -3.2,
            -9.8,
            -1.2,
            -3.4,
            -5.2,
            -7.6,
            -3.0,
            -8.9,
            -9.0,
            -4.8,
            -5.7,
            -7.5,
            -1.9,
            -7.6,
            -12.2,
            -9.8,
            -11.4,
            -14.9,
            -9.2,
            -11.3,
        ]
    ),
    is_los=False,
)

# TS 38.901 Table 7.7.2-3: TDL-C (NLOS, 24 taps)
TDL_C = TDLProfile(
    name="TDL-C",
    delays_norm=np.array(
        [
            0.0000,
            0.2099,
            0.2219,
            0.2329,
            0.2176,
            0.6366,
            0.6448,
            0.6560,
            0.6584,
            0.7935,
            0.8213,
            0.9336,
            1.2285,
            1.3083,
            2.1704,
            2.7105,
            4.2589,
            4.6003,
            5.4902,
            5.6077,
            6.3065,
            6.6374,
            7.0427,
            8.6523,
        ]
    ),
    powers_dB=np.array(
        [
            -4.4,
            -1.2,
            -3.5,
            -5.2,
            -2.5,
            0.0,
            -2.2,
            -3.9,
            -7.4,
            -7.1,
            -10.7,
            -11.1,
            -5.1,
            -6.8,
            -8.7,
            -13.2,
            -13.9,
            -13.9,
            -15.8,
            -17.1,
            -16.0,
            -15.7,
            -21.6,
            -22.8,
        ]
    ),
    is_los=False,
)

# TS 38.901 Table 7.7.2-4: TDL-D (LOS, 13 taps, K=13.3 dB)
TDL_D = TDLProfile(
    name="TDL-D",
    delays_norm=np.array(
        [
            0.0000,
            0.0350,
            0.6120,
            1.3630,
            1.4050,
            1.8040,
            2.5960,
            1.7750,
            4.0420,
            7.9370,
            9.4240,
            9.7080,
            12.5250,
        ]
    ),
    powers_dB=np.array(
        [
            -0.2,
            -13.5,
            -18.8,
            -21.0,
            -22.8,
            -17.9,
            -20.1,
            -21.9,
            -22.9,
            -27.8,
            -23.6,
            -24.8,
            -30.0,
        ]
    ),
    k_factor_dB=13.3,
    los_tap_index=0,
    is_los=True,
)

# TS 38.901 Table 7.7.2-5: TDL-E (LOS, 14 taps, K=22.0 dB)
TDL_E = TDLProfile(
    name="TDL-E",
    delays_norm=np.array(
        [
            0.0000,
            0.0317,
            0.2014,
            0.4986,
            0.5302,
            0.7236,
            0.8090,
            0.9009,
            1.2610,
            1.7698,
            2.5283,
            3.7925,
            5.0228,
            5.8668,
        ]
    ),
    powers_dB=np.array(
        [
            -0.03,
            -22.03,
            -15.8,
            -18.1,
            -19.8,
            -22.9,
            -22.4,
            -18.6,
            -20.8,
            -22.6,
            -22.3,
            -25.6,
            -20.2,
            -29.8,
        ]
    ),
    k_factor_dB=22.0,
    los_tap_index=0,
    is_los=True,
)

_PROFILES: dict[str, TDLProfile] = {
    "TDL-A": TDL_A,
    "TDL-B": TDL_B,
    "TDL-C": TDL_C,
    "TDL-D": TDL_D,
    "TDL-E": TDL_E,
}


def get_tdl_profile(name: str) -> TDLProfile:
    """Look up a TDL profile by name (case-insensitive, with or without hyphen)."""
    key = name.upper().replace("_", "-")
    if key not in _PROFILES:
        raise ValueError(f"Unknown TDL profile {name!r}; available: {list(_PROFILES.keys())}")
    return _PROFILES[key]


def list_tdl_profiles() -> list[str]:
    return list(_PROFILES.keys())
