"""3GPP 38.901 Clustered Delay Line (CDL) channel model profiles.

Tables reproduced from TS 38.901 v17.1.0 Tables 7.7.1-1 through 7.7.1-5.
Each profile defines a set of clusters with normalised delays (to be scaled
by the scenario's RMS delay spread), relative powers in dB, and angular
parameters (AoD, AoA, ZoD, ZoA) in degrees.

CDL-A, CDL-B, CDL-C: NLOS profiles (Rayleigh fading on all clusters).
CDL-D, CDL-E: LOS profiles (first cluster is Rician with K-factor).

Usage::

    profile = get_cdl_profile("CDL-A")
    delays_s = profile.delays_seconds(tau_rms_s)
    powers_lin = profile.powers_linear()
    aod = profile.aod_rad()
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CDLProfile",
    "get_cdl_profile",
    "list_cdl_profiles",
    "CDL_A",
    "CDL_B",
    "CDL_C",
    "CDL_D",
    "CDL_E",
]


@dataclass(frozen=True)
class CDLProfile:
    """A single CDL channel model profile."""

    name: str
    delays_norm: np.ndarray  # normalised delays (multiply by tau_rms)
    powers_dB: np.ndarray  # relative power per cluster in dB
    aod_deg: np.ndarray  # azimuth of departure per cluster in degrees
    aoa_deg: np.ndarray  # azimuth of arrival per cluster in degrees
    zod_deg: np.ndarray  # zenith of departure per cluster in degrees
    zoa_deg: np.ndarray  # zenith of arrival per cluster in degrees
    k_factor_dB: float | None = None  # Rician K-factor for LOS cluster (CDL-D/E only)
    los_tap_index: int = 0  # which cluster is the LOS component
    is_los: bool = False

    @property
    def num_clusters(self) -> int:
        return len(self.delays_norm)

    def delays_seconds(self, tau_rms_s: float) -> np.ndarray:
        return self.delays_norm * tau_rms_s

    def powers_linear(self) -> np.ndarray:
        return 10.0 ** (self.powers_dB / 10.0)

    def powers_normalized(self) -> np.ndarray:
        """Powers normalised so that total power = 1."""
        p = self.powers_linear()
        return p / p.sum()

    def aod_rad(self) -> np.ndarray:
        return np.deg2rad(self.aod_deg)

    def aoa_rad(self) -> np.ndarray:
        return np.deg2rad(self.aoa_deg)

    def zod_rad(self) -> np.ndarray:
        return np.deg2rad(self.zod_deg)

    def zoa_rad(self) -> np.ndarray:
        return np.deg2rad(self.zoa_deg)


# TS 38.901 Table 7.7.1-1: CDL-A (NLOS, 23 clusters)
CDL_A = CDLProfile(
    name="CDL-A",
    delays_norm=np.array(
        [
            0.0000, 0.3819, 0.4025, 0.5868, 0.4610,
            0.5375, 0.6708, 0.5750, 0.7618, 1.5375,
            1.8978, 2.2242, 0.6932, 0.7643, 2.4566,
            1.2119, 0.8386, 1.2727, 0.9566, 1.0910,
            1.2820, 1.4424, 2.1573,
        ]
    ),
    powers_dB=np.array(
        [
            -13.4, 0.0, -2.2, -4.0, -6.0,
            -8.2, -9.8, -1.2, -7.5, -2.5,
            -7.5, -3.0, -8.6, -13.0, -3.5,
            -5.2, -12.0, -5.1, -7.5, -11.9,
            -12.3, -8.5, -8.4,
        ]
    ),
    aod_deg=np.array(
        [
            -178.1, -4.2, -4.2, -4.2, 90.2,
            90.2, 90.2, 121.5, 121.5, -81.7,
            -81.7, 158.4, -83.0, 134.8, -153.0,
            -172.0, -129.7, -146.9, 1.2, 73.0,
            -19.2, 31.5, -15.4,
        ]
    ),
    aoa_deg=np.array(
        [
            51.3, -152.7, -152.7, -152.7, 76.6,
            76.6, 76.6, -1.8, -1.8, -41.9,
            -41.9, 94.2, -157.0, -137.1, -91.1,
            -22.5, 155.0, -135.7, 11.7, -80.7,
            -143.1, -24.4, -30.1,
        ]
    ),
    zod_deg=np.array(
        [
            50.2, 93.2, 93.2, 93.2, 122.0,
            122.0, 122.0, 150.2, 150.2, 55.2,
            55.2, 26.4, 126.4, 171.6, 153.7,
            155.2, 12.0, 26.2, 171.0, 22.7,
            156.4, 139.0, 144.2,
        ]
    ),
    zoa_deg=np.array(
        [
            125.4, 91.3, 91.3, 91.3, 94.0,
            94.0, 94.0, 47.1, 47.1, 56.0,
            56.0, 30.1, 58.8, 26.0, 146.0,
            150.0, 36.0, 143.1, 32.7, 150.7,
            47.0, 55.9, 56.2,
        ]
    ),
    is_los=False,
)

# TS 38.901 Table 7.7.1-2: CDL-B (NLOS, 23 clusters)
CDL_B = CDLProfile(
    name="CDL-B",
    delays_norm=np.array(
        [
            0.0000, 0.1072, 0.2155, 0.2095, 0.2870,
            0.2986, 0.3752, 0.5055, 0.3681, 0.3697,
            0.5700, 0.5283, 1.1021, 1.2756, 1.5474,
            1.7842, 2.0169, 2.8294, 3.0219, 3.6413,
            4.1854, 4.2789, 4.7113,
        ]
    ),
    powers_dB=np.array(
        [
            0.0, -2.2, -4.0, -3.2, -9.8,
            -1.2, -3.4, -5.2, -7.6, -3.0,
            -8.9, -9.0, -4.8, -5.7, -7.5,
            -1.9, -7.6, -12.2, -9.8, -11.4,
            -14.9, -9.2, -11.3,
        ]
    ),
    aod_deg=np.array(
        [
            9.3, 9.3, 9.3, -34.1, 52.5,
            -14.5, -14.5, 60.4, -24.6, -24.6,
            -37.2, 27.2, 73.0, -64.5, 80.3,
            -70.3, -72.5, -9.3, -60.8, 10.7,
            -1.0, 30.4, 0.3,
        ]
    ),
    aoa_deg=np.array(
        [
            -173.3, -173.3, -173.3, 125.5, -155.5,
            -152.5, -152.5, -132.1, 167.7, 167.7,
            126.4, -140.0, -96.0, 150.5, 88.7,
            126.4, -91.4, -129.7, 163.9, 74.6,
            -129.4, -145.1, -173.1,
        ]
    ),
    zod_deg=np.array(
        [
            105.8, 105.8, 105.8, 115.3, 119.3,
            103.2, 103.2, 118.2, 102.0, 102.0,
            111.5, 98.3, 88.2, 91.2, 84.2,
            93.4, 84.7, 105.0, 90.5, 96.3,
            97.5, 94.2, 104.2,
        ]
    ),
    zoa_deg=np.array(
        [
            78.9, 78.9, 78.9, 63.8, 67.5,
            76.5, 76.5, 66.3, 81.3, 81.3,
            69.7, 84.0, 93.3, 88.6, 95.8,
            87.3, 95.3, 78.2, 89.3, 82.2,
            81.1, 85.4, 76.1,
        ]
    ),
    is_los=False,
)

# TS 38.901 Table 7.7.1-3: CDL-C (NLOS, 24 clusters)
CDL_C = CDLProfile(
    name="CDL-C",
    delays_norm=np.array(
        [
            0.0000, 0.2099, 0.2219, 0.2329, 0.2176,
            0.6366, 0.6448, 0.6560, 0.6584, 0.7935,
            0.8213, 0.9336, 1.2285, 1.3083, 2.1704,
            2.7105, 4.2589, 4.6003, 5.4902, 5.6077,
            6.3065, 6.6374, 7.0427, 8.6523,
        ]
    ),
    powers_dB=np.array(
        [
            -4.4, -1.2, -3.5, -5.2, -2.5,
            0.0, -2.2, -3.9, -7.4, -7.1,
            -10.7, -11.1, -5.1, -6.8, -8.7,
            -13.2, -13.9, -13.9, -15.8, -17.1,
            -16.0, -15.7, -21.6, -22.8,
        ]
    ),
    aod_deg=np.array(
        [
            -46.6, -22.8, -22.8, -22.8, -40.7,
            0.3, 0.3, 0.3, 73.1, -64.5,
            80.2, -70.3, -72.5, -9.3, -60.8,
            10.7, -1.0, 30.4, 0.3, -26.5,
            -21.9, 21.5, -47.5, -22.1,
        ]
    ),
    aoa_deg=np.array(
        [
            -101.0, 120.0, 120.0, 120.0, -127.5,
            170.4, 170.4, 170.4, 55.4, -136.3,
            -96.0, 150.5, 88.7, 126.4, -91.4,
            -129.7, 163.9, 74.6, -129.4, -145.1,
            -173.1, 87.2, 62.1, -168.1,
        ]
    ),
    zod_deg=np.array(
        [
            97.2, 98.6, 98.6, 98.6, 100.6,
            97.8, 97.8, 97.8, 105.2, 95.2,
            88.2, 91.2, 84.2, 93.4, 84.7,
            105.0, 90.5, 96.3, 97.5, 94.2,
            104.2, 93.5, 90.2, 95.8,
        ]
    ),
    zoa_deg=np.array(
        [
            87.6, 72.4, 72.4, 72.4, 75.2,
            67.4, 67.4, 67.4, 58.1, 69.1,
            93.3, 88.6, 95.8, 87.3, 95.3,
            78.2, 89.3, 82.2, 81.1, 85.4,
            76.1, 87.3, 92.8, 79.3,
        ]
    ),
    is_los=False,
)

# TS 38.901 Table 7.7.1-4: CDL-D (LOS, 13 clusters, K=13.3 dB)
CDL_D = CDLProfile(
    name="CDL-D",
    delays_norm=np.array(
        [
            0.0000, 0.0000, 0.0350, 0.6120, 1.6630,
            2.7380, 0.2040, 0.2440, 0.8850, 1.0130,
            1.0520, 1.7230, 2.0710,
        ]
    ),
    powers_dB=np.array(
        [
            -0.2, -13.5, -18.8, -21.0, -22.8,
            -17.9, -20.1, -21.9, -22.9, -18.5,
            -20.8, -22.6, -22.3,
        ]
    ),
    aod_deg=np.array(
        [
            0.0, 0.0, 89.2, 89.2, 89.2,
            13.0, -13.0, 13.0, -13.0, 13.0,
            -13.0, 36.4, -36.4,
        ]
    ),
    aoa_deg=np.array(
        [
            180.0, 180.0, 89.2, 89.2, 89.2,
            163.0, -163.0, 163.0, -163.0, 163.0,
            -163.0, -30.6, 30.6,
        ]
    ),
    zod_deg=np.array(
        [
            98.5, 98.5, 85.5, 85.5, 85.5,
            97.5, 97.5, 97.5, 97.5, 100.6,
            100.6, 98.8, 98.8,
        ]
    ),
    zoa_deg=np.array(
        [
            81.5, 81.5, 94.5, 94.5, 94.5,
            82.5, 82.5, 82.5, 82.5, 79.4,
            79.4, 81.2, 81.2,
        ]
    ),
    k_factor_dB=13.3,
    los_tap_index=0,
    is_los=True,
)

# TS 38.901 Table 7.7.1-5: CDL-E (LOS, 14 clusters, K=22.0 dB)
CDL_E = CDLProfile(
    name="CDL-E",
    delays_norm=np.array(
        [
            0.0000, 0.0000, 0.5133, 0.5440, 0.5630,
            0.5440, 0.7112, 1.9092, 1.9293, 1.9589,
            2.6426, 3.7136, 5.4524, 12.0034,
        ]
    ),
    powers_dB=np.array(
        [
            -0.03, -22.03, -15.8, -18.1, -19.8,
            -22.9, -22.4, -18.6, -20.8, -22.6,
            -22.3, -25.6, -20.2, -29.8,
        ]
    ),
    aod_deg=np.array(
        [
            0.0, 0.0, -31.5, -31.5, -31.5,
            45.0, -31.5, -15.3, -15.3, -15.3,
            54.3, -40.8, -3.2, 14.7,
        ]
    ),
    aoa_deg=np.array(
        [
            180.0, 180.0, -152.5, -152.5, -152.5,
            132.0, -152.5, -50.3, -50.3, -50.3,
            -130.2, 140.6, -4.0, -24.5,
        ]
    ),
    zod_deg=np.array(
        [
            99.6, 99.6, 104.2, 104.2, 104.2,
            100.6, 104.2, 98.2, 98.2, 98.2,
            101.5, 99.3, 103.3, 99.3,
        ]
    ),
    zoa_deg=np.array(
        [
            80.4, 80.4, 75.8, 75.8, 75.8,
            79.4, 75.8, 81.8, 81.8, 81.8,
            78.5, 80.7, 76.7, 80.7,
        ]
    ),
    k_factor_dB=22.0,
    los_tap_index=0,
    is_los=True,
)

_PROFILES: dict[str, CDLProfile] = {
    "CDL-A": CDL_A,
    "CDL-B": CDL_B,
    "CDL-C": CDL_C,
    "CDL-D": CDL_D,
    "CDL-E": CDL_E,
}


def get_cdl_profile(name: str) -> CDLProfile:
    """Look up a CDL profile by name (case-insensitive, with or without hyphen)."""
    key = name.upper().replace("_", "-")
    if key not in _PROFILES:
        raise ValueError(f"Unknown CDL profile {name!r}; available: {list(_PROFILES.keys())}")
    return _PROFILES[key]


def list_cdl_profiles() -> list[str]:
    return list(_PROFILES.keys())
