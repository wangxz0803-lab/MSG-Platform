"""Hexagonal cellular grid + 3-sector site layout.

Generates a honeycomb of base-station sites arranged around a central origin.
Each ring ``r`` adds ``6 * r`` sites, giving cumulative site counts
``1, 7, 19, 37, ...`` for ``num_rings = 0, 1, 2, 3, ...``. Every site can be
split into 1 (omni) or 3 sectors with boresight azimuths 0/120/240 degrees,
matching the 3GPP 38.901 macro layout convention.

The returned list is sorted deterministically by ``(site_id, sector_id)`` so
downstream PCI planning and pathloss evaluation are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["CellSite", "make_hex_grid", "hex_ring_positions"]


@dataclass
class CellSite:
    """A single cell (site + sector) in the topology.

    Attributes:
        site_id: Zero-based physical tower index.
        position: XYZ position in metres, shape ``(3,)`` float64.
        sector_id: Sector index within the site (``0`` for omni, ``0..2`` for
            3-sector).
        azimuth_deg: Boresight azimuth in degrees measured clockwise from +x.
        tx_height_m: Antenna height in metres (equals ``position[2]``).
        pci: Physical Cell Identity (0..1007). ``-1`` means "unassigned".
        scenario: 3GPP scenario tag, e.g. ``'UMa_NLOS'`` / ``'UMi_LOS'`` /
            ``'InF'``.
    """

    site_id: int
    position: np.ndarray = field(repr=False)
    sector_id: int
    azimuth_deg: float
    tx_height_m: float
    pci: int = -1
    scenario: str = "UMa_NLOS"

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)


# ------------------------------------------------------------------
# Hexagonal lattice geometry
# ------------------------------------------------------------------

# Unit-distance hex-ring direction vectors (axial -> cartesian), CCW starting
# from +x. For ring ``r``, visit each corner and walk ``r`` steps toward the
# next corner. Distances are in units of the inter-site distance (ISD).
_HEX_CORNERS = np.array(
    [
        [1.0, 0.0],
        [0.5, np.sqrt(3) / 2],
        [-0.5, np.sqrt(3) / 2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3) / 2],
        [0.5, -np.sqrt(3) / 2],
    ],
    dtype=np.float64,
)


def hex_ring_positions(num_rings: int, isd_m: float) -> np.ndarray:
    """Return the 2D XY positions of the hex-grid sites.

    Args:
        num_rings: Number of rings around the centre. ``0`` -> only origin,
            ``1`` -> 7 sites, ``2`` -> 19 sites, etc.
        isd_m: Inter-site distance in metres (distance between adjacent sites).

    Returns:
        Float64 array of shape ``(N, 2)`` with the deterministic site order
        ``[centre, ring1_site0, ring1_site1, ..., ring2_site0, ...]``.
    """
    if num_rings < 0:
        raise ValueError(f"num_rings must be >= 0, got {num_rings}")
    if isd_m <= 0:
        raise ValueError(f"isd_m must be > 0, got {isd_m}")

    positions: list[np.ndarray] = [np.zeros(2, dtype=np.float64)]
    for r in range(1, num_rings + 1):
        # Start at corner 4 (lower-left) and walk toward corner 0, so the first
        # visited site of each ring is directly right of the centre for r=1.
        # Standard hex-ring traversal: for each of the 6 edges, take r steps.
        start = _HEX_CORNERS[4] * r  # start corner
        for edge in range(6):
            step_dir = _HEX_CORNERS[edge]
            for step in range(r):
                positions.append((start + step_dir * step) * isd_m)
            # Advance to the next corner.
            start = start + step_dir * r
    return np.asarray(positions, dtype=np.float64)


# ------------------------------------------------------------------
# Public builder
# ------------------------------------------------------------------


def make_hex_grid(
    num_rings: int,
    isd_m: float,
    sectors: int = 3,
    tx_height_m: float = 25.0,
    scenario: str = "UMa_NLOS",
) -> list[CellSite]:
    """Build a list of :class:`CellSite` on a hexagonal grid.

    Args:
        num_rings: Number of hex rings (``0`` -> 1 site, ``1`` -> 7 sites,
            ``2`` -> 19 sites).
        isd_m: Inter-site distance in metres. Typical values: macro 500,
            dense 200, micro 50.
        sectors: ``1`` for omni, ``3`` for tri-sector sites with azimuths
            ``0 / 120 / 240`` degrees.
        tx_height_m: Antenna height above ground (z-coordinate).
        scenario: 3GPP scenario tag stored on each returned :class:`CellSite`.

    Returns:
        List of :class:`CellSite`, deterministically ordered by
        ``(site_id, sector_id)``. PCIs are left as ``-1`` — run a planner from
        :mod:`msg_embedding.topology.pci_planner` to populate them.
    """
    if sectors not in (1, 3):
        raise ValueError(f"sectors must be 1 or 3, got {sectors}")

    xy = hex_ring_positions(num_rings, isd_m)
    sites: list[CellSite] = []
    azimuths = (0.0,) if sectors == 1 else (0.0, 120.0, 240.0)

    for site_id, (x, y) in enumerate(xy):
        pos = np.array([x, y, tx_height_m], dtype=np.float64)
        for sector_id, az in enumerate(azimuths):
            sites.append(
                CellSite(
                    site_id=site_id,
                    position=pos.copy(),
                    sector_id=sector_id,
                    azimuth_deg=float(az),
                    tx_height_m=float(tx_height_m),
                    pci=-1,
                    scenario=scenario,
                )
            )

    return sites
