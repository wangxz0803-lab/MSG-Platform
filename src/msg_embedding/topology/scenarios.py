"""Pre-configured deployment scenarios.

Thin wrappers around :func:`~msg_embedding.topology.hex_grid.make_hex_grid`
and :func:`~msg_embedding.topology.pci_planner.assign_pci_mod3` that produce
ready-to-use :class:`Scenario` bundles. Each scenario sets a sensible carrier
frequency and picks a 38.901 pathloss model tag so downstream channel
simulators can branch on ``pathloss_model``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .hex_grid import CellSite, make_hex_grid
from .pci_planner import assign_pci_mod3

__all__ = [
    "Scenario",
    "scenario_macro_19site_3sector",
    "scenario_micro_dense",
    "scenario_indoor_factory",
    "scenario_sionna_munich_osm",
]


@dataclass
class Scenario:
    """A named deployment snapshot.

    Attributes:
        name: Human-readable identifier.
        sites: List of :class:`CellSite` with PCIs assigned (may be empty for
            stub scenarios, e.g. the Munich OSM placeholder).
        cell_radius_m: Nominal cell radius (typically ``isd_m / sqrt(3)`` for
            hex grids).
        carrier_freq_hz: Carrier frequency in Hz.
        pathloss_model: 38.901 pathloss model tag, e.g.
            ``'38.901-UMa-NLOS'``.
        meta: Free-form metadata dict for scenario-specific fields
            (OSM paths, site types, simulator hints, ...).
    """

    name: str
    sites: list[CellSite]
    cell_radius_m: float
    carrier_freq_hz: float
    pathloss_model: str
    meta: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Macro: canonical 19-site tri-sector UMa layout
# ------------------------------------------------------------------


def scenario_macro_19site_3sector() -> Scenario:
    """3GPP 38.901 urban-macro reference: 19 tri-sector sites, ISD 500 m.

    Matches the "Case 1" macro-cell layout used throughout 38.901 / 38.913.
    Tx height is 25 m, carrier is 3.5 GHz, NLOS pathloss is assumed by default.
    """
    isd = 500.0
    sites = make_hex_grid(
        num_rings=2,
        isd_m=isd,
        sectors=3,
        tx_height_m=25.0,
        scenario="UMa_NLOS",
    )
    sites = assign_pci_mod3(sites)
    return Scenario(
        name="macro_19site_3sector",
        sites=sites,
        cell_radius_m=isd / 3**0.5,
        carrier_freq_hz=3.5e9,
        pathloss_model="38.901-UMa-NLOS",
        meta={"isd_m": isd, "num_rings": 2, "site_type": "macro"},
    )


# ------------------------------------------------------------------
# Micro: dense urban, 7 tri-sector sites
# ------------------------------------------------------------------


def scenario_micro_dense() -> Scenario:
    """Dense urban micro layout: 7 tri-sector sites, ISD 200 m, 3.5 GHz."""
    isd = 200.0
    sites = make_hex_grid(
        num_rings=1,
        isd_m=isd,
        sectors=3,
        tx_height_m=10.0,
        scenario="UMi_NLOS",
    )
    sites = assign_pci_mod3(sites)
    return Scenario(
        name="micro_dense",
        sites=sites,
        cell_radius_m=isd / 3**0.5,
        carrier_freq_hz=3.5e9,
        pathloss_model="38.901-UMi-NLOS",
        meta={"isd_m": isd, "num_rings": 1, "site_type": "micro"},
    )


# ------------------------------------------------------------------
# Indoor Factory — 3GPP 38.901 §7.2 InF topology
# ------------------------------------------------------------------


def scenario_indoor_factory() -> Scenario:
    """Indoor-factory layout (3GPP 38.901 §7.2 InF).

    Uses a 2x2 regular grid of ceiling-mounted omni APs (4 sites, ISD 50 m)
    inside a 100x100 m hall. Tx height is 8 m (InF-SH subscenario), carrier
    is 2 GHz. ``scenario`` is tagged ``'InF'`` so the pathloss engine can
    pick the correct 38.901 InF sub-variant (DH/SH/DL/SL) from
    ``meta['sub_scenario']``.
    """
    isd = 50.0
    # 2x2 regular grid centred on the origin.
    positions: list[tuple[float, float]] = [
        (-isd / 2, -isd / 2),
        (isd / 2, -isd / 2),
        (-isd / 2, isd / 2),
        (isd / 2, isd / 2),
    ]
    sites: list[CellSite] = []
    for site_id, (x, y) in enumerate(positions):
        sites.append(
            CellSite(
                site_id=site_id,
                position=np.array([x, y, 8.0], dtype=np.float64),
                sector_id=0,
                azimuth_deg=0.0,
                tx_height_m=8.0,
                pci=-1,
                scenario="InF",
            )
        )
    sites = assign_pci_mod3(sites)
    return Scenario(
        name="indoor_factory",
        sites=sites,
        cell_radius_m=isd,
        carrier_freq_hz=2.0e9,
        pathloss_model="38.901-InF-SH",
        meta={
            "isd_m": isd,
            "hall_size_m": (100.0, 100.0, 10.0),
            "sub_scenario": "InF-SH",  # sparse clutter, high BS
            "site_type": "indoor_ap",
        },
    )


# ------------------------------------------------------------------
# Sionna / Munich OSM stub (Phase 1.5)
# ------------------------------------------------------------------


def scenario_sionna_munich_osm() -> Scenario:
    """Placeholder scenario for the Munich OSM ray-tracing map.

    The actual site list will be produced by the Sionna ray-tracer in
    Phase 1.5 once OSM ingestion lands. For now this returns an empty
    ``sites`` list and stores the OSM path in ``meta`` so callers can
    detect the stub.
    """
    return Scenario(
        name="sionna_munich_osm",
        sites=[],
        cell_radius_m=0.0,
        carrier_freq_hz=3.5e9,
        pathloss_model="sionna-rt",
        meta={
            "osm_path": "configs/osm/munich.xml",
            "note": "Phase 1.5 接入",
            "site_type": "rt_stub",
        },
    )
