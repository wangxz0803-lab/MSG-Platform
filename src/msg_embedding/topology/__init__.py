"""5G network topology primitives: hex grids, PCI planning, and scenarios.

Public API
----------
* :class:`CellSite`, :func:`make_hex_grid` — honeycomb site layouts.
* :func:`assign_pci_mod3`, :func:`assign_pci_graph_coloring`,
  :class:`PciPlanResult` — PCI planners.
* :class:`Scenario` + the four ``scenario_*`` builders — preset deployments.
"""

from __future__ import annotations

from .hex_grid import CellSite, hex_ring_positions, make_hex_grid
from .pci_planner import (
    PciPlanResult,
    assign_pci_graph_coloring,
    assign_pci_mod3,
)
from .scenarios import (
    Scenario,
    scenario_indoor_factory,
    scenario_macro_19site_3sector,
    scenario_micro_dense,
    scenario_sionna_munich_osm,
)

__all__ = [
    "CellSite",
    "PciPlanResult",
    "Scenario",
    "assign_pci_graph_coloring",
    "assign_pci_mod3",
    "hex_ring_positions",
    "make_hex_grid",
    "scenario_indoor_factory",
    "scenario_macro_19site_3sector",
    "scenario_micro_dense",
    "scenario_sionna_munich_osm",
]
