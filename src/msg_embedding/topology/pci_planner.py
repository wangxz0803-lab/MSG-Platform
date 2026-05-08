"""Physical Cell Identity (PCI) planning utilities.

Two strategies are provided:

* :func:`assign_pci_mod3` — fixed mapping ``PCI = 3 * (site_id mod 168) + sector_id``.
  Guarantees ``PCI mod 3 == sector_id`` on every site, which spreads the three
  PSS sequences evenly across sectors (a common engineering rule-of-thumb).

* :func:`assign_pci_graph_coloring` — Welsh–Powell greedy graph colouring
  against a physical-distance interference graph. Two sectors share an edge
  whenever their 3-D site distance is below ``interference_radius_m`` (or they
  share the same site). Adjacent sectors are guaranteed to receive *different*
  PCIs.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .hex_grid import CellSite

__all__ = [
    "assign_pci_mod3",
    "assign_pci_graph_coloring",
    "assign_pci_hypercell",
    "PciPlanResult",
]

# Total number of NR PCIs per 38.331.
_NUM_PCI_TOTAL = 1008
# Number of unique cell-ID groups (Nid1); PCI = 3 * Nid1 + Nid2.
_NUM_PCI_GROUPS = 336


# ------------------------------------------------------------------
# Mod-3 planner
# ------------------------------------------------------------------


def assign_pci_mod3(sites: list[CellSite]) -> list[CellSite]:
    """Assign PCIs via ``PCI = 3 * (site_id mod 168) + sector_id``.

    The formula guarantees:

    * Every sector of a given site gets a distinct PCI (they differ by
      ``sector_id``).
    * ``PCI mod 3 == sector_id``, avoiding PSS (Nid2) collisions between
      same-index sectors on different sites.

    Args:
        sites: Output of :func:`~msg_embedding.topology.hex_grid.make_hex_grid`.

    Returns:
        New list of :class:`CellSite` with ``pci`` populated; the input list
        is not mutated.
    """
    out: list[CellSite] = []
    for s in sites:
        if not 0 <= s.sector_id <= 2:
            raise ValueError(f"sector_id {s.sector_id} out of range [0, 2] on site {s.site_id}")
        pci = 3 * (s.site_id % 168) + s.sector_id
        if pci >= _NUM_PCI_TOTAL:
            raise ValueError(f"computed PCI {pci} exceeds {_NUM_PCI_TOTAL - 1}")
        out.append(replace(s, pci=int(pci)))
    return out


# ------------------------------------------------------------------
# Graph-colouring planner
# ------------------------------------------------------------------


class PciPlanResult(list[CellSite]):
    """A list of :class:`CellSite` plus plan metadata.

    Behaves like a normal ``list[CellSite]`` so callers that only need the
    sites can ignore the extra attributes.
    """

    num_used_colors: int = 0
    num_conflicts: int = 0


def _build_adjacency(sites: list[CellSite], interference_radius_m: float) -> list[list[int]]:
    """Return an adjacency list where sectors at the same site and sectors
    within ``interference_radius_m`` of each other are connected.
    """
    n = len(sites)
    positions = np.stack([s.position for s in sites], axis=0)
    site_ids = np.array([s.site_id for s in sites], dtype=np.int64)

    # Pairwise 3D distances (upper triangle only is enough).
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    same_site = site_ids[:, None] == site_ids[None, :]
    adj_mask = (dist < interference_radius_m) | same_site
    np.fill_diagonal(adj_mask, False)

    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        adj[i] = [int(j) for j in np.flatnonzero(adj_mask[i])]
    return adj


def assign_pci_graph_coloring(
    sites: list[CellSite],
    interference_radius_m: float = 1000.0,
    num_colors: int = 30,
) -> PciPlanResult:
    """Greedy Welsh–Powell PCI colouring.

    Builds an interference graph where sectors are connected iff they share a
    site or their 3-D distance is below ``interference_radius_m``. Sectors are
    then processed in descending order of degree and each is assigned the
    smallest *colour* index not used by any of its already-coloured neighbours.
    ``PCI`` is finally ``colour * 3 + sector_id`` so that the PSS property of
    :func:`assign_pci_mod3` is preserved while still satisfying the
    graph-colouring constraint.

    Args:
        sites: Output of :func:`~msg_embedding.topology.hex_grid.make_hex_grid`.
        interference_radius_m: Maximum 3-D distance (m) at which two sectors
            are still considered PCI-conflicting.
        num_colors: Upper bound on the number of colours. The planner raises
            ``RuntimeError`` if no proper colouring fits within this budget.

    Returns:
        A :class:`PciPlanResult` (``list[CellSite]`` subclass) whose
        ``num_used_colors`` holds the number of distinct colours actually used
        and ``num_conflicts`` holds the number of same-colour neighbour pairs
        (should always be 0 for a valid colouring).
    """
    if num_colors <= 0:
        raise ValueError(f"num_colors must be > 0, got {num_colors}")
    if interference_radius_m <= 0:
        raise ValueError(f"interference_radius_m must be > 0, got {interference_radius_m}")

    n = len(sites)
    if n == 0:
        return PciPlanResult()

    adj = _build_adjacency(sites, interference_radius_m)
    degrees = np.array([len(a) for a in adj], dtype=np.int64)
    # Welsh–Powell: process high-degree first; stable by original index.
    order = sorted(range(n), key=lambda i: (-int(degrees[i]), i))

    colours: list[int] = [-1] * n
    for idx in order:
        used = {colours[j] for j in adj[idx] if colours[j] >= 0}
        for c in range(num_colors):
            if c not in used:
                colours[idx] = c
                break
        if colours[idx] < 0:
            raise RuntimeError(
                f"could not colour sector {idx} within num_colors={num_colors}; "
                f"increase num_colors or interference_radius_m"
            )

    # Check: no adjacent sectors share a colour.
    conflicts = 0
    for i in range(n):
        for j in adj[i]:
            if j > i and colours[i] == colours[j]:
                conflicts += 1

    result = PciPlanResult()
    assert len(sites) == len(colours)
    for s, c in zip(sites, colours, strict=False):
        pci = c * 3 + s.sector_id
        if pci >= _NUM_PCI_TOTAL:
            # Fold back into valid PCI space while keeping Nid2 == sector_id.
            group = (c * 3 + s.sector_id) % _NUM_PCI_GROUPS
            pci = group * 3 + s.sector_id
        result.append(replace(s, pci=int(pci)))
    result.num_used_colors = len({c for c in colours if c >= 0})
    result.num_conflicts = conflicts
    return result


# ------------------------------------------------------------------
# HyperCell planner (HSR linear topology)
# ------------------------------------------------------------------


def assign_pci_hypercell(
    sites: list[CellSite],
    hypercell_size: int = 4,
) -> list[CellSite]:
    """Assign PCIs for HyperCell grouping along a linear track.

    Consecutive *hypercell_size* sites share the same PCI group so that
    a moving UE does not trigger handover within a HyperCell. Different
    HyperCell groups use different PCI groups.

    Formula: ``PCI = 3 * group_index + sector_id`` where
    ``group_index = site_id // hypercell_size``.

    Args:
        sites: Output of :func:`make_linear_grid` (sorted by site_id).
        hypercell_size: Number of consecutive RRH sites per HyperCell.
            ``1`` degenerates to standard per-site PCI assignment.
    """
    if hypercell_size < 1:
        raise ValueError(f"hypercell_size must be >= 1, got {hypercell_size}")
    out: list[CellSite] = []
    for s in sites:
        group_idx = s.site_id // hypercell_size
        pci = 3 * (group_idx % 168) + s.sector_id
        if pci >= _NUM_PCI_TOTAL:
            pci = 3 * (group_idx % _NUM_PCI_GROUPS) + s.sector_id
        out.append(replace(s, pci=int(pci)))
    return out
