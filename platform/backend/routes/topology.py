"""Topology preview endpoint -- generates cell site + UE positions."""

from __future__ import annotations

import math
import random

from fastapi import APIRouter

from ..schemas.topology import (
    SitePosition,
    TopologyPreviewRequest,
    TopologyPreviewResponse,
    UEPosition,
)

router = APIRouter(prefix="/api/topology", tags=["topology"])


def _hex_grid_positions(
    num_sites: int, isd_m: float, sectors: int, tx_height: float
) -> list[SitePosition]:
    """Generate hexagonal grid site positions."""
    rings_map = {1: 0, 7: 1, 19: 2, 37: 3, 57: 3}
    num_rings = 0
    for sites, rings in sorted(rings_map.items()):
        if num_sites <= sites:
            num_rings = rings
            break
    else:
        num_rings = 3

    result: list[SitePosition] = []

    center_positions: list[tuple[float, float]] = [(0.0, 0.0)]

    for ring in range(1, num_rings + 1):
        for i in range(6):
            angle1 = math.pi / 3 * i + math.pi / 6
            for step in range(ring):
                angle2 = math.pi / 3 * ((i + 2) % 6) + math.pi / 6
                x = ring * isd_m * math.cos(angle1) + step * isd_m * math.cos(angle2)
                y = ring * isd_m * math.sin(angle1) + step * isd_m * math.sin(angle2)
                center_positions.append((x, y))

    center_positions = center_positions[:num_sites]

    pci = 0
    for site_id, (cx, cy) in enumerate(center_positions):
        for s in range(sectors):
            azimuth = s * (360.0 / sectors)
            result.append(
                SitePosition(
                    site_id=site_id,
                    x=cx,
                    y=cy,
                    z=tx_height,
                    sector_id=s,
                    azimuth_deg=azimuth,
                    pci=pci,
                )
            )
            pci += 1

    return result


def _linear_grid_positions(
    num_sites: int, isd_m: float, sectors: int, tx_height: float,
    hypercell_size: int = 1, track_offset_m: float = 80.0,
) -> list[SitePosition]:
    """Generate linear (track-side) site positions for HSR preview.

    Sites alternate on both sides of the track (y = ±track_offset_m),
    spaced by *isd_m* along the x-axis.
    """
    total_len = (num_sites - 1) * isd_m
    start_x = -total_len / 2

    result: list[SitePosition] = []
    pci = 0
    for site_id in range(num_sites):
        cx = start_x + site_id * isd_m
        side = 1.0 if site_id % 2 == 0 else -1.0
        cy = side * track_offset_m
        toward_track = 270.0 if side > 0 else 90.0
        group_idx = site_id // hypercell_size if hypercell_size > 1 else site_id
        for s in range(sectors):
            azimuth = toward_track + s * (360.0 / sectors)
            cell_pci = 3 * (group_idx % 168) + s if hypercell_size > 1 else pci
            result.append(
                SitePosition(
                    site_id=site_id,
                    x=cx,
                    y=cy,
                    z=tx_height,
                    sector_id=s,
                    azimuth_deg=azimuth % 360,
                    pci=cell_pci,
                )
            )
            pci += 1

    return result


def _generate_ues_track(
    num_ues: int,
    sites: list[SitePosition],
    train_length_m: float = 400.0,
    train_width_m: float = 3.4,
) -> list[UEPosition]:
    """Place UEs inside a train on the track centerline (y≈0).

    The train is a ~400m × 3.4m rectangle centered in the coverage area.
    UEs scatter inside with slight random offsets to mimic seat positions.
    """
    xs = sorted({s.x for s in sites})
    if not xs:
        return []
    track_center_x = (xs[0] + xs[-1]) / 2
    seed = hash((num_ues, len(xs), train_length_m)) & 0xFFFFFFFF
    rng = random.Random(seed)

    half_len = train_length_m / 2
    half_wid = train_width_m / 2

    ues: list[UEPosition] = []
    for i in range(num_ues):
        cx = track_center_x + rng.uniform(-half_len, half_len)
        cy = rng.uniform(-half_wid, half_wid)
        ues.append(UEPosition(ue_id=i, x=cx, y=cy))
    return ues


def _generate_ues(
    num_ues: int,
    sites: list[SitePosition],
    cell_radius: float,
    distribution: str,
) -> list[UEPosition]:
    """Generate UE positions based on distribution type."""
    ues: list[UEPosition] = []

    centers = sorted({(s.x, s.y) for s in sites})

    if not centers:
        return ues

    seed = hash((num_ues, len(centers), cell_radius, distribution)) & 0xFFFFFFFF
    rng = random.Random(seed)

    all_x = [c[0] for c in centers]
    all_y = [c[1] for c in centers]
    spread = cell_radius * 1.5

    for i in range(num_ues):
        if distribution == "uniform":
            cx = rng.uniform(min(all_x) - spread, max(all_x) + spread)
            cy = rng.uniform(min(all_y) - spread, max(all_y) + spread)
        elif distribution == "clustered":
            center = rng.choice(centers)
            angle = rng.uniform(0, 2 * math.pi)
            r = rng.gauss(0, cell_radius * 0.3)
            cx = center[0] + abs(r) * math.cos(angle)
            cy = center[1] + abs(r) * math.sin(angle)
        else:
            if rng.random() < 0.7:
                center = centers[0]
                angle = rng.uniform(0, 2 * math.pi)
                r = rng.gauss(0, cell_radius * 0.4)
                cx = center[0] + abs(r) * math.cos(angle)
                cy = center[1] + abs(r) * math.sin(angle)
            else:
                center = rng.choice(centers)
                angle = rng.uniform(0, 2 * math.pi)
                r = rng.uniform(0, cell_radius)
                cx = center[0] + r * math.cos(angle)
                cy = center[1] + r * math.sin(angle)

        ues.append(UEPosition(ue_id=i, x=cx, y=cy))

    return ues


@router.post("/preview", response_model=TopologyPreviewResponse)
def preview_topology(req: TopologyPreviewRequest) -> TopologyPreviewResponse:
    """Compute cell-site and UE positions for visualization."""
    cell_radius = req.isd_m / math.sqrt(3)

    if req.topology_layout == "linear":
        sites = _linear_grid_positions(
            req.num_sites, req.isd_m, req.sectors_per_site, req.tx_height_m,
            req.hypercell_size, req.track_offset_m,
        )
    else:
        sites = _hex_grid_positions(
            req.num_sites, req.isd_m, req.sectors_per_site, req.tx_height_m
        )
    if req.topology_layout == "linear":
        ues = _generate_ues_track(req.num_ues, sites)
    else:
        ues = _generate_ues(req.num_ues, sites, cell_radius, req.ue_distribution)

    all_x = [s.x for s in sites] + [u.x for u in ues]
    all_y = [s.y for s in sites] + [u.y for u in ues]

    margin = cell_radius
    bx_min = min(all_x) - margin if all_x else -500
    bx_max = max(all_x) + margin if all_x else 500
    by_min = min(all_y) - margin if all_y else -500
    by_max = max(all_y) + margin if all_y else 500

    if req.topology_layout == "linear":
        x_span = bx_max - bx_min
        y_span = by_max - by_min
        min_y_span = x_span / 3.0
        if y_span < min_y_span:
            cy = (by_min + by_max) / 2
            by_min = cy - min_y_span / 2
            by_max = cy + min_y_span / 2

    bounds = {
        "min_x": bx_min, "max_x": bx_max,
        "min_y": by_min, "max_y": by_max,
    }

    return TopologyPreviewResponse(
        sites=sites,
        ues=ues,
        cell_radius_m=cell_radius,
        bounds=bounds,
    )
