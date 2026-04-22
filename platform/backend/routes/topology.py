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

    sites = _hex_grid_positions(
        req.num_sites, req.isd_m, req.sectors_per_site, req.tx_height_m
    )
    ues = _generate_ues(req.num_ues, sites, cell_radius, req.ue_distribution)

    all_x = [s.x for s in sites] + [u.x for u in ues]
    all_y = [s.y for s in sites] + [u.y for u in ues]

    margin = cell_radius
    bounds = {
        "min_x": min(all_x) - margin if all_x else -500,
        "max_x": max(all_x) + margin if all_x else 500,
        "min_y": min(all_y) - margin if all_y else -500,
        "max_y": max(all_y) + margin if all_y else 500,
    }

    return TopologyPreviewResponse(
        sites=sites,
        ues=ues,
        cell_radius_m=cell_radius,
        bounds=bounds,
    )
