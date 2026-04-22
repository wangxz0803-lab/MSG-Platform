"""Tests for /api/topology endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_topology_preview(client: TestClient) -> None:
    resp = client.post(
        "/api/topology/preview",
        json={
            "num_sites": 7,
            "isd_m": 500.0,
            "sectors_per_site": 3,
            "tx_height_m": 25.0,
            "num_ues": 20,
            "ue_distribution": "uniform",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["sites"]) == 7 * 3
    assert len(data["ues"]) == 20
    assert data["cell_radius_m"] > 0
    assert "min_x" in data["bounds"]


def test_topology_preview_single_site(client: TestClient) -> None:
    resp = client.post(
        "/api/topology/preview",
        json={"num_sites": 1, "isd_m": 200.0, "num_ues": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["sites"]) == 3
    assert len(data["ues"]) == 5
