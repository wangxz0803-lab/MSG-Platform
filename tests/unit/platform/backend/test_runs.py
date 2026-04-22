"""Tests for /api/runs endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_list_runs_empty(client: TestClient) -> None:
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 0
    assert isinstance(data["items"], list)


def test_compare_runs_empty_ids_400(client: TestClient) -> None:
    resp = client.get("/api/runs/compare", params={"ids": ""})
    assert resp.status_code == 400
