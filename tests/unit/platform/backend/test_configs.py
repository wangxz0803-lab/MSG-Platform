"""Tests for /api/configs endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_get_schema(client: TestClient) -> None:
    resp = client.get("/api/configs/schema")
    assert resp.status_code == 200
    data = resp.json()
    assert "type" in data


def test_get_defaults_invalid_section(client: TestClient) -> None:
    resp = client.get("/api/configs/defaults", params={"section": "bogus"})
    assert resp.status_code == 400
