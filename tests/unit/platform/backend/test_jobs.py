"""Tests for /api/jobs endpoints."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient


def test_create_job(client: TestClient, tmp_layout) -> None:
    resp = client.post(
        "/api/jobs",
        json={
            "type": "train",
            "config_overrides": {"train.batch_size": 32},
            "display_name": "smoke-train",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["type"] == "train"
    assert data["status"] == "queued"
    assert data["display_name"] == "smoke-train"
    assert data["config_overrides"] == {"train.batch_size": 32}

    queue_files = list(tmp_layout["worker_queue"].glob("*.json"))
    assert len(queue_files) == 1
    payload = json.loads(queue_files[0].read_text())
    assert payload["job_id"] == data["job_id"]
    assert payload["type"] == "train"


def test_create_job_rejects_unknown_type(client: TestClient) -> None:
    resp = client.post("/api/jobs", json={"type": "not_a_real_type"})
    assert resp.status_code in (400, 422)


def test_list_and_get_job(client: TestClient) -> None:
    create = client.post("/api/jobs", json={"type": "eval"})
    assert create.status_code == 201
    job_id = create.json()["job_id"]

    listing = client.get("/api/jobs")
    assert listing.status_code == 200
    items = listing.json()["items"]
    assert any(it["job_id"] == job_id for it in items)

    detail = client.get(f"/api/jobs/{job_id}")
    assert detail.status_code == 200
    assert detail.json()["job_id"] == job_id


def test_get_job_progress_missing_log_ok(client: TestClient) -> None:
    create = client.post("/api/jobs", json={"type": "infer"})
    job_id = create.json()["job_id"]
    resp = client.get(f"/api/jobs/{job_id}/progress")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job_id
    assert data["progress_pct"] == 0.0
    assert data["status"] == "queued"


def test_get_job_logs_empty(client: TestClient) -> None:
    create = client.post("/api/jobs", json={"type": "report"})
    job_id = create.json()["job_id"]
    resp = client.get(f"/api/jobs/{job_id}/logs", params={"tail": 50})
    assert resp.status_code == 200
    assert resp.json()["lines"] == []


def test_cancel_job(client: TestClient, tmp_layout) -> None:
    create = client.post("/api/jobs", json={"type": "train"})
    job_id = create.json()["job_id"]
    resp = client.post(f"/api/jobs/{job_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"

    flag_file = tmp_layout["worker_cancel"] / f"{job_id}.flag"
    assert flag_file.exists()

    follow_up = client.get(f"/api/jobs/{job_id}")
    assert follow_up.json()["status"] == "cancelled"


def test_cancel_missing_job_404(client: TestClient) -> None:
    resp = client.delete("/api/jobs/doesnotexist")
    assert resp.status_code == 404


def test_get_missing_job_404(client: TestClient) -> None:
    resp = client.get("/api/jobs/doesnotexist")
    assert resp.status_code == 404
