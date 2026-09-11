"""
Endpoint-level tests for the worker's HTTP API - `test_jobs.py`/`test_runner.py`
cover the job store and request-building in isolation; this file exercises
the actual routes (auth, status codes, response shapes) via TestClient.
"""

import threading

import pytest

from app.main import FRONTEND_DIST


def _fake_request_dict():
    return {
        "aoi": {"west": 13.3, "south": 52.4, "east": 13.5, "north": 52.6},
        "start": "2024-06-01",
        "end": "2024-06-02",
        "sensors": ["sentinel3"],
    }


def test_health_requires_no_auth(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_usage_requires_auth(client):
    assert client.get("/usage").status_code == 401


def test_jobs_endpoints_require_auth(client):
    assert client.post("/jobs", json=_fake_request_dict()).status_code == 401
    assert client.get("/jobs").status_code == 401
    assert client.get("/jobs/does-not-exist").status_code == 401
    assert client.get("/jobs/does-not-exist/result").status_code == 401


def test_submit_get_and_list_round_trip(client, auth_headers, monkeypatch):
    """POST /jobs -> GET /jobs/{id} must echo back request_params/name/created_at
    (the gap this change closes - previously only status/progress survived),
    and GET /jobs must list it too."""

    def fake_execute(_job_id, _request_dict, *, output_root, progress_cb=None):
        return output_root  # never reaches a terminal state before assertions below

    monkeypatch.setattr("app.jobs.execute_and_persist", fake_execute)

    request_dict = _fake_request_dict()
    response = client.post("/jobs", json=request_dict, params={"name": "Berlin test"}, headers=auth_headers)
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    detail = client.get(f"/jobs/{job_id}", headers=auth_headers).json()
    assert detail["job_id"] == job_id
    assert detail["name"] == "Berlin test"
    assert detail["request_params"] == request_dict
    assert detail["created_at"]  # a real ISO timestamp, not null

    listed = client.get("/jobs", headers=auth_headers).json()
    assert any(row["job_id"] == job_id and row["name"] == "Berlin test" for row in listed)
    # The list view is a summary - it must not echo the full request back.
    assert "request_params" not in listed[0]


def test_get_unknown_job_returns_404(client, auth_headers):
    assert client.get("/jobs/does-not-exist", headers=auth_headers).status_code == 404


def test_download_before_success_returns_409(client, auth_headers, monkeypatch):
    ready = threading.Event()

    def fake_execute(_job_id, _request_dict, *, output_root, progress_cb=None):
        ready.wait(timeout=5)
        return output_root

    monkeypatch.setattr("app.jobs.execute_and_persist", fake_execute)

    job_id = client.post("/jobs", json=_fake_request_dict(), headers=auth_headers).json()["job_id"]
    assert client.get(f"/jobs/{job_id}/result", headers=auth_headers).status_code == 409
    ready.set()


def test_submit_rejects_invalid_request(client, auth_headers):
    response = client.post("/jobs", json={"aoi": {}}, headers=auth_headers)
    assert response.status_code == 422


@pytest.mark.skipif(not FRONTEND_DIST.is_dir(), reason="frontend not built (run `npm run build` in worker/frontend first)")
def test_frontend_is_served_at_root_without_shadowing_the_api(client, auth_headers):
    """Proves the StaticFiles mount is registered *after* the API routes -
    the SPA answers unmatched paths, but /jobs (an exact API route) must
    still resolve to the API, not fall through to index.html."""

    root = client.get("/")
    assert root.status_code == 200
    assert "text/html" in root.headers["content-type"]

    jobs_response = client.get("/jobs", headers=auth_headers)
    assert jobs_response.status_code == 200
    assert jobs_response.headers["content-type"].startswith("application/json")
