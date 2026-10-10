"""
Tests for the citycube-worker API. `app.jobs.execute_and_persist` is
monkeypatched in every test that submits a job, so nothing here needs
citycube's heavy extras or real CDSE/CDS/OpenAQ credentials.
"""

import threading
import time

from app import jobs


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


def test_jobs_require_auth(client):
    response = client.post("/jobs", json=_fake_request_dict())
    assert response.status_code == 401


def test_submit_rejects_invalid_request(client, auth_headers):
    response = client.post("/jobs", json={"aoi": {}}, headers=auth_headers)
    assert response.status_code == 422


def test_unknown_job_returns_404(client, auth_headers):
    response = client.get("/jobs/does-not-exist", headers=auth_headers)
    assert response.status_code == 404


def test_submit_and_poll_success(client, auth_headers, monkeypatch, tmp_path):
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    (result_dir / "cube.zarr").mkdir()

    def fake_execute(job_id, request_dict, *, output_root, progress_cb=None):
        if progress_cb:
            progress_cb("sentinel3", 1, 1)
        return result_dir

    monkeypatch.setattr(jobs, "execute_and_persist", fake_execute)

    response = client.post("/jobs", json=_fake_request_dict(), headers=auth_headers)
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert response.json()["status"] == "PENDING"

    status = {}
    for _ in range(50):
        status = client.get(f"/jobs/{job_id}", headers=auth_headers).json()
        if status["status"] in ("SUCCEEDED", "FAILED"):
            break
        time.sleep(0.05)

    assert status["status"] == "SUCCEEDED"
    assert status["progress"] == {"sensor": "sentinel3", "done": 1, "total": 1}

    download = client.get(f"/jobs/{job_id}/result", headers=auth_headers)
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/zip"


def test_failed_job_reports_error_message(client, auth_headers, monkeypatch):
    def fake_execute(job_id, request_dict, *, output_root, progress_cb=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(jobs, "execute_and_persist", fake_execute)

    response = client.post("/jobs", json=_fake_request_dict(), headers=auth_headers)
    job_id = response.json()["job_id"]

    status = {}
    for _ in range(50):
        status = client.get(f"/jobs/{job_id}", headers=auth_headers).json()
        if status["status"] in ("SUCCEEDED", "FAILED"):
            break
        time.sleep(0.05)

    assert status["status"] == "FAILED"
    assert status["error_message"] == "boom"


def test_download_before_success_returns_409(client, auth_headers, monkeypatch):
    ready = threading.Event()

    def fake_execute(job_id, request_dict, *, output_root, progress_cb=None):
        ready.wait(timeout=5)
        return output_root

    monkeypatch.setattr(jobs, "execute_and_persist", fake_execute)

    response = client.post("/jobs", json=_fake_request_dict(), headers=auth_headers)
    job_id = response.json()["job_id"]

    download = client.get(f"/jobs/{job_id}/result", headers=auth_headers)
    assert download.status_code == 409

    ready.set()


def test_jobs_beyond_the_concurrency_limit_wait_as_pending(client, auth_headers, monkeypatch, tmp_path):
    release = threading.Event()
    running = []

    def fake_execute(job_id, request_dict, *, output_root, progress_cb=None):
        running.append(job_id)
        release.wait(timeout=5)
        return tmp_path

    monkeypatch.setattr(jobs, "execute_and_persist", fake_execute)
    monkeypatch.setattr(jobs, "_run_slots", threading.BoundedSemaphore(1))

    first = client.post("/jobs", json=_fake_request_dict(), headers=auth_headers).json()["job_id"]
    second = client.post("/jobs", json=_fake_request_dict(), headers=auth_headers).json()["job_id"]
    for _ in range(50):
        if running:
            break
        time.sleep(0.02)
    time.sleep(0.1)

    assert running == [first]
    assert client.get(f"/jobs/{second}", headers=auth_headers).json()["status"] == "PENDING"

    release.set()
    for _ in range(100):
        if client.get(f"/jobs/{second}", headers=auth_headers).json()["status"] == "SUCCEEDED":
            break
        time.sleep(0.05)
    assert running == [first, second]


def test_submit_rejects_a_request_beyond_the_worker_limits(client, auth_headers, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "max_aoi_km2", 100)
    oversized = {**_fake_request_dict(), "aoi": {"west": 13.0, "south": 52.0, "east": 14.0, "north": 53.0}}

    response = client.post("/jobs", json=oversized, headers=auth_headers)

    assert response.status_code == 422
    assert "AOI is" in response.json()["detail"]


def test_build_request_passes_overpass_error_policy_and_downscaling():
    from app.runner import build_request

    request = build_request({
        **_fake_request_dict(),
        "sensors": ["sentinel3", "sentinel2"],
        "thermal_overpass": "day",
        "on_product_error": "raise",
        "downscale": {"model": "linear"},
    })

    assert request.thermal_overpass == "day"
    assert request.on_product_error == "raise"
    assert request.downscale is not None and request.downscale.model == "linear"
