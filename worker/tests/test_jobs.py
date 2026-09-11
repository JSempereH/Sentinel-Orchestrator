"""
Tests for the sentinel-worker API. `app.jobs.execute_and_persist` is
monkeypatched in every test that submits a job, so nothing here needs
sentinel_analysis's heavy extras or real CDSE/CDS/OpenAQ credentials.
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
