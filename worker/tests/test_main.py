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


def _wait_status(client, auth_headers, job_id, statuses, attempts=200):
    import time

    for _ in range(attempts):
        body = client.get(f"/jobs/{job_id}", headers=auth_headers).json()
        if body["status"] in statuses:
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} stayed {body['status']}")


def test_health_reports_the_running_build(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["version"] and "git_commit" in body


def test_readiness_requires_auth_and_reports_503_when_a_check_fails(client, auth_headers, monkeypatch):
    from citycube.credentials import CredentialCheck

    from app import main

    monkeypatch.setattr(main, "check_credentials", lambda: {"cdse_oauth_client": CredentialCheck("cdse_oauth_client", "error", "rejected: HTTP 401")})
    assert client.get("/health/ready").status_code == 401

    response = client.get("/health/ready", params={"refresh": True}, headers=auth_headers)

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert body["checks"]["credentials.cdse_oauth_client"]["detail"] == "rejected: HTTP 401"
    assert body["checks"]["disk"]["status"] == "ok"

    monkeypatch.setattr(main, "check_credentials", lambda: {"openaq": CredentialCheck("openaq", "warning", "expires in 2 days")})
    response = client.get("/health/ready", params={"refresh": True}, headers=auth_headers)
    assert response.status_code == 200 and response.json()["status"] == "warning"


def test_pending_job_can_be_cancelled_then_deleted(client, auth_headers, monkeypatch, tmp_path):
    from app import jobs

    release = threading.Event()

    def blocking(job_id, request_dict, *, output_root, progress_cb=None):
        release.wait(timeout=5)
        return tmp_path

    monkeypatch.setattr(jobs, "execute_and_persist", blocking)
    monkeypatch.setattr(jobs, "_run_slots", threading.BoundedSemaphore(1))
    first = client.post("/jobs", json=_fake_request_dict(), headers=auth_headers).json()["job_id"]
    second = client.post("/jobs", json=_fake_request_dict(), headers=auth_headers).json()["job_id"]
    _wait_status(client, auth_headers, first, {"RUNNING"})

    assert client.post(f"/jobs/{second}/cancel", headers=auth_headers).json()["status"] == "CANCELLED"
    assert client.delete(f"/jobs/{first}", headers=auth_headers).status_code == 409  # still running
    # Without process isolation a running job cannot be cancelled.
    assert client.post(f"/jobs/{first}/cancel", headers=auth_headers).status_code == 409

    release.set()
    _wait_status(client, auth_headers, first, {"SUCCEEDED"})
    assert client.delete(f"/jobs/{second}", headers=auth_headers).json() == {"job_id": second, "deleted": True}
    assert client.get(f"/jobs/{second}", headers=auth_headers).status_code == 404
    assert client.post("/jobs/missing/cancel", headers=auth_headers).status_code == 404


def test_job_log_returns_the_tail(client, auth_headers, monkeypatch):
    from app import jobs

    monkeypatch.setattr(jobs, "execute_and_persist", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    job_id = client.post("/jobs", json=_fake_request_dict(), headers=auth_headers).json()["job_id"]
    _wait_status(client, auth_headers, job_id, {"FAILED"})
    path = jobs.log_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"line {index}\n" for index in range(10)))

    assert client.get(f"/jobs/{job_id}/log", params={"lines": 3}, headers=auth_headers).text == "line 7\nline 8\nline 9\n"


def test_submit_is_refused_when_the_disk_is_nearly_full(client, auth_headers, monkeypatch):
    from app import jobs
    from app.config import settings

    monkeypatch.setattr(settings, "min_free_disk_gb", 50)
    monkeypatch.setattr(jobs, "free_disk_gb", lambda: 3.0)

    response = client.post("/jobs", json=_fake_request_dict(), headers=auth_headers)

    assert response.status_code == 507
    assert "3.0 GB free" in response.json()["detail"]


def test_job_logs_never_contain_openeo_client_ids(tmp_path):
    import logging

    from app.logging_setup import configure_job_logging

    log_file = tmp_path / "job.log"
    configure_job_logging("job-1234", log_file)
    logging.getLogger("openeo.rest.auth.oidc").info("token request with client_id 'SECRET-CLIENT-ID'")
    logging.getLogger("citycube.workflow.runner").info("sentinel3: acquiring 2 products")
    for handler in logging.getLogger().handlers:
        handler.flush()

    text = log_file.read_text()
    assert "SECRET-CLIENT-ID" not in text
    assert "acquiring 2 products" in text


def test_concurrent_downloads_of_one_result_do_not_collide(client, auth_headers, monkeypatch, tmp_path):
    import io
    import threading
    import zipfile

    from app import jobs

    result_dir = tmp_path / "result"
    result_dir.mkdir()
    (result_dir / "provenance.json").write_text("{}")
    (result_dir / "data.bin").write_bytes(b"x" * 2_000_000)
    monkeypatch.setattr(jobs, "execute_and_persist", lambda *a, **k: result_dir)
    job_id = client.post("/jobs", json=_fake_request_dict(), headers=auth_headers).json()["job_id"]
    _wait_status(client, auth_headers, job_id, {"SUCCEEDED"})

    bodies, threads = [], []
    for _ in range(4):
        thread = threading.Thread(target=lambda: bodies.append(client.get(f"/jobs/{job_id}/result", headers=auth_headers).content))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

    assert len(bodies) == 4
    for body in bodies:
        assert sorted(zipfile.ZipFile(io.BytesIO(body)).namelist()) == ["data.bin", "provenance.json"]
