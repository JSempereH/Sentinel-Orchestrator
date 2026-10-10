"""
Test isolation: point the worker at a throwaway output dir and a fixed
token *before* app.config is imported anywhere, so tests never touch a
real WORKER_OUTPUT_DIR or require a real WORKER_API_TOKEN.
"""

import os
import tempfile

_tmp_dir = tempfile.mkdtemp(prefix="citycube-worker-test-")
os.environ["WORKER_API_TOKEN"] = "test-token"
os.environ["WORKER_OUTPUT_DIR"] = os.path.join(_tmp_dir, "runs")
# Most tests monkeypatch execute_and_persist, which a spawned job process
# would not see; test_isolation.py turns isolation back on explicitly.
os.environ["JOB_PROCESS_ISOLATION"] = "false"
os.environ["MIN_FREE_DISK_GB"] = "0"

import pytest
from fastapi.testclient import TestClient

from app import jobs
from app.main import app


def _no_real_runs(job_id, request_dict, *, output_root, progress_cb=None):
    raise RuntimeError("tests must never run a real analysis; monkeypatch execute_and_persist")


# The baseline every per-test monkeypatch reverts to. A job thread can
# outlive its test and only then look up execute_and_persist; with the real
# function as the baseline that started live CDSE downloads with the
# credentials in worker/.env.
jobs.execute_and_persist = _no_real_runs


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def auth_headers():
    return {"Authorization": "Bearer test-token"}


@pytest.fixture(autouse=True)
def _isolate_jobs():
    jobs._jobs.clear()
    yield
    jobs._jobs.clear()
