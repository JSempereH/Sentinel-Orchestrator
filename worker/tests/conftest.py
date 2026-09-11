"""
Test isolation: point the worker at a throwaway output dir and a fixed
token *before* app.config is imported anywhere, so tests never touch a
real WORKER_OUTPUT_DIR or require a real WORKER_API_TOKEN.
"""

import os
import tempfile

_tmp_dir = tempfile.mkdtemp(prefix="sentinel-worker-test-")
os.environ["WORKER_API_TOKEN"] = "test-token"
os.environ["WORKER_OUTPUT_DIR"] = os.path.join(_tmp_dir, "runs")

import pytest
from fastapi.testclient import TestClient

from app import jobs
from app.main import app


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
