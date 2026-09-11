"""
sentinel-worker API

Thin FastAPI wrapper around sentinel_analysis, deployable to any machine
with the heavy geospatial dependencies installed (a beefier PC, a cloud VM,
reachable over SSH tunnel/VPN/LAN). Exposes a simple submit/poll/download
job shape: submit a job, poll its status, download the result once
SUCCEEDED.

The built frontend (`worker/frontend/dist/`, see `worker/frontend/README.md`)
is served from this same process via the StaticFiles mount at the bottom of
this file - one deployable service, no CORS needed anywhere: the mount is
registered *after* every API route, so `/health`, `/jobs`, `/usage` keep
matching those exact routes first and only unmatched paths fall through to
the SPA's `index.html`. If the frontend hasn't been built yet (e.g. in
tests, or a bare API-only deployment), the mount is skipped entirely and
this API still works standalone exactly as before.

Run with: uvicorn app.main:app --host 0.0.0.0 --port 8100
"""

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from . import jobs, usage
from .auth import require_token
from .runner import build_request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

app = FastAPI(title="sentinel-worker API", version="0.1.0")

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


@app.on_event("startup")
def startup_event():
    jobs.load_from_disk()


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/usage", dependencies=[Depends(require_token)])
def usage_check():
    """Best-effort quota/credentials snapshot for the auxiliary providers this
    worker calls. Unlike HyP3, CDSE/CDS-ADS/OpenAQ are not numeric-credit
    systems; this reports what each one actually exposes (OpenAQ rate-limit
    headers; presence of credentials for the others)."""
    return usage.check_all()


def _job_summary(state: jobs.JobState) -> dict:
    return {
        "job_id": state.id,
        "name": state.name,
        "status": state.status,
        "progress": state.progress,
        "error_message": state.error_message,
        "created_at": state.created_at,
    }


@app.post("/jobs", dependencies=[Depends(require_token)])
def submit_job(request: dict, name: str | None = None):
    try:
        build_request(request)
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid request: {exc}") from exc
    job_id = jobs.create(request, name=name)
    return {"job_id": job_id, "status": "PENDING"}


@app.get("/jobs", dependencies=[Depends(require_token)])
def list_jobs():
    return [_job_summary(state) for state in jobs.list_all()]


@app.get("/jobs/{job_id}", dependencies=[Depends(require_token)])
def get_job(job_id: str):
    state = jobs.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {**_job_summary(state), "request_params": state.request_params}


@app.get("/jobs/{job_id}/result", dependencies=[Depends(require_token)])
def download_result(job_id: str):
    state = jobs.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if state.status != "SUCCEEDED" or state.result_dir is None:
        raise HTTPException(status_code=409, detail=f"Job is {state.status}, not SUCCEEDED")

    archive_base = Path(tempfile.gettempdir()) / f"sentinel-worker-{job_id}"
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=state.result_dir)
    # Deleted once the response finishes streaming, not left in /tmp forever -
    # a long-running worker downloading many distinct jobs would otherwise
    # slowly fill the host's temp directory with one zip per job ever
    # downloaded.
    cleanup = BackgroundTask(lambda: Path(archive_path).unlink(missing_ok=True))
    return FileResponse(archive_path, media_type="application/zip", filename=f"{job_id}.zip", background=cleanup)


# Registered last on purpose - see module docstring. Skipped entirely if the
# frontend hasn't been built, so this API keeps working standalone.
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
