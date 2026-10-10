"""
citycube-worker API

Thin FastAPI wrapper around citycube, deployable to any machine
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

Health: `/health` is an unauthenticated liveness probe (the container
HEALTHCHECK); `/health/ready` (authenticated) also checks free disk space,
the job database and every data-provider credential, cached for
READY_CACHE_S so polling it does not hammer the providers.

Run with: uvicorn app.main:app --host 0.0.0.0 --port 8100
"""

import logging
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from citycube.credentials import ERROR, WARNING, check_credentials
from citycube.version import build_info
from citycube.workflow.limits import check_request
from starlette.background import BackgroundTask

from . import jobs, usage
from .auth import require_token
from .config import settings
from .logging_setup import configure_logging
from .runner import build_request, request_limits

configure_logging()
logger = logging.getLogger(__name__)

BUILD = build_info()
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
RETENTION_INTERVAL_S = 3600
READY_CACHE_S = 600
LOG_TAIL_LINES = 500

# (monotonic time of the check, its result)
_ready_cache: tuple[float, dict] | None = None
_ready_lock = threading.Lock()


def _retention_loop(stop: threading.Event) -> None:
    while not stop.wait(RETENTION_INTERVAL_S):
        try:
            jobs.prune_expired()
        except Exception:  # noqa: BLE001 - keep the loop alive
            logger.exception("Retention pass failed")


@asynccontextmanager
async def lifespan(_: FastAPI):
    jobs.load_from_disk()
    jobs.prune_expired()
    stop = threading.Event()
    threading.Thread(target=_retention_loop, args=(stop,), name="retention", daemon=True).start()
    logger.info("citycube-worker %s (commit %s) started", BUILD["version"], BUILD["git_commit"])
    yield
    stop.set()


app = FastAPI(title="citycube-worker API", version=str(BUILD["version"]), lifespan=lifespan)


@app.get("/health")
def health_check():
    return {"status": "ok", "version": BUILD["version"], "git_commit": BUILD["git_commit"]}


def _readiness() -> dict:
    checks: dict[str, dict] = {}
    free_gb = jobs.free_disk_gb()
    checks["disk"] = {
        "status": "ok" if jobs.has_enough_disk() else ERROR,
        "detail": f"{free_gb:.1f} GB free (minimum {settings.min_free_disk_gb} GB)",
    }
    try:
        jobs._db_connect().close()
        checks["job_database"] = {"status": "ok", "detail": str(jobs._db_path())}
    except Exception as exc:  # noqa: BLE001
        checks["job_database"] = {"status": ERROR, "detail": f"{type(exc).__name__}: {exc}"}
    checks.update({f"credentials.{name}": result.to_dict() for name, result in check_credentials().items()})
    statuses = {check["status"] for check in checks.values()}
    overall = ERROR if ERROR in statuses else WARNING if WARNING in statuses else "ok"
    return {"status": overall, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": checks}


@app.get("/health/ready", dependencies=[Depends(require_token)])
def readiness_check(response: Response, refresh: bool = False):
    """Disk, job database and live credential checks; HTTP 503 when any fails."""
    global _ready_cache
    with _ready_lock:
        if refresh or _ready_cache is None or time.monotonic() - _ready_cache[0] > READY_CACHE_S:
            _ready_cache = (time.monotonic(), _readiness())
        result = _ready_cache[1]
    if result["status"] == ERROR:
        response.status_code = 503
    return result


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
        "started_at": state.started_at,
        "finished_at": state.finished_at,
        "metrics": state.metrics,
    }


def _require_job(job_id: str) -> jobs.JobState:
    state = jobs.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return state


@app.post("/jobs", dependencies=[Depends(require_token)])
def submit_job(request: dict, name: str | None = None):
    try:
        # Raises RequestTooLargeError (a ValueError) before the job is queued.
        check_request(build_request(request), request_limits())
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid request: {exc}") from exc
    if not jobs.has_enough_disk():
        raise HTTPException(
            status_code=507,
            detail=f"Only {jobs.free_disk_gb():.1f} GB free in the output directory (minimum {settings.min_free_disk_gb} GB); delete old jobs first",
        )
    job_id = jobs.create(request, name=name)
    return {"job_id": job_id, "status": jobs.PENDING}


@app.get("/jobs", dependencies=[Depends(require_token)])
def list_jobs():
    return [_job_summary(state) for state in jobs.list_all()]


@app.get("/jobs/{job_id}", dependencies=[Depends(require_token)])
def get_job(job_id: str):
    state = _require_job(job_id)
    return {**_job_summary(state), "request_params": state.request_params}


@app.post("/jobs/{job_id}/cancel", dependencies=[Depends(require_token)])
def cancel_job(job_id: str):
    try:
        status = jobs.cancel(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found") from None
    except jobs.JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"job_id": job_id, "status": status, "cancel_requested": True}


@app.delete("/jobs/{job_id}", dependencies=[Depends(require_token)])
def delete_job(job_id: str):
    try:
        jobs.delete(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found") from None
    except jobs.JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"job_id": job_id, "deleted": True}


@app.get("/jobs/{job_id}/log", dependencies=[Depends(require_token)], response_class=PlainTextResponse)
def job_log(job_id: str, lines: int = LOG_TAIL_LINES):
    """The last `lines` lines of the job's own log (job.log)."""
    _require_job(job_id)
    path = jobs.log_path(job_id)
    if not path.exists():
        return ""
    with path.open(encoding="utf-8", errors="replace") as handle:
        tail = handle.readlines()[-max(1, lines):]
    return "".join(tail)


@app.get("/jobs/{job_id}/result", dependencies=[Depends(require_token)])
def download_result(job_id: str):
    state = _require_job(job_id)
    if state.status != jobs.SUCCEEDED or state.result_dir is None:
        raise HTTPException(status_code=409, detail=f"Job is {state.status}, not SUCCEEDED")

    # A fresh directory per request: with a fixed name per job, two concurrent
    # downloads of the same result overwrote one zip and the first to finish
    # deleted it under the other. Removed once the response has streamed.
    scratch = Path(tempfile.mkdtemp(prefix="citycube-worker-download-"))
    archive_path = shutil.make_archive(str(scratch / job_id), "zip", root_dir=state.result_dir)
    cleanup = BackgroundTask(shutil.rmtree, scratch, ignore_errors=True)
    return FileResponse(archive_path, media_type="application/zip", filename=f"{job_id}.zip", background=cleanup)


# Registered last on purpose - see module docstring. Skipped entirely if the
# frontend hasn't been built, so this API keeps working standalone.
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
