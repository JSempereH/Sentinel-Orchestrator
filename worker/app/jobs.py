"""Job store and background execution for analysis runs.

Each job gets a supervising daemon thread that waits for a run slot: jobs
are not cheap (one Sentinel-1 SNAP run alone peaked at ~11 GB of RAM, and
two concurrent live-data runs froze the host twice, see docs/history.md),
so at most MAX_CONCURRENT_JOBS (default 1) execute at a time and the rest
stay PENDING.

With JOB_PROCESS_ISOLATION (the default) the run itself happens in a child
process started by that thread, in its own process group:
- it can be cancelled (POST /jobs/{id}/cancel) and is killed after
  JOB_TIMEOUT_HOURS, together with anything it spawned (SNAP's gpt);
- a crash or out-of-memory kill takes down only that job, not the API;
- it dies with the worker (PR_SET_PDEATHSIG) instead of running orphaned;
- its log goes to <WORKER_OUTPUT_DIR>/<job id>/job.log.
Progress and the outcome come back over a multiprocessing queue. Without
isolation (tests) the run happens in the supervising thread itself and a
running job cannot be cancelled.

State is cached in memory and mirrored to SQLite (<WORKER_OUTPUT_DIR>/jobs.db)
so job history survives a restart. A restart cannot resume a run, so any
job still PENDING/RUNNING when the process starts is marked FAILED with an
explanatory message. Finished jobs older than JOB_RETENTION_DAYS are deleted
with their files by `prune_expired`, which the API runs periodically.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import queue as queue_module
import resource
import shutil
import signal
import sqlite3
import threading
import time
import uuid
from importlib import import_module
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import settings
from .runner import execute_and_persist

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_jobs: dict[str, "JobState"] = {}
_run_slots = threading.BoundedSemaphore(max(1, settings.max_concurrent_jobs))
_cancel_requested: set[str] = set()

PENDING, RUNNING, SUCCEEDED, FAILED, CANCELLED = "PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"
TERMINAL = frozenset({SUCCEEDED, FAILED, CANCELLED})

_RESTART_ERROR = "Worker process restarted while this job was in progress; it did not complete."
_POLL_S = 0.5
_TERMINATE_GRACE_S = 15
# What a job process runs, as "module:function" (resolved in the child, so
# tests can point it at controllable stand-ins).
JOB_RUNNER = "app.runner:execute_and_persist"
_COLUMNS = ("id", "status", "progress", "error_message", "result_dir", "request_params", "name", "created_at", "started_at", "finished_at", "metrics")
_JSON_COLUMNS = {"progress", "request_params", "metrics"}


class JobConflictError(RuntimeError):
    """The job is not in a state that allows the requested operation."""


@dataclass
class JobState:
    id: str
    status: str = PENDING
    progress: dict[str, Any] | None = None
    error_message: str | None = None
    result_dir: Path | None = None
    request_params: dict[str, Any] | None = None
    name: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    # duration_s, peak_rss_mb, failed_products
    metrics: dict[str, Any] = field(default_factory=dict)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_dir(job_id: str) -> Path:
    return Path(settings.worker_output_dir) / job_id


def log_path(job_id: str) -> Path:
    return job_dir(job_id) / "job.log"


# ── persistence ──────────────────────────────────────────────────────────


def _db_path() -> Path:
    return Path(settings.worker_output_dir) / "jobs.db"


def _db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            progress TEXT,
            error_message TEXT,
            result_dir TEXT,
            request_params TEXT,
            name TEXT,
            created_at TEXT
        )
        """
    )
    _migrate_schema(conn)
    return conn


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the initial CREATE TABLE, since there's
    no Alembic here (CREATE TABLE IF NOT EXISTS only handles brand-new
    tables, not existing ones)."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    for column in ("request_params", "name", "created_at", "started_at", "finished_at", "metrics"):
        if column not in columns:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} TEXT")
    conn.commit()


def _db_write(state: JobState) -> None:
    values = []
    for column in _COLUMNS:
        value = getattr(state, column)
        if column in _JSON_COLUMNS:
            value = json.dumps(value) if value is not None else None
        elif isinstance(value, Path):
            value = str(value)
        values.append(value)
    try:
        with _db_connect() as conn:
            conn.execute(f"INSERT OR REPLACE INTO jobs ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' * len(_COLUMNS))})", values)
    except Exception:
        logger.exception("Could not persist job %s to %s", state.id, _db_path())


def _db_delete(job_id: str) -> None:
    with _db_connect() as conn:
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


def load_from_disk() -> None:
    """Load job history from disk at startup, failing out any job left
    PENDING/RUNNING by a previous process that never reached a terminal state."""
    if not _db_path().exists():
        return
    with _db_connect() as conn:
        rows = conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM jobs").fetchall()
    for row in rows:
        values: dict[str, Any] = dict(zip(_COLUMNS, row))
        for column in _JSON_COLUMNS:
            values[column] = json.loads(values[column]) if values[column] else None
        values["metrics"] = values["metrics"] or {}
        values["result_dir"] = Path(values["result_dir"]) if values["result_dir"] else None
        state = JobState(**values)
        if state.status not in TERMINAL:
            state.status, state.error_message, state.finished_at = FAILED, _RESTART_ERROR, _now()
            _db_write(state)
        _jobs[state.id] = state
    logger.info("Loaded %d job(s) from %s", len(rows), _db_path())


# ── public operations ────────────────────────────────────────────────────


def create(request_dict: dict, *, name: str | None = None) -> str:
    job_id = str(uuid.uuid4())
    state = JobState(id=job_id, request_params=request_dict, name=name, created_at=_now())
    with _lock:
        _jobs[job_id] = state
    _db_write(state)
    threading.Thread(target=_run, args=(job_id, request_dict), name=f"job-{job_id[:8]}", daemon=True).start()
    return job_id


def get(job_id: str) -> JobState | None:
    with _lock:
        state = _jobs.get(job_id)
        return JobState(**vars(state)) if state is not None else None


def list_all() -> list[JobState]:
    """All known jobs, most recently created first."""
    with _lock:
        states = [JobState(**vars(state)) for state in _jobs.values()]
    return sorted(states, key=lambda state: state.created_at or "", reverse=True)


def cancel(job_id: str) -> str:
    """Cancel a pending or running job; returns its status after the request.

    A pending job is cancelled at once. A running one is terminated by its
    supervising thread within a second, so it is still RUNNING on return.
    """

    with _lock:
        state = _jobs.get(job_id)
        if state is None:
            raise KeyError(job_id)
        if state.status in TERMINAL:
            raise JobConflictError(f"Job is already {state.status}")
        if state.status == RUNNING and not settings.job_process_isolation:
            raise JobConflictError("Running jobs can only be cancelled with JOB_PROCESS_ISOLATION enabled")
        _cancel_requested.add(job_id)
        status = state.status
    if status == PENDING:
        _update(job_id, status=CANCELLED, error_message="Cancelled before it started", finished_at=_now())
        return CANCELLED
    return RUNNING


def delete(job_id: str) -> None:
    """Delete a finished job's record and every file it produced."""

    with _lock:
        state = _jobs.get(job_id)
        if state is None:
            raise KeyError(job_id)
        if state.status not in TERMINAL:
            raise JobConflictError(f"Job is {state.status}; cancel it first")
        del _jobs[job_id]
        _cancel_requested.discard(job_id)
    _remove_files(job_id)
    _db_delete(job_id)
    logger.info("Deleted job %s", job_id)


def prune_expired(now: datetime | None = None) -> list[str]:
    """Delete finished jobs older than JOB_RETENTION_DAYS; returns their ids."""

    if settings.job_retention_days <= 0:
        return []
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=settings.job_retention_days)
    expired = [
        state.id for state in list_all()
        if state.status in TERMINAL and datetime.fromisoformat(state.finished_at or state.created_at or _now()) < cutoff
    ]
    for job_id in expired:
        try:
            delete(job_id)
        except (KeyError, JobConflictError):  # deleted or restarted meanwhile
            continue
    if expired:
        logger.info("Retention: deleted %d job(s) older than %s days", len(expired), settings.job_retention_days)
    return expired


def free_disk_gb() -> float:
    return shutil.disk_usage(settings.worker_output_dir).free / 1e9


def has_enough_disk() -> bool:
    return settings.min_free_disk_gb <= 0 or free_disk_gb() >= settings.min_free_disk_gb


# ── execution ────────────────────────────────────────────────────────────


def _update(job_id: str, **fields: Any) -> None:
    snapshot: JobState | None = None
    with _lock:
        state = _jobs.get(job_id)
        if state is not None:
            for key, value in fields.items():
                setattr(state, key, value)
            snapshot = JobState(**vars(state))
    if snapshot is not None:
        _db_write(snapshot)


def _remove_files(job_id: str) -> None:
    root = Path(settings.worker_output_dir).resolve()
    path = job_dir(job_id).resolve()
    if path.parent == root and path.exists():  # never outside the output dir
        shutil.rmtree(path, ignore_errors=True)


def _run(job_id: str, request_dict: dict) -> None:
    with _run_slots:
        state = get(job_id)
        if state is None or state.status != PENDING:  # cancelled or deleted while waiting
            with _lock:
                _cancel_requested.discard(job_id)
            return
        if not has_enough_disk():
            _update(job_id, status=FAILED, finished_at=_now(),
                    error_message=f"Not started: {free_disk_gb():.1f} GB free, below MIN_FREE_DISK_GB={settings.min_free_disk_gb}")
            return
        started = time.monotonic()
        _update(job_id, status=RUNNING, started_at=_now())
        try:
            if settings.job_process_isolation:
                status, error, result_dir, metrics = _execute_in_process(job_id, request_dict)
            else:
                status, error, result_dir, metrics = _execute_in_thread(job_id, request_dict)
        except Exception as exc:  # noqa: BLE001 - a supervisor bug must still finish the job
            logger.exception("Supervising job %s failed", job_id)
            status, error, result_dir, metrics = FAILED, f"Worker error: {exc}", None, {}
        metrics = {**metrics, "duration_s": round(time.monotonic() - started, 1)}
        if result_dir is not None:
            metrics["failed_products"] = _failed_product_count(result_dir)
            if not settings.keep_work_dir:
                shutil.rmtree(job_dir(job_id) / "work", ignore_errors=True)
        _update(job_id, status=status, error_message=error, result_dir=result_dir, finished_at=_now(), metrics=metrics)
        with _lock:
            _cancel_requested.discard(job_id)
        logger.info("Job %s %s in %.0f s", job_id, status, metrics["duration_s"])


def _failed_product_count(result_dir: Path) -> int | None:
    try:
        provenance = json.loads((result_dir / "provenance.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return len(provenance.get("failed_products", []))


def _progress_callback(job_id: str):
    def report(sensor: str, done: int, total: int) -> None:
        _update(job_id, progress={"sensor": sensor, "done": done, "total": total})

    return report


def _execute_in_thread(job_id: str, request_dict: dict) -> tuple[str, str | None, Path | None, dict]:
    try:
        result_dir = execute_and_persist(job_id, request_dict, output_root=Path(settings.worker_output_dir), progress_cb=_progress_callback(job_id))
    except Exception as exc:  # noqa: BLE001 - surface any failure to the caller
        logger.exception("Job %s failed", job_id)
        return FAILED, str(exc), None, {}
    return SUCCEEDED, None, result_dir, {}


def _execute_in_process(job_id: str, request_dict: dict) -> tuple[str, str | None, Path | None, dict]:
    context = multiprocessing.get_context("spawn")
    messages = context.Queue()
    process = context.Process(
        target=_child_main,
        args=(job_id, request_dict, settings.worker_output_dir, messages, JOB_RUNNER),
        name=f"sentinel-job-{job_id[:8]}",
    )
    process.start()
    timeout_s = settings.job_timeout_hours * 3600 if settings.job_timeout_hours > 0 else None
    deadline = time.monotonic() + timeout_s if timeout_s else None
    report = _progress_callback(job_id)
    outcome: tuple[str, str | None, Path | None, dict] | None = None
    stopped: tuple[str, str] | None = None

    while outcome is None:
        try:
            kind, *payload = messages.get(timeout=_POLL_S)
        except queue_module.Empty:
            kind, payload = None, []
        if kind == "progress":
            report(*payload)
        elif kind == "succeeded":
            outcome = (SUCCEEDED, None, Path(payload[0]), payload[1])
        elif kind == "failed":
            outcome = (FAILED, payload[0], None, payload[1])
        elif not process.is_alive():
            break
        with _lock:
            cancelled = job_id in _cancel_requested
        if outcome is None and cancelled:
            stopped = (CANCELLED, "Cancelled while running")
        elif outcome is None and deadline is not None and time.monotonic() > deadline:
            stopped = (FAILED, f"Timed out after JOB_TIMEOUT_HOURS={settings.job_timeout_hours} and was stopped")
        if stopped:
            _terminate(process)
            return stopped[0], stopped[1], None, {}

    process.join(_TERMINATE_GRACE_S)
    if process.is_alive():
        _terminate(process)
    # The child may have sent its outcome and exited between the last
    # queue read and the liveness check: read what is left before deciding.
    while outcome is None:
        try:
            kind, *payload = messages.get(timeout=_POLL_S)
        except queue_module.Empty:
            break
        if kind == "succeeded":
            outcome = (SUCCEEDED, None, Path(payload[0]), payload[1])
        elif kind == "failed":
            outcome = (FAILED, payload[0], None, payload[1])
    if outcome is not None:
        return outcome
    return FAILED, _exit_message(process.exitcode), None, {}


def _terminate(process: multiprocessing.process.BaseProcess) -> None:
    """Stop the job's whole process group (the run and anything it spawned)."""

    if process.pid is None:
        return
    for sig, wait in ((signal.SIGTERM, _TERMINATE_GRACE_S), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        process.join(wait)
        if not process.is_alive():
            break


def _exit_message(exitcode: int | None) -> str:
    if exitcode == -signal.SIGKILL:
        return "The job process was killed (SIGKILL), most likely by the out-of-memory killer; reduce the request size"
    if exitcode is not None and exitcode < 0:
        return f"The job process was terminated by signal {-exitcode}"
    return f"The job process exited unexpectedly with code {exitcode}"


def _child_main(job_id: str, request_dict: dict, output_root: str, messages, runner: str) -> None:
    """Entry point of a job's child process (spawned, so a fresh interpreter)."""

    os.setsid()  # own process group, so cancellation also reaches SNAP's gpt
    _die_with_parent()
    from .logging_setup import configure_job_logging

    configure_job_logging(job_id, log_path(job_id))
    log = logging.getLogger(__name__)
    from citycube.version import build_info

    log.info("Job started (citycube %s, commit %s)", build_info()["version"], build_info()["git_commit"])

    def metrics() -> dict[str, float]:
        # The larger of this process and any external tool it ran (SNAP's
        # gpt): whichever set the machine's memory peak.
        own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
        return {"peak_rss_mb": round(max(own, children), 1)}

    module, _, function = runner.partition(":")
    run = getattr(import_module(module), function)
    try:
        result_dir = run(
            job_id,
            request_dict,
            output_root=Path(output_root),
            progress_cb=lambda sensor, done, total: messages.put(("progress", sensor, done, total)),
        )
    except BaseException as exc:  # noqa: BLE001 - every failure must reach the parent
        log.exception("Job %s failed", job_id)
        messages.put(("failed", f"{type(exc).__name__}: {exc}", metrics()))
    else:
        log.info("Result written to %s", result_dir)
        messages.put(("succeeded", str(result_dir), metrics()))
    finally:
        messages.close()
        messages.join_thread()


def _die_with_parent() -> None:
    """Ask Linux to SIGTERM this process when the worker that started it dies."""

    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        pr_set_pdeathsig = 1
        libc.prctl(pr_set_pdeathsig, signal.SIGTERM)
    except (OSError, AttributeError):  # not Linux: an orphan is then possible
        pass
