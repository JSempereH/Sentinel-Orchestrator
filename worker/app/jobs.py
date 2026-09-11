"""Job store and background execution for analysis runs.

Each submitted job runs in its own daemon thread. Jobs here are independent
runs - possibly for different projects/AOIs entirely - so, unlike a queue
that deliberately serializes calls to one rate-limited external API through
a single worker thread, there is no reason to queue them behind one another.
AnalysisWorkflow.execute()
is a long, partly CPU-bound synchronous call, so it must run off the event
loop regardless; a plain thread per job is the simplest way to do that.

State is cached in memory for speed and mirrored to a small SQLite file
(<WORKER_OUTPUT_DIR>/jobs.db) so job *history* survives a worker restart. A
restart still kills any thread that was actually running - there is no way to
resume mid-AnalysisWorkflow.execute() - so any job still PENDING/RUNNING when
the process starts is marked FAILED with an explanatory message instead of
silently reverting to a 404 the backend's polling loop has to guess about.

`request_params`/`name`/`created_at` exist so a UI can show job *history*,
not just current status - what was actually requested for a past job, not
only its outcome.
"""

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import settings
from .runner import execute_and_persist

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_jobs: dict[str, "JobState"] = {}

_RESTART_ERROR = "Worker process restarted while this job was in progress; it did not complete."


@dataclass
class JobState:
    id: str
    status: str = "PENDING"  # PENDING | RUNNING | SUCCEEDED | FAILED
    progress: dict[str, Any] | None = None
    error_message: str | None = None
    result_dir: Path | None = None
    request_params: dict[str, Any] | None = None
    name: str | None = None
    created_at: str | None = None


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
    for column in ("request_params", "name", "created_at"):
        if column not in columns:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} TEXT")
    conn.commit()


def _db_write(state: "JobState") -> None:
    try:
        with _db_connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO jobs "
                "(id, status, progress, error_message, result_dir, request_params, name, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    state.id,
                    state.status,
                    json.dumps(state.progress) if state.progress is not None else None,
                    state.error_message,
                    str(state.result_dir) if state.result_dir else None,
                    json.dumps(state.request_params) if state.request_params is not None else None,
                    state.name,
                    state.created_at,
                ),
            )
    except Exception:
        logger.exception("Could not persist job %s to %s", state.id, _db_path())


def load_from_disk() -> None:
    """Load job history from disk at startup, failing out any job left
    PENDING/RUNNING by a previous process that never reached a terminal state."""
    if not _db_path().exists():
        return
    with _db_connect() as conn:
        rows = conn.execute(
            "SELECT id, status, progress, error_message, result_dir, request_params, name, created_at FROM jobs"
        ).fetchall()
        for job_id, status, progress, error_message, result_dir, request_params, name, created_at in rows:
            if status in ("PENDING", "RUNNING"):
                status, error_message = "FAILED", _RESTART_ERROR
            state = JobState(
                id=job_id,
                status=status,
                progress=json.loads(progress) if progress else None,
                error_message=error_message,
                result_dir=Path(result_dir) if result_dir else None,
                request_params=json.loads(request_params) if request_params else None,
                name=name,
                created_at=created_at,
            )
            _jobs[job_id] = state
            conn.execute(
                "UPDATE jobs SET status = ?, error_message = ? WHERE id = ?",
                (status, error_message, job_id),
            )
        conn.commit()
    logger.info("Loaded %d job(s) from %s", len(rows), _db_path())


def create(request_dict: dict, *, name: str | None = None) -> str:
    job_id = str(uuid.uuid4())
    state = JobState(
        id=job_id,
        request_params=request_dict,
        name=name,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    with _lock:
        _jobs[job_id] = state
    _db_write(state)
    thread = threading.Thread(target=_run, args=(job_id, request_dict), daemon=True)
    thread.start()
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


def _run(job_id: str, request_dict: dict) -> None:
    _update(job_id, status="RUNNING")
    try:
        result_dir = execute_and_persist(
            job_id,
            request_dict,
            output_root=Path(settings.worker_output_dir),
            progress_cb=lambda sensor, done, total: _update(
                job_id, progress={"sensor": sensor, "done": done, "total": total}
            ),
        )
        _update(job_id, status="SUCCEEDED", result_dir=result_dir)
        logger.info("Job %s succeeded", job_id)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the caller
        logger.exception("Job %s failed", job_id)
        _update(job_id, status="FAILED", error_message=str(exc))
