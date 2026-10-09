"""Jobs in real spawned child processes: outcome, cancellation, timeout,
out-of-memory kills, retention and deletion."""

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app import jobs
from app.config import settings

REQUEST = {"aoi": {"west": 13.3, "south": 52.4, "east": 13.5, "north": 52.6}, "start": "2024-06-01", "end": "2024-06-02"}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setattr(settings, "job_process_isolation", True)
    monkeypatch.setattr(jobs, "_TERMINATE_GRACE_S", 3)


def _use(monkeypatch, target: str) -> None:
    monkeypatch.setattr(jobs, "JOB_RUNNER", f"job_targets:{target}")


def _wait(job_id: str, predicate, timeout: float = 60.0) -> jobs.JobState:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = jobs.get(job_id)
        if state is not None and predicate(state):
            return state
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not reach the expected state: {jobs.get(job_id)}")


def _finished(job_id: str) -> jobs.JobState:
    return _wait(job_id, lambda state: state.status in jobs.TERMINAL)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A killed child of an exited process may linger as a zombie briefly.
    stat = Path(f"/proc/{pid}/stat")
    return not (stat.exists() and stat.read_text().split()[2] == "Z")


def test_successful_job_reports_progress_metrics_and_drops_its_work_dir(monkeypatch):
    _use(monkeypatch, "succeed")

    state = _finished(jobs.create(REQUEST))

    assert state.status == jobs.SUCCEEDED, state.error_message
    assert state.progress == {"sensor": "sentinel3", "done": 2, "total": 2}
    assert state.metrics["failed_products"] == 1
    assert state.metrics["peak_rss_mb"] > 0 and state.metrics["duration_s"] >= 0
    assert state.started_at and state.finished_at
    assert state.result_dir is not None and state.result_dir.is_dir()
    assert not (jobs.job_dir(state.id) / "work").exists()


def test_failure_in_the_child_is_reported_with_its_log(monkeypatch):
    _use(monkeypatch, "fail")

    state = _finished(jobs.create(REQUEST))

    assert state.status == jobs.FAILED
    assert state.error_message == "ValueError: bad scene"
    assert "bad scene" in jobs.log_path(state.id).read_text()


def test_cancelling_a_running_job_stops_it_and_everything_it_started(monkeypatch):
    _use(monkeypatch, "sleep_with_grandchild")
    job_id = jobs.create(REQUEST)
    marker = Path(settings.worker_output_dir) / f"{job_id}.grandchild"
    _wait(job_id, lambda state: state.progress is not None and marker.exists())
    grandchild = int(marker.read_text())

    assert jobs.cancel(job_id) == jobs.RUNNING
    state = _finished(job_id)

    assert state.status == jobs.CANCELLED
    deadline = time.monotonic() + 10
    while _alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _alive(grandchild)


def test_a_job_past_its_timeout_is_stopped(monkeypatch):
    _use(monkeypatch, "sleep_with_grandchild")
    monkeypatch.setattr(settings, "job_timeout_hours", 2 / 3600)

    state = _finished(jobs.create(REQUEST))

    assert state.status == jobs.FAILED
    assert "Timed out" in (state.error_message or "")


def test_an_out_of_memory_kill_fails_only_that_job(monkeypatch):
    _use(monkeypatch, "killed")

    state = _finished(jobs.create(REQUEST))

    assert state.status == jobs.FAILED
    assert "out-of-memory" in (state.error_message or "")


def test_cancel_and_delete_rules(monkeypatch):
    _use(monkeypatch, "fail")
    job_id = jobs.create(REQUEST)
    _finished(job_id)

    with pytest.raises(jobs.JobConflictError):
        jobs.cancel(job_id)
    jobs.delete(job_id)
    assert jobs.get(job_id) is None
    assert not jobs.job_dir(job_id).exists()
    with pytest.raises(KeyError):
        jobs.delete(job_id)


def test_retention_deletes_only_finished_jobs_older_than_the_limit(monkeypatch):
    _use(monkeypatch, "fail")
    monkeypatch.setattr(settings, "job_retention_days", 30)
    old, recent = jobs.create(REQUEST), jobs.create(REQUEST)
    _finished(old)
    _finished(recent)
    jobs._update(old, finished_at=(datetime.now(timezone.utc) - timedelta(days=31)).isoformat())

    assert jobs.prune_expired() == [old]
    assert jobs.get(old) is None and jobs.get(recent) is not None
