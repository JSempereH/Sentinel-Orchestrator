"""Logging for the API process and for each job's child process.

LOG_FORMAT=json emits one JSON object per line (for journald, Loki and
similar); the default is human-readable text. Job processes additionally
write their own log to <job dir>/job.log, which GET /jobs/{id}/log serves.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import settings

TEXT_FORMAT = "%(asctime)s %(levelname)s [%(name)s]%(job)s %(message)s"
# Third-party loggers that write credentials at INFO level: openeo's OIDC
# client logs the client id of every token request. Job logs are stored on
# disk and served by GET /jobs/{id}/log, so these stay at WARNING.
CREDENTIAL_LOGGERS = ("openeo.rest.auth",)


class _JobFilter(logging.Filter):
    """Adds the job id to every record emitted in a job's process."""

    def __init__(self, job_id: str | None) -> None:
        super().__init__()
        self.job_id = job_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = self.job_id
        record.job = f" [job {self.job_id[:8]}]" if self.job_id else ""
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        job_id = getattr(record, "job_id", None)
        if job_id:
            payload["job_id"] = job_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _formatter() -> logging.Formatter:
    return JsonFormatter() if settings.log_format == "json" else logging.Formatter(TEXT_FORMAT)


def _handler(handler: logging.Handler, job_id: str | None) -> logging.Handler:
    handler.setFormatter(_formatter())
    handler.addFilter(_JobFilter(job_id))
    return handler


def _quiet_credential_loggers() -> None:
    for name in CREDENTIAL_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def configure_logging() -> None:
    """Configure the API process's root logger (stderr)."""

    root = logging.getLogger()
    root.handlers[:] = [_handler(logging.StreamHandler(), None)]
    root.setLevel(logging.INFO)
    _quiet_credential_loggers()


def configure_job_logging(job_id: str, path: Path) -> None:
    """Configure a job process to log to stderr and to its own job.log."""

    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers[:] = [_handler(logging.StreamHandler(), job_id), _handler(logging.FileHandler(path, encoding="utf-8"), job_id)]
    root.setLevel(logging.INFO)
    _quiet_credential_loggers()
