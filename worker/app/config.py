"""
Central worker configuration via pydantic-settings.

Credentials for citycube itself (CDSE, CDS/ADS, OpenAQ) are NOT
read here - they stay local to this machine and are read directly by
citycube.config.ClientConfig.from_env() / the auxiliary providers,
exactly as when running the citycube CLI by hand. This file only
configures the worker's own HTTP-facing behaviour.
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Shared secret every caller (the worker's own frontend, a script, curl)
    # must send as "Authorization: Bearer <token>". Empty means every request
    # is rejected - a worker must be explicitly configured before it accepts jobs.
    worker_api_token: str = ""

    # Where per-job downloads and result cubes are written.
    worker_output_dir: str = "./runs"

    # Max parallel product downloads within a single job (passed through to
    # AnalysisWorkflow.execute(max_workers=...)).
    max_download_workers: int = 4

    # Max jobs executing at once; further submissions wait as PENDING. Each
    # job can need several GB of RAM (SNAP, full-scene reads), so the safe
    # default is strictly serial.
    max_concurrent_jobs: int = 1

    # Requests beyond these are rejected at submission (HTTP 422) instead of
    # failing mid-run or exhausting the machine's memory. The estimate is the
    # in-memory size of the prepared cubes (see
    # citycube.workflow.limits); peak RAM is up to about twice that.
    # 0 disables a limit.
    max_aoi_km2: float = 10000
    max_products_per_sensor: int = 100
    max_estimated_gb: float = 8

    # Run each job in its own child process: required for cancelling a
    # running job and for JOB_TIMEOUT_HOURS, and keeps an out-of-memory kill
    # from taking the API down with it. Only tests turn it off.
    job_process_isolation: bool = True

    # A running job is stopped after this many hours (0 disables).
    job_timeout_hours: float = 12

    # Finished jobs (and all their files) are deleted after this many days
    # (0 keeps them forever).
    job_retention_days: float = 30

    # Downloads, extracted archives and AOI subsets under <job>/work are
    # deleted once a job succeeds; only result/ and job.log are kept.
    # Failed jobs keep work/ for diagnosis until retention removes them.
    keep_work_dir: bool = False

    # New jobs are refused (HTTP 507) and queued jobs fail instead of
    # starting below this much free space in WORKER_OUTPUT_DIR (0 disables).
    min_free_disk_gb: float = 20

    # "text" or "json" (one object per line, for log collectors).
    log_format: Literal["text", "json"] = "text"

    # .env also carries CDSE/CDS/CAMS/OpenAQ credentials read directly by
    # citycube (see module docstring) - pydantic-settings must not
    # reject the file for containing fields this model doesn't define.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
Path(settings.worker_output_dir).mkdir(parents=True, exist_ok=True)
