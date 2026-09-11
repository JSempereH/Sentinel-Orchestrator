"""
Central worker configuration via pydantic-settings.

Credentials for sentinel_analysis itself (CDSE, CDS/ADS, OpenAQ) are NOT
read here - they stay local to this machine and are read directly by
sentinel_analysis.config.ClientConfig.from_env() / the auxiliary providers,
exactly as when running the sentinel-analysis CLI by hand. This file only
configures the worker's own HTTP-facing behaviour.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


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

    class Config:
        env_file = ".env"
        # .env also carries CDSE/CDS/CAMS/OpenAQ credentials read directly by
        # sentinel_analysis (see module docstring) - pydantic-settings must
        # not reject the file for containing fields this model doesn't define.
        extra = "ignore"


settings = Settings()
Path(settings.worker_output_dir).mkdir(parents=True, exist_ok=True)
