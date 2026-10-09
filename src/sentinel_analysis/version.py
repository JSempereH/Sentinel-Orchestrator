"""Version and build identification recorded in every result's provenance.

A result is only reproducible if it says which code produced it: the
package version alone does not change between commits, so the git commit is
recorded too. Inside a container there is no ``.git``; the image passes the
commit in ``SENTINEL_ANALYSIS_GIT_COMMIT`` at build time instead.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import subprocess

COMMIT_ENV = "SENTINEL_ANALYSIS_GIT_COMMIT"

try:
    __version__ = version("sentinel-analysis")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"


@lru_cache(maxsize=1)
def build_info() -> dict[str, str | bool | None]:
    """Package version, git commit and whether the working tree had local changes.

    ``git_dirty`` is ``None`` when it cannot be determined (no git checkout).
    """

    commit = os.getenv(COMMIT_ENV) or None
    dirty: bool | None = None
    if commit is None:
        root = Path(__file__).resolve().parents[2]
        if (root / ".git").exists():
            commit = _git(root, "rev-parse", "HEAD")
            status = _git(root, "status", "--porcelain", "--untracked-files=no")
            dirty = None if status is None else bool(status)
    return {"version": __version__, "git_commit": commit, "git_dirty": dirty}


def _git(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()
