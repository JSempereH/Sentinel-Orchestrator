"""Shared safety guard for scripts that run real, memory-heavy multi-product
fetches (SNAP/xarray/dask over real Sentinel/Landsat data).

This machine has frozen twice running two such scripts concurrently, each
consuming several GB with no OOM-kill logged in dmesg/journalctl - the
desktop session restarting immediately after was the only signal (see
docs/roadmap.md, "This machine froze twice"). `acquire()` refuses to start a
second live-data script instead of repeating that, rather than relying on
whoever is running these to remember the rule.
"""

from __future__ import annotations

import atexit
import os
import sys
from pathlib import Path

LOCK_PATH = Path(__file__).resolve().parent.parent / "output" / ".live-script.lock"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists, just owned by someone else
    return True


def acquire(script_name: str) -> None:
    """Exit the process if another live-data script already holds the lock;
    otherwise take it and release it automatically on exit.

    Call this once, at the very top of a script's real work (before any
    download/fetch begins) - not at import time, so dry-run/--help paths
    that never touch real data are unaffected.
    """
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        try:
            holder_pid_str, holder_name = LOCK_PATH.read_text().strip().split(":", 1)
            holder_pid = int(holder_pid_str)
        except (ValueError, OSError):
            holder_pid, holder_name = None, "unknown"
        if holder_pid is not None and _pid_alive(holder_pid):
            print(
                f"error: {holder_name} (pid {holder_pid}) is already running a live-data "
                f"script. Running two of these at once has frozen this machine before "
                f"(see docs/roadmap.md, \"This machine froze twice\") - wait for it to "
                f"finish before starting {script_name}.",
                file=sys.stderr,
            )
            sys.exit(1)
        # holder process is gone - stale lock, safe to overwrite below.
    LOCK_PATH.write_text(f"{os.getpid()}:{script_name}")
    atexit.register(lambda: LOCK_PATH.unlink(missing_ok=True))
