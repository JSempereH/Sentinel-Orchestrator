"""Stand-ins for execute_and_persist, run inside real spawned job processes
by test_isolation.py (resolved by name in the child, see jobs.JOB_RUNNER)."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def succeed(job_id, request_dict, *, output_root, progress_cb):
    progress_cb("sentinel3", 1, 2)
    progress_cb("sentinel3", 2, 2)
    result = Path(output_root) / job_id / "result"
    result.mkdir(parents=True)
    (result / "provenance.json").write_text(json.dumps({"failed_products": [{"product": "x"}]}))
    (Path(output_root) / job_id / "work").mkdir()
    return result


def fail(job_id, request_dict, *, output_root, progress_cb):
    raise ValueError("bad scene")


def sleep_with_grandchild(job_id, request_dict, *, output_root, progress_cb):
    # Stands in for SNAP's gpt: a separate process the job started, which a
    # cancellation must stop too.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    (Path(output_root) / f"{job_id}.grandchild").write_text(str(child.pid))
    progress_cb("sentinel1", 0, 1)
    time.sleep(600)


def killed(job_id, request_dict, *, output_root, progress_cb):
    os.kill(os.getpid(), signal.SIGKILL)  # what the out-of-memory killer does
