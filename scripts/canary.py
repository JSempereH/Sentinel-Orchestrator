"""Daily end-to-end check against live data: are credentials and providers still working?

Unit tests never touch the real catalogues, so an expired credential or a
provider-side API change is otherwise only noticed when a real job fails.
This script runs the smallest real analysis that exercises the main path
and exits non-zero on any failure, so cron (MAILTO) or a systemd timer
(OnFailure=) can alert on it:

1. every configured credential authenticates (`check_credentials`);
2. up to four daytime Sentinel-3 LST products over a ~10 x 10 km AOI in
   central Berlin are discovered, partially downloaded (~17 MB each),
   gridded and fused. At least one clear cell must come out with a valid
   temperature. If every observed cell is flagged (cloud or quality), the
   run is "inconclusive" (exit 0, recorded with a warning): the pipeline
   worked and the sky did not cooperate. Observed cells that are neither
   valid nor flagged mean a reader regression and fail the run;
3. with --full, also Sentinel-2 L2A COGs read in place (Planetary Computer)
   and per-scene linear downscaling onto a 100 m grid.

Each run appends one JSON line to output/canary/history.jsonl. Work files go
to a temporary directory that is always removed.

    uv run --extra cdse --extra optical --extra cloud python scripts/canary.py [--full]

Example crontab entry (06:00 every day, mail on failure):

    MAILTO=you@example.org
    0 6 * * * cd /path/to/Sentinel-Orchestrator && timeout 1800 uv run --extra cdse --extra optical --extra cloud python scripts/canary.py > /dev/null

The canary uses short network timeouts (2 minutes per read, 2 download
attempts) so a stalled provider fails the check instead of hanging it;
`timeout` bounds the run as a whole.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _liveguard import acquire  # noqa: E402
from sentinel_analysis import AOI, AnalysisGrid, AnalysisRequest, AnalysisWorkflow, DownscaleSpec, build_info, check_credentials  # noqa: E402

HISTORY = PROJECT_ROOT / "output" / "canary" / "history.jsonl"
# Central Berlin: flat, frequently clear in summer, one Sentinel-3 tile.
AOI_BERLIN = AOI(west=13.33, south=52.47, east=13.47, north=52.56)
# Sentinel-3 revisits daily; ten days nearly always contain a daytime pass.
LOOKBACK_DAYS = 10


def build_request(full: bool) -> AnalysisRequest:
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=LOOKBACK_DAYS)
    predictor = AnalysisGrid.for_aoi(AOI_BERLIN, resolution_m=100)
    thermal = AnalysisGrid.for_aoi(AOI_BERLIN, resolution_m=1000, crs=predictor.crs)
    return AnalysisRequest(
        aoi=AOI_BERLIN,
        start=start.isoformat(),
        end=end.isoformat(),
        sensors=("sentinel3", "sentinel2") if full else ("sentinel3",),
        grid=predictor,
        predictor_grid=predictor,
        thermal_grid=thermal,
        max_products_per_sensor=4,
        thermal_overpass="day",
        sentinel2_source="stac_cog",
        s2_cloud_cover_max=60,
        temporal_tolerance=np.timedelta64(LOOKBACK_DAYS, "D"),
        downscale=DownscaleSpec(model="linear", predictors=("NDVI", "NDBI"), min_samples=10) if full else None,
    )


def run(full: bool) -> dict:
    record: dict = {"started_at": datetime.now(timezone.utc).isoformat(), "full": full, "software": build_info(), "checks": {}}
    started = time.monotonic()

    credentials = check_credentials(PROJECT_ROOT / ".env")
    record["checks"]["credentials"] = {name: result.to_dict() for name, result in credentials.items()}
    failed = [name for name, result in credentials.items() if result.failed]
    if failed:
        record["error"] = f"credentials failed: {', '.join(failed)}"
        return record

    with tempfile.TemporaryDirectory(prefix="sentinel-canary-") as work:
        result = AnalysisWorkflow(build_request(full)).execute(work)
        lst = result.cube["lst"]
        thermal = result.thermal_cube
        finite = int(np.isfinite(lst.values).sum())
        observed = int((thermal["source_footprint_count"].values > 0).sum()) if thermal is not None else 0
        flagged = int((thermal["lst_invalid_observation_count"].values > 0).sum()) if thermal is not None else 0
        record["checks"]["fusion"] = {
            "times": int(lst.sizes["time"]),
            "observed_cells": observed,
            "flagged_cells": flagged,
            "finite_lst_cells": finite,
            "lst_median_c": round(float(np.nanmedian(lst.values)), 2) if finite else None,
            "failed_products": result.provenance.get("failed_products", []),
        }
        if not finite and observed and flagged < observed:
            record["error"] = f"{observed - flagged} observed cells are neither valid nor flagged: a reader regression"
        elif not finite:
            record["warning"] = "inconclusive: every observed cell was cloud- or quality-flagged" if observed else "inconclusive: no product covered the AOI"
        if full:
            downscaled = result.downscaled
            record["checks"]["downscaling"] = None if downscaled is None else {
                "scenes": int(downscaled.sizes["time"]),
                "finite_fine_cells": int(np.isfinite(downscaled["lst_downscaled"].values).sum()),
            }
    record["duration_s"] = round(time.monotonic() - started, 1)
    return record


# A health check must fail fast: the library's defaults (600 s per read, 6
# attempts) suit long analyses, not a daily probe. Explicit settings win.
os.environ.setdefault("CDSE_READ_TIMEOUT_S", "120")
os.environ.setdefault("CDSE_DOWNLOAD_RETRIES", "2")


def main() -> int:
    # Progress on stderr, so a hung or failed run shows which step it reached.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    for noisy in ("httpx", "openeo", "urllib3", "rasterio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Daily end-to-end check against live data.")
    parser.add_argument("--full", action="store_true", help="also read Sentinel-2 COGs and downscale")
    args = parser.parse_args()
    acquire("canary.py")
    try:
        record = run(args.full)
    except Exception as exc:  # noqa: BLE001 - every failure must be recorded and alerted on
        record = {"started_at": datetime.now(timezone.utc).isoformat(), "full": args.full, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    record["ok"] = "error" not in record
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
    print(json.dumps(record, indent=2, default=str))
    if not record["ok"]:
        print(f"CANARY FAILED: {record['error']}", file=sys.stderr)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
