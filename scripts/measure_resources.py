"""Time, peak memory and disk of real citycube runs, one scenario per process.

Each scenario runs a full workflow and prints one JSON line (scenario, wall
seconds, peak resident memory, subset cache size, output sizes). Run each
scenario in its own process under a memory cap, one at a time, so the peak
is that scenario's alone:

    systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \\
      uv run --extra cdse --extra optical --extra cloud --extra landsat --extra ml \\
      python scripts/measure_resources.py core-100m >> output/resources/results.jsonl

Scenarios (Berlin, daytime Sentinel-3 with the cloud probe, Sentinel-2 COGs):

* ``core-100m``: the built-in Berlin AOI (24 x 21 km), 1-21 August 2026,
  Sentinel-3 + Sentinel-2 at 100 m with per-scene downscaling. Run it twice
  to measure a cold and a warm cache.
* ``city-100m``: the whole city boundary (46 x 38 km, ~3.5x the area), same.
* ``year-YYYY``: the core AOI, 1-21 August of that year (the multi-year use).
* ``core-30m-landsat``: Landsat + Sentinel-2 at 30 m, 1 August to 20
  September 2026, then ``sharpen_landsat``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import citycube as cc  # noqa: E402

WORK = PROJECT_ROOT / "output" / "resources" / "work"
CITY_BOUNDARY = PROJECT_ROOT / "output" / "notebooks" / "berlin_districts.geojson"  # written by notebook 08


def _size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def _thermal_request(aoi_city: cc.CitySpec, start: str, end: str) -> cc.AnalysisRequest:
    request = cc.AnalysisRequest.for_city(aoi_city, start, end, sensors=("sentinel3", "sentinel2"), resolution_m=100, max_products_per_sensor=20)
    return dataclasses.replace(
        request, sentinel2_source="stac_cog", s2_cloud_cover_max=60, thermal_overpass="day",
        min_clear_fraction=0.3, terrain_predictors=True, downscale=cc.DownscaleSpec(),
    )


def _city_boundary() -> cc.CitySpec:
    import shapely
    from shapely.geometry import mapping, shape

    districts = json.loads(CITY_BOUNDARY.read_text(encoding="utf-8"))
    outline = shapely.simplify(shapely.unary_union([shape(f["geometry"]) for f in districts["features"]]), 0.0005)
    aoi = cc.AOI.from_geometry(mapping(outline))
    berlin = cc.get_city("berlin")
    return dataclasses.replace(berlin, city_id="berlin-city", aoi=aoi)


def run(scenario: str) -> dict:
    started = time.time()
    extra: dict = {}
    if scenario == "core-100m":
        request = _thermal_request(cc.get_city("berlin"), "2026-08-01", "2026-08-21")
    elif scenario == "city-100m":
        request = _thermal_request(_city_boundary(), "2026-08-01", "2026-08-21")
    elif scenario.startswith("year-"):
        year = scenario.removeprefix("year-")
        request = _thermal_request(cc.get_city("berlin"), f"{year}-08-01", f"{year}-08-21")
    elif scenario == "core-30m-landsat":
        request = cc.AnalysisRequest.for_city(cc.get_city("berlin"), "2026-08-01", "2026-09-20", sensors=("landsat", "sentinel2"), resolution_m=30, max_products_per_sensor=40)
        request = dataclasses.replace(request, sentinel2_source="stac_cog", s2_cloud_cover_max=40, target_sensor="landsat",
                                      temporal_tolerances={"sentinel2": np.timedelta64(5, "D")})
    else:
        raise SystemExit(f"unknown scenario {scenario!r}")

    estimate = cc.estimate_request(request)
    output = WORK / "runs" / scenario
    shutil.rmtree(output / "result", ignore_errors=True)
    result = cc.AnalysisWorkflow(request).execute(WORK, max_workers=1)
    acquired = time.time()
    if scenario == "core-30m-landsat":
        sharpened = cc.sharpen_landsat(result.predictors["landsat"], result.predictors["sentinel2"], max_gap=np.timedelta64(5, "D"))
        extra["sharpened_scenes"] = int(sharpened.sizes["time"])
    saved = result.save(output / "result")
    finished = time.time()
    return {
        "scenario": scenario,
        "aoi_km2": round(estimate.aoi_km2, 1),
        "estimated_gb": round(estimate.estimated_gb, 3),
        "thermal_scenes": int(result.cube.sizes["time"]),
        "downscaled_scenes": int(result.downscaled.sizes["time"]) if result.downscaled is not None else 0,
        "acquire_and_process_s": round(acquired - started),
        "total_s": round(finished - started),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "subset_cache_mb": round(_size(WORK / "subsets") / 1e6),
        "result_mb": round(_size(saved) / 1e6),
        **extra,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    for noisy in ("httpx", "urllib3", "rasterio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    print(json.dumps(run(sys.argv[1])), flush=True)
