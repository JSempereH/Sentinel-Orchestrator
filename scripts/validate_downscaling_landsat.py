"""Does per-scene downscaling beat the 1 km observation against Landsat?

Fetches Sentinel-3 (daytime, cloud-probed), Sentinel-2 and Landsat 8/9 over
Berlin, then for each downscaling model:

* compares the 100 m result with every Landsat scene of the same morning
  (Landsat is never used for training) against the baseline of repeating
  the 1 km Sentinel-3 value: RMSE, correlation, and the error after removing
  each map's mean (spatial pattern only);
* scores it on a blocked holdout (5 x 5 km checkerboard hidden from training,
  scored at 1 km).

Writes ``output/downscaling-landsat-validation/results.json`` and prints a
table. Run one live-data script at a time:

    uv run --extra cdse --extra optical --extra cloud --extra landsat --extra ml \\
      python scripts/validate_downscaling_landsat.py

Environment variables: VALIDATION_START / VALIDATION_END (default
2026-08-01 / 2026-08-21), VALIDATION_CITY (default berlin).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _liveguard import acquire  # noqa: E402
import sentinel_analysis as sa  # noqa: E402

START = os.getenv("VALIDATION_START", "2026-08-01")
END = os.getenv("VALIDATION_END", "2026-08-21")
CITY = os.getenv("VALIDATION_CITY", "berlin")
OUTPUT = PROJECT_ROOT / "output" / "downscaling-landsat-validation"
WORK = PROJECT_ROOT / "output" / "notebooks" / "work"  # shares the guide notebooks' cache
MODELS = {
    "linear": {"model": "linear"},
    "random_forest": {"model": "random_forest"},
    "local_trees_w5": {"model": "local_trees", "model_options": {"window": 5}},
    "local_trees_w10": {"model": "local_trees", "model_options": {"window": 10}},
    "local_trees_w15": {"model": "local_trees", "model_options": {"window": 15}},
}
BLOCK = 5


def landsat_scores(result, downscaled) -> list[dict]:
    """Downscaled and repeated-1-km maps against each same-morning Landsat scene."""
    landsat = result.predictors.get("landsat")
    if landsat is None:
        return []
    rows = []
    for t_landsat in landsat.time.values:
        reference = landsat["landsat_lst"].sel(time=t_landsat)
        if float(reference.notnull().mean()) < 0.3:
            continue
        nearest = downscaled.time.values[np.argmin(np.abs(downscaled.time.values - t_landsat))]
        if abs(nearest - t_landsat) > np.timedelta64(3, "h"):
            continue
        for name, estimate in (
            ("downscaled", downscaled["lst_downscaled"].sel(time=nearest)),
            ("repeated_1km", result.cube["lst"].sel(time=nearest).reindex(y=reference.y, x=reference.x, method="nearest")),
        ):
            both = np.isfinite(estimate.values) & np.isfinite(reference.values)
            if both.sum() < 100:
                continue
            e, r = estimate.values[both], reference.values[both]
            rows.append({
                "landsat": str(t_landsat)[:16], "estimate": name, "pixels": int(both.sum()),
                "rmse": float(np.sqrt(np.mean((e - r) ** 2))), "correlation": float(np.corrcoef(e, r)[0, 1]),
                "pattern_rmse": float(np.sqrt(np.mean(((e - e.mean()) - (r - r.mean())) ** 2))),
            })
    return rows


def holdout_rmse(result, spec: dict) -> float:
    cube = result.cube
    rows, cols = np.indices((cube.sizes["y"], cube.sizes["x"]))
    hidden = xr.DataArray(((rows // BLOCK + cols // BLOCK) % 2 == 1), dims=("y", "x"), coords={"y": cube.y, "x": cube.x})
    training = cube.assign(lst=cube["lst"].where(~hidden))
    fine = sa.downscale_per_scene(training, result.predictors["sentinel2"], terrain=result.terrain, min_samples=20, mask_unobserved=False, **spec)
    aggregated = sa.reaggregate_to_target(fine["lst_downscaled"].assign_attrs(crs=cube.attrs["crs"]), cube["lst"].assign_attrs(crs=cube.attrs["crs"]))
    return float(sa.compare_to_reference(aggregated.where(hidden), cube["lst"].sel(time=aggregated.time).where(hidden))["rmse"])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "openeo", "urllib3", "rasterio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    acquire("validate_downscaling_landsat.py")
    city = sa.get_city(CITY)
    request = sa.AnalysisRequest.for_city(city, START, END, sensors=("sentinel3", "sentinel2", "landsat"), resolution_m=100, max_products_per_sensor=20)
    request = dataclasses.replace(
        request, sentinel2_source="stac_cog", s2_cloud_cover_max=60, thermal_overpass="day", terrain_predictors=True,
        min_clear_fraction=0.3, temporal_tolerances={"landsat": np.timedelta64(1, "D")},
    )
    started = time.time()
    result = sa.AnalysisWorkflow(request).execute(WORK, max_workers=1)
    report: dict = {"city": CITY, "period": [START, END], "scenes": int(result.cube.sizes["time"]), "probed_out": len(result.provenance.get("probed_out", [])), "models": {}}
    for label, spec in MODELS.items():
        logging.info("evaluating %s", label)
        downscaled = sa.downscale_per_scene(result.cube, result.predictors["sentinel2"], terrain=result.terrain, **spec)
        report["models"][label] = {"landsat": landsat_scores(result, downscaled), "holdout_rmse": holdout_rmse(result, spec)}
    report["seconds"] = round(time.time() - started)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "results.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"\n{CITY} {START}..{END}: {report['scenes']} scenes, {report['probed_out']} rejected by the cloud probe")
    print(f"{'model':18s} {'holdout':>8s}   Landsat: downscaled vs repeated 1 km (rmse | r | pattern rmse)")
    for label, scores in report["models"].items():
        pairs = {}
        for row in scores["landsat"]:
            pairs.setdefault(row["landsat"], {})[row["estimate"]] = row
        cells = [f"{d['downscaled']['rmse']:.2f}/{d['repeated_1km']['rmse']:.2f} | {d['downscaled']['correlation']:.2f}/{d['repeated_1km']['correlation']:.2f} | {d['downscaled']['pattern_rmse']:.2f}/{d['repeated_1km']['pattern_rmse']:.2f}"
                 for d in pairs.values() if {"downscaled", "repeated_1km"} <= set(d)]
        print(f"{label:18s} {scores['holdout_rmse']:8.2f}   " + ("; ".join(cells) or "no same-morning Landsat scene"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
