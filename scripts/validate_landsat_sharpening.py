"""How well does Sentinel-2 sharpen Landsat land surface temperature?

There is no independent 30 m thermal truth, so this follows Wald's protocol:
Landsat averaged to a truth resolution is degraded 3x, sharpened back with
Sentinel-2 averaged to the truth resolution, and compared with the truth.
At 90 m the truth is still blurred by the sensor (~100 m thermal
resolution, cubic resampling to 30 m), which favours smooth estimates, so
the test also runs with truths at 180 and 270 m. Each model and residual
correction is scored against two baselines that use no Sentinel-2: the
270 m value repeated, and bilinear interpolation. The same-sensor test is
an upper bound on real performance (the scale ratio is the same, 3, as the
operational 90 m -> 30 m step).

Then the best configuration sharpens the real 30 m product, and a figure
compares it with the delivered (resampled) one.

Writes ``output/landsat-sharpening-validation/results.json`` and
``sharpened.png``. Run one live-data script at a time:

    uv run --extra cdse --extra optical --extra cloud --extra landsat --extra ml --extra notebook \\
      python scripts/validate_landsat_sharpening.py

Environment variables: VALIDATION_START / VALIDATION_END (default
2026-08-01 / 2026-09-20), VALIDATION_CITY (default berlin).
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
import citycube as cc  # noqa: E402
from citycube.downscale.landsat import match_sentinel2  # noqa: E402

START = os.getenv("VALIDATION_START", "2026-08-01")
END = os.getenv("VALIDATION_END", "2026-09-20")
CITY = os.getenv("VALIDATION_CITY", "berlin")
OUTPUT = PROJECT_ROOT / "output" / "landsat-sharpening-validation"
WORK = PROJECT_ROOT / "output" / "landsat-sharpening-validation" / "work"
MIN_CLEAR = 0.4
MAX_GAP = np.timedelta64(5, "D")
# Truth at 90, 180 and 270 m (block factors of the 30 m product), each sharpened from 3x coarser.
TRUTH_FACTORS = (3, 6, 9)
MODELS = {
    "linear": {"model": "linear"},
    "random_forest": {"model": "random_forest"},
    "local_trees": {"model": "local_trees"},
}
CORRECTIONS = ("block", "smooth", "atpk")


def scores(estimate: np.ndarray, truth: np.ndarray) -> dict:
    both = np.isfinite(estimate) & np.isfinite(truth)
    e, t = estimate[both], truth[both]
    return {
        "cells": int(both.sum()),
        "rmse": float(np.sqrt(np.mean((e - t) ** 2))),
        "correlation": float(np.corrcoef(e, t)[0, 1]),
        "bias": float(np.mean(e - t)),
    }


def baselines(truth90: xr.DataArray) -> dict[str, np.ndarray]:
    """The 3x coarser value repeated, and bilinearly interpolated, on the truth grid."""

    coarse = cc.block_aggregate(truth90.to_dataset(name="v"), 3)["v"]
    repeated = coarse.reindex(y=truth90.y, x=truth90.x, method="nearest").values
    ordered = coarse.sortby("y").sortby("x")
    bilinear = ordered.interp(y=truth90.y, x=truth90.x).combine_first(ordered.reindex(y=truth90.y, x=truth90.x, method="nearest"))
    return {"repeated_coarse": repeated, "bilinear_coarse": bilinear.transpose("y", "x").values}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "urllib3", "rasterio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    acquire("validate_landsat_sharpening.py")
    started = time.time()
    cache = OUTPUT / "cache"
    if (cache / "landsat.zarr").exists() and (cache / "sentinel2.zarr").exists():
        landsat, sentinel2 = cc.open_zarr(cache / "landsat.zarr").load(), cc.open_zarr(cache / "sentinel2.zarr").load()
    else:
        request = cc.AnalysisRequest.for_city(cc.get_city(CITY), START, END, sensors=("landsat", "sentinel2"), resolution_m=30, max_products_per_sensor=40)
        request = dataclasses.replace(request, sentinel2_source="stac_cog", s2_cloud_cover_max=40, target_sensor="landsat",
                                      temporal_tolerances={"sentinel2": MAX_GAP})
        result = cc.AnalysisWorkflow(request).execute(WORK, max_workers=1)
        landsat = result.predictors["landsat"][["landsat_lst"]].load()
        sentinel2 = result.predictors["sentinel2"][["NDVI", "NDBI", "EVI", "NDMI"]].load()
        cc.write_zarr(landsat, cache / "landsat.zarr")
        cc.write_zarr(sentinel2, cache / "sentinel2.zarr")
    clear = landsat["landsat_lst"].notnull().mean(["y", "x"]).values >= MIN_CLEAR
    landsat = landsat.isel(time=np.flatnonzero(clear))
    logging.info("%d clear Landsat passes, %d Sentinel-2 acquisitions", landsat.sizes["time"], sentinel2.sizes["time"])

    report: dict = {"city": CITY, "period": [START, END], "landsat_passes": [str(t)[:16] for t in landsat.time.values], "wald": {}}
    for truth_factor in TRUTH_FACTORS:
        truth_m, coarse_m = 30 * truth_factor, 90 * truth_factor
        scale = f"{coarse_m}m->{truth_m}m"
        truth = cc.block_aggregate(landsat, truth_factor)
        predictors = cc.block_aggregate(sentinel2, truth_factor)
        entries: dict = {}
        for label, spec in MODELS.items():
            for correction in CORRECTIONS:
                name = f"{label}+{correction}"
                logging.info("Wald test %s: %s", scale, name)
                try:
                    sharpened = cc.sharpen_landsat(truth, predictors, correction=correction, min_samples=20, max_gap=MAX_GAP, **spec)
                except ValueError as exc:
                    entries[name] = {"error": str(exc)}
                    continue
                entries[name] = {
                    str(t)[:16]: scores(sharpened["lst_downscaled"].sel(time=t).values, truth["landsat_lst"].sel(time=t).values)
                    for t in sharpened.time.values
                }
        scored = sorted({t for entry in entries.values() for t in entry if t != "error"})
        for t in truth.time.values:
            if str(t)[:16] not in scored:
                continue  # baselines only on the passes the models were scored on
            reference = truth["landsat_lst"].sel(time=t)
            for name, values in baselines(reference).items():
                entries.setdefault(name, {})[str(t)[:16]] = scores(values, reference.values)
        report["wald"][scale] = entries

    def mean_rmse(entry: dict) -> float:
        values = [v["rmse"] for v in entry.values() if isinstance(v, dict) and "rmse" in v]
        return float(np.mean(values)) if values else float("inf")

    report["wald_mean_rmse"] = {
        scale: dict(sorted(((name, round(mean_rmse(entry), 3)) for name, entry in entries.items()), key=lambda item: item[1]))
        for scale, entries in report["wald"].items()
    }
    pooled = {name: float(np.mean([report["wald_mean_rmse"][scale][name] for scale in report["wald_mean_rmse"]]))
              for name in next(iter(report["wald_mean_rmse"].values())) if "+" in name}
    best = min(pooled, key=lambda name: pooled[name])
    report["best"] = best

    # The real step: 90 m blocks of the delivered product -> 30 m.
    model, correction = best.split("+")
    sharpened30 = cc.sharpen_landsat(landsat, sentinel2, correction=correction, min_samples=30, max_gap=MAX_GAP, **MODELS[model])
    report["sharpened_30m_scenes"] = int(sharpened30.sizes["time"])
    report["seconds"] = round(time.time() - started)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "results.json").write_text(json.dumps(report, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scene = int(sharpened30["lst_downscaled"].notnull().mean(["y", "x"]).argmax())
    t = sharpened30.time.values[scene]
    delivered = landsat["landsat_lst"].sel(time=t)
    ny, nx = delivered.sizes["y"], delivered.sizes["x"]
    window = {"y": slice(ny // 2 - 100, ny // 2 + 100), "x": slice(nx // 2 - 100, nx // 2 + 100)}  # 6 x 6 km in the centre
    panels = [
        (delivered.isel(**window), "Landsat as delivered, 30 m (resampled from ~100 m)"),
        (sharpened30["lst_downscaled"].sel(time=t).isel(**window), f"sharpened with Sentinel-2, 30 m ({best})"),
        (sentinel2["NDVI"].sel(time=match_sentinel2(np.array([t]), sentinel2, max_gap=np.timedelta64(3, "D"))[0]).isel(**window), "Sentinel-2 NDVI"),
    ]
    low, high = np.nanpercentile(panels[0][0].values, [2, 98])
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
    for ax, (data, title) in zip(axes, panels):
        is_ndvi = "NDVI" in title
        image = ax.imshow(data.values, cmap="RdYlGn" if is_ndvi else "inferno", vmin=None if is_ndvi else low, vmax=None if is_ndvi else high, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        plt.colorbar(image, ax=ax, shrink=0.8, label="" if is_ndvi else "degC")
    fig.suptitle(f"{CITY.title()}, {np.datetime_as_string(t, unit='m')} UTC, 6 x 6 km", fontsize=11)
    fig.savefig(OUTPUT / "sharpened.png", dpi=110, bbox_inches="tight")

    print(f"\n{CITY} {START}..{END}: {landsat.sizes['time']} clear Landsat passes")
    for scale, table in report["wald_mean_rmse"].items():
        print(f"Wald test {scale}, mean RMSE over the scored passes (K):")
        for name, value in table.items():
            print(f"  {name:24s} {value:.3f}")
    print(f"best: {best}; sharpened {report['sharpened_30m_scenes']} scenes to 30 m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
