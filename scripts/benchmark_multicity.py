"""Multi-city, multi-season downscaling benchmark, including pyDMS.

`scripts/benchmark_downscalers.py` compared models on one city and one
period, and its ranking flipped between a one-week and a four-week run.
This script widens the evidence before any further model work:

* every preset city (Berlin, Guadalajara, Mexico City, Lagos, Nairobi) in
  two periods (default: early August and early April 2026);
* daytime Sentinel-3 passes only (local solar time 08-16 h), filtered
  *before* download - the NDVI/LST relation these models learn is a daytime
  one, and night passes dilute it;
* a genuinely spatial holdout: `blocked_spatiotemporal_split(block_size=5)`
  holds out 5 x 5 km blocks (the default single-pixel diagonal lets each
  held-out cell's direct neighbours sit in training) from the latest 30 %
  of acquisitions;
* pyDMS (Guzinski's Data Mining Sharpener, the closest external
  Sentinel-3 -> Sentinel-2 sharpening implementation) as an external
  reference, trained on the same training cells and times;
* conformal 90 % intervals (`fit_conformal_downscaler`) calibrated on the
  training-time held-out blocks, with their coverage on the validation set.

Two protocols are reported per model:

* ``coarse``: `validate_downscaler` on the 1 km validation cube (predictors
  aggregated to 1 km) - the existing benchmark's protocol.
* ``fine_to_coarse``: predict at 100 m from the matching Sentinel-2
  predictors, reaggregate to 1 km and compare with the held-out observed
  LST. This is what a downscaler is for, and the only protocol pyDMS
  (which always predicts at the high-resolution grid) can be scored on.
  pyDMS's local/global combination and residual correction are *not* used:
  both read the observed coarse LST of the scene being predicted, which
  would leak the validation target.

GWR is scored on the coarse protocol only: its per-pixel local fits make a
100 m prediction over a whole city take hours.

pyDMS needs GDAL's Python bindings built against numpy 2, which the project
environment does not have; run this from an environment that does, e.g.:

    uv venv --python /usr/bin/python3 /tmp/dmsenv
    uv pip install --python /tmp/dmsenv -e ".[optical,ml,cloud,landsat]" python-dms setuptools wheel
    uv pip install --python /tmp/dmsenv --no-build-isolation --no-deps --no-cache "gdal==$(gdal-config --version).*"
    /tmp/dmsenv/bin/python scripts/benchmark_multicity.py

Without pyDMS the script still runs and reports the remaining models.

Environment variables:
    BENCHMARK_CITIES          comma-separated city ids, default all presets
    BENCHMARK_WINDOWS         comma-separated start/end pairs,
                              default 2026-08-01/2026-08-14,2026-04-01/2026-04-14
    BENCHMARK_MAX_PRODUCTS    default 24 (per sensor, after the daytime filter)
    BENCHMARK_OUTPUT          default output/multicity-benchmark
    BENCHMARK_RESULTS         results file name inside the output dir, default results.json
    BENCHMARK_PROTOCOLS       comma-separated subset of "pooled,scene", default both
"""

from __future__ import annotations

from contextlib import redirect_stdout
import dataclasses
from datetime import datetime
import io
import json
import os
import resource
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _liveguard import acquire  # noqa: E402
from sentinel_analysis.sensors.terrain import terrain_predictors  # noqa: E402
from sentinel_analysis import (  # noqa: E402
    AnalysisRequest,
    AnalysisWorkflow,
    ClientConfig,
    blocked_calibration_split,
    blocked_spatiotemporal_split,
    compare_to_reference,
    fit_conformal_downscaler,
    fit_gwr_downscaler,
    fit_linear_downscaler,
    fit_random_forest_downscaler,
    fit_tsharp_downscaler,
    fit_xgboost_downscaler,
    get_city,
    reaggregate_to_target,
    validate_downscaler,
)

CITIES = os.getenv("BENCHMARK_CITIES", "berlin,guadalajara,mexico-city,lagos,nairobi").split(",")
WINDOWS = [tuple(item.split("/")) for item in os.getenv("BENCHMARK_WINDOWS", "2026-08-01/2026-08-14,2026-04-01/2026-04-14").split(",")]
MAX_PRODUCTS = int(os.getenv("BENCHMARK_MAX_PRODUCTS", "24"))
OUTPUT_DIR = PROJECT_ROOT / os.getenv("BENCHMARK_OUTPUT", "output/multicity-benchmark")
RESULTS_FILE = os.getenv("BENCHMARK_RESULTS", "results.json")
PROTOCOLS = set(os.getenv("BENCHMARK_PROTOCOLS", "pooled,scene").split(","))
PREDICTORS = ("NDVI", "NDBI", "EVI", "NDMI")
TERRAIN = ("elevation", "slope", "cos_incidence")
PREDICTOR_SETS = os.getenv("BENCHMARK_PREDICTOR_SETS", "indices,indices+terrain").split(",")
_TERRAIN: xr.Dataset | None = None  # fine static terrain of the run being evaluated
BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
SPLIT = {"validation_fraction": 0.3, "spatial_block_period": 2, "block_size": 5}


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


_T0 = time.time()


def log(message: str) -> None:
    print(f"    [{time.time() - _T0:7.0f}s rss<= {peak_rss_mb():.0f} MB] {message}", flush=True)


def local_solar_hour(timestamp: str, longitude: float) -> float:
    utc = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return (utc.hour + utc.minute / 60 + longitude / 15) % 24


def fetch(city_id: str, start: str, end: str):
    city = get_city(city_id)
    request = AnalysisRequest.for_city(
        city, start, end, sensors=("sentinel3", "sentinel2"), variables=("lst",), resolution_m=100,
        max_products_per_sensor=MAX_PRODUCTS,
    )
    request = dataclasses.replace(request, sentinel2_source="stac_cog", s2_cloud_cover_max=70, terrain_predictors=True)
    workflow = AnalysisWorkflow(request)
    longitude = (city.aoi.west + city.aoi.east) / 2
    discover = workflow.discover

    def daytime_discover():
        products = discover()
        products["sentinel3"] = [
            product for product in products.get("sentinel3", [])
            if product.start_datetime and 8 <= local_solar_hour(product.start_datetime, longitude) <= 16
        ]
        return products

    workflow.discover = daytime_discover  # type: ignore[method-assign]
    run_dir = OUTPUT_DIR / "runs" / f"{city_id}_{start}"
    # max_workers=1: serial downloads/reads, the standing memory-safety rule here.
    result = workflow.execute(
        run_dir, config=ClientConfig.from_env(), max_workers=1,
        progress=lambda sensor, done, total: log(f"acquired {sensor}: {done}/{total}"),
    )
    log("fused cube ready")
    global _TERRAIN
    _TERRAIN = result.terrain
    return result.cube.sortby("time"), result.predictor_cube


def fine_predictors_at(cube: xr.Dataset, predictor_cube: xr.Dataset, index: int) -> xr.Dataset | None:
    matched = cube["sentinel2_matched_time"].values[index] if "sentinel2_matched_time" in cube else np.datetime64("NaT")
    if np.isnat(matched):
        return None
    fine = predictor_cube.sel(time=matched).drop_vars("time", errors="ignore")
    if _TERRAIN is not None:
        # Illumination at the *Sentinel-3* acquisition time being predicted,
        # not at the Sentinel-2 time the optical predictors come from.
        terrain = terrain_predictors(_TERRAIN, [cube.time.values[index]], crs=_TERRAIN.attrs["crs"]).isel(time=0, drop=True)
        fine = fine.assign({name: terrain[name].reindex_like(fine, method="nearest", tolerance=1e-6) for name in TERRAIN})
    return fine


def fine_to_coarse_metrics(name: str, predict_fine, cube: xr.Dataset, predictor_cube: xr.Dataset, validation: xr.Dataset) -> dict:
    """Predict at 100 m for each validation time, reaggregate to 1 km and
    score against the held-out observed LST.

    Also returns ``scene_corrected``: per validation scene, the mean
    observed-minus-predicted offset over that scene's *non-held-out* cells
    is added to the held-out predictions. In real use the whole coarse scene
    is always available (that is what is being downscaled), and no
    regression trained on other days can know today's overall temperature
    level, so this isolates the spatial-pattern skill a downscaler is for.
    The held-out cells themselves are never used for the offset.
    """

    estimates, corrected, observations = [], [], []
    for index in range(validation.sizes["time"]):
        observed = validation["lst"].isel(time=index)
        held_out = np.isfinite(observed.values)
        if not held_out.any():
            continue
        fine = fine_predictors_at(cube, predictor_cube, index)
        if fine is None:
            continue
        prediction = predict_fine(fine, index)
        if prediction is None:
            continue
        coarse = reaggregate_to_target(prediction, observed).values
        scene = cube["lst"].sel(time=validation.time.values[index]).values
        reference_cells = np.isfinite(scene) & ~held_out & np.isfinite(coarse)
        offset = float(np.mean(scene[reference_cells] - coarse[reference_cells])) if reference_cells.sum() >= 10 else np.nan
        estimates.append(coarse.ravel())
        corrected.append((coarse + offset).ravel())
        observations.append(observed.values.ravel())
    if not estimates:
        raise ValueError("no validation time had matching Sentinel-2 predictors")
    reference = xr.DataArray(np.concatenate(observations))
    total = int(np.isfinite(reference.values).sum())
    metrics = compare_to_reference(xr.DataArray(np.concatenate(estimates)), reference, name=name)
    metrics["coverage"] = metrics["samples"] / total if total else float("nan")
    scene_corrected = compare_to_reference(xr.DataArray(np.concatenate(corrected)), reference, name=f"{name}_scene_corrected")
    scene_corrected["coverage"] = scene_corrected["samples"] / total if total else float("nan")
    metrics["scene_corrected"] = scene_corrected
    return metrics


def write_tif(path: Path, data: np.ndarray, grid_like: xr.Dataset) -> None:
    import rasterio
    from rasterio.transform import from_origin

    x, y = grid_like["x"].values, grid_like["y"].values
    dx, dy = abs(x[1] - x[0]), abs(y[1] - y[0])
    transform = from_origin(x.min() - dx / 2, y.max() + dy / 2, dx, dy)
    bands = data if data.ndim == 3 else data[None]
    with rasterio.open(
        path, "w", driver="GTiff", width=bands.shape[2], height=bands.shape[1], count=bands.shape[0],
        dtype="float32", crs=grid_like.attrs["crs"], transform=transform, nodata=np.nan,
    ) as destination:
        destination.write(bands.astype(np.float32))


def train_pydms(pairs, cube, predictor_cube, variables: tuple[str, ...], window: int):
    """Train pyDMS on (coarse LST in degC, fine predictors) pairs; return a
    fine predictor, or None if pyDMS is unavailable or there is nothing to
    train on."""

    try:
        from pyDMS.pyDMS import DecisionTreeSharpener
    except ImportError:
        return None
    workdir = Path(tempfile.mkdtemp(prefix="pydms-", dir=OUTPUT_DIR))
    high_files, low_files = [], []
    for number, (lst, fine) in enumerate(pairs):
        high, low = workdir / f"hr_{number}.tif", workdir / f"lr_{number}.tif"
        write_tif(high, np.stack([fine[name].values for name in variables]), predictor_cube)
        write_tif(low, lst + 273.15, cube)  # pyDMS's temperature mode works in kelvin
        high_files.append(str(high))
        low_files.append(str(low))
    if not high_files:
        return None
    sharpener = DecisionTreeSharpener(
        high_files, low_files, movingWindowSize=window, disaggregatingTemperature=True,
        perLeafLinearRegression=True, linearRegressionExtrapolationRatio=0.25,
        baggingRegressorOpt={"n_estimators": 30, "max_samples": 0.8},
    )
    with redirect_stdout(io.StringIO()):
        sharpener.trainSharpener()

    def predict(fine: xr.Dataset, index: int):
        path = workdir / f"apply_{index}.tif"
        write_tif(path, np.stack([fine[name].values for name in variables]), predictor_cube)
        with redirect_stdout(io.StringIO()):
            output = sharpener.applySharpener(str(path))  # no lowResFilename: no validation-target leakage
        values = output.GetRasterBand(1).ReadAsArray().astype(np.float32) - 273.15
        return xr.DataArray(values, dims=("y", "x"), coords={"y": predictor_cube.y, "x": predictor_cube.x}, attrs={"crs": predictor_cube.attrs["crs"]})

    return predict


def pydms_predictor(cube, predictor_cube, train, variables: tuple[str, ...], window: int):
    """Train pyDMS on all training cells/times (the pooled protocol)."""

    pairs = []
    for index in range(train.sizes["time"]):
        lst = train["lst"].isel(time=index).values
        fine = fine_predictors_at(cube, predictor_cube, index)
        if np.isfinite(lst).sum() >= 20 and fine is not None:
            pairs.append((lst, fine))
    return train_pydms(pairs, cube, predictor_cube, variables, window)


def scene_protocol(cube: xr.Dataset, predictor_cube: xr.Dataset, validation: xr.Dataset, predictors: list[str]) -> dict:
    """Per-scene protocol - pyDMS's intended use, applied to every model.

    For each validation scene, each model is trained only on that scene's
    *non-held-out* coarse cells (with the coarse predictors of that scene),
    predicts the scene at 100 m, and is scored on the held-out cells after
    reaggregation to 1 km. The held-out cells are never used for training.
    """

    from sentinel_analysis import fit_random_forest_downscaler

    names = ("no_skill", "ols", "random_forest", "pydms_global", "pydms_local")
    estimates: dict[str, list] = {name: [] for name in names}
    observations: list = []
    for index in range(validation.sizes["time"]):
        held = validation["lst"].isel(time=index)
        held_mask = np.isfinite(held.values)
        fine = fine_predictors_at(cube, predictor_cube, index)
        if not held_mask.any() or fine is None:
            continue
        scene = cube.isel(time=[index])
        train_lst = scene["lst"].where(~xr.DataArray(held_mask, dims=("y", "x")))
        if int(np.isfinite(train_lst.values).sum()) < 30:
            continue
        training = scene.assign(lst=train_lst)
        predictions: dict[str, xr.DataArray | None] = {}
        mean_level = float(np.nanmean(train_lst.values))
        predictions["no_skill"] = xr.full_like(fine[predictors[0]], mean_level, dtype=float).assign_attrs(crs=predictor_cube.attrs["crs"])
        for name, fit in (
            ("ols", lambda data=training: fit_linear_downscaler(data, predictors=predictors, min_samples=30)),
            ("random_forest", lambda data=training: fit_random_forest_downscaler(data, predictors=predictors, n_estimators=100, min_samples=30)),
        ):
            try:
                predictions[name] = fit().predict(fine)["lst_downscaled"]
            except Exception:  # noqa: BLE001 - too few complete cells in this scene
                predictions[name] = None
        for name, window in (("pydms_global", 0), ("pydms_local", 15)):
            try:
                predict = train_pydms([(train_lst.isel(time=0).values, fine)], cube, predictor_cube, tuple(predictors), window)
                predictions[name] = predict(fine, index) if predict else None
            except Exception:  # noqa: BLE001
                predictions[name] = None
        observations.append(held.values.ravel())
        for name in names:
            prediction = predictions.get(name)
            if prediction is None:
                estimates[name].append(np.full(held.size, np.nan))
            else:
                estimates[name].append(reaggregate_to_target(prediction, held).values.ravel())
    if not observations:
        return {"error": "no validation scene had matching predictors and enough training cells"}
    reference = xr.DataArray(np.concatenate(observations))
    total = int(np.isfinite(reference.values).sum())
    result = {}
    for name in names:
        try:
            metrics = compare_to_reference(xr.DataArray(np.concatenate(estimates[name])), reference, name=name)
            metrics["coverage"] = metrics["samples"] / total if total else float("nan")
            result[name] = metrics
        except ValueError as exc:
            result[name] = {"error": str(exc)}
    return result


def benchmark_one(city_id: str, start: str, end: str) -> dict:
    started = time.time()
    cube, predictor_cube = fetch(city_id, start, end)
    record: dict = {"city": city_id, "start": start, "end": end, "s3_times": int(cube.sizes["time"])}
    if predictor_cube is None:
        record["error"] = "no Sentinel-2 predictors"
        return record
    indices = [name for name in PREDICTORS if name in cube and np.isfinite(cube[name].values).any()]
    train, validation = blocked_spatiotemporal_split(cube, **SPLIT)
    calibration = blocked_calibration_split(cube, **SPLIT)
    record["validation_observations"] = int(np.isfinite(validation["lst"].values).sum())
    record["training_observations"] = int(np.isfinite(train["lst"].values).sum())
    if record["validation_observations"] < 30 or record["training_observations"] < 100:
        record["error"] = "too few clear daytime observations for a holdout"
        return record
    record["sets"] = {}
    for set_name in PREDICTOR_SETS:
        predictors = list(indices) + ([name for name in TERRAIN if name in cube] if "terrain" in set_name else [])
        log(f"predictor set {set_name}: {predictors}")
        record["sets"][set_name] = evaluate(cube, predictor_cube, train, validation, calibration, predictors, record["validation_observations"])
    record["seconds"] = round(time.time() - started, 1)
    record["peak_rss_mb"] = round(peak_rss_mb(), 1)
    return record


def evaluate(cube, predictor_cube, train, validation, calibration, predictors: list[str], validation_observations: int) -> dict:
    """Every protocol and model for one predictor set."""

    record: dict = {"predictors": predictors}
    if "scene" in PROTOCOLS:
        log("per-scene protocol")
        record["scene"] = scene_protocol(cube, predictor_cube, validation, predictors)
    if "pooled" not in PROTOCOLS:
        return record

    spacing = float(abs(cube.x.values[1] - cube.x.values[0]))
    fitters = {
        "ols": lambda: fit_linear_downscaler(train, predictors=predictors, min_samples=50),
        "tsharp_ndvi": lambda: fit_tsharp_downscaler(train, predictor="NDVI", min_samples=50),
        "random_forest": lambda: fit_random_forest_downscaler(train, predictors=predictors, n_estimators=100, min_samples=50),
        "xgboost": lambda: fit_xgboost_downscaler(train, predictors=predictors, min_samples=50),
        "gwr_15x": lambda: fit_gwr_downscaler(train, predictors=predictors, bandwidth=15 * spacing, min_samples=50, max_local_samples=150),
    }
    models: dict = {}
    coarse, fine, intervals = {}, {}, {}
    for name, fit in fitters.items():
        try:
            log(f"fit {name}")
            models[name] = fit()
            log(f"coarse validation {name}")
            coarse[name] = validate_downscaler(models[name], validation)
            coarse[name]["coverage"] = coarse[name]["samples"] / validation_observations
        except Exception as exc:  # noqa: BLE001 - one failing model must not abort the benchmark
            coarse[name] = {"error": str(exc)}
    # No-skill reference: a spatially constant field. After the scene
    # correction it predicts every held-out cell as the scene's mean, so any
    # model that does not beat it there has learned no spatial pattern.
    try:
        fine["no_skill"] = fine_to_coarse_metrics(
            "no_skill", lambda f, _i: xr.zeros_like(f[predictors[0]], dtype=float).assign_attrs(crs=predictor_cube.attrs["crs"]),
            cube, predictor_cube, validation,
        )
    except Exception as exc:  # noqa: BLE001
        fine["no_skill"] = {"error": str(exc)}
    for name, model in models.items():
        if name.startswith("gwr"):
            continue
        try:
            log(f"fine-to-coarse {name}")
            fine[name] = fine_to_coarse_metrics(name, lambda f, _i, m=model: m.predict(f)["lst_downscaled"], cube, predictor_cube, validation)
        except Exception as exc:  # noqa: BLE001
            fine[name] = {"error": str(exc)}
    for name in ("random_forest", "ols"):
        if name in models:
            try:
                log(f"conformal {name}")
                calibrated = fit_conformal_downscaler(models[name], calibration, alpha=0.1)
                metrics = validate_downscaler(calibrated, validation)
                intervals[name] = {key: metrics[key] for key in ("interval_coverage", "interval_mean_width")}
                intervals[name]["method"] = "normalized" if calibrated.normalized else "constant"
            except Exception as exc:  # noqa: BLE001
                intervals[name] = {"error": str(exc)}
    for label, variables, window in (("pydms_global", tuple(predictors), 0), ("pydms_local", tuple(predictors), 15), ("pydms_global_bands", BANDS + tuple(name for name in predictors if name in TERRAIN), 0)):
        try:
            # Terrain variables are added per acquisition by fine_predictors_at,
            # so only the optical ones must already be in the predictor cube.
            missing = set(variables).difference(predictor_cube.data_vars).difference(TERRAIN)
            if missing:
                fine[label] = {"error": f"predictors not in the predictor cube: {sorted(missing)}"}
                continue
            log(f"train {label}")
            predict = pydms_predictor(cube, predictor_cube, train, variables, window)
            if predict is None:
                fine[label] = {"error": "pyDMS unavailable or no training scenes"}
                continue
            fine[label] = fine_to_coarse_metrics(label, predict, cube, predictor_cube, validation)
        except Exception as exc:  # noqa: BLE001
            fine[label] = {"error": f"{type(exc).__name__}: {exc}"}
    record.update({"coarse": coarse, "fine_to_coarse": fine, "intervals": intervals})
    return record




def main() -> int:
    acquire("benchmark_multicity.py")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / RESULTS_FILE
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    done = {(item["city"], item["start"]) for item in results if "error" not in item or item["error"] != "exception"}
    for city_id in CITIES:
        for start, end in WINDOWS:
            if (city_id, start) in done:
                print(f"skip {city_id} {start}: already in results.json")
                continue
            print(f"=== {city_id} {start}..{end}", flush=True)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    record = benchmark_one(city_id, start, end)
            except Exception as exc:  # noqa: BLE001
                record = {"city": city_id, "start": start, "end": end, "error": "exception", "detail": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:]}
            results.append(record)
            results_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
            summary = {}
            for set_name, result in record.get("sets", {}).items():
                summary.update({f"{set_name}:{name}": round(m["scene_corrected"]["rmse"], 2) for name, m in result.get("fine_to_coarse", {}).items() if "scene_corrected" in m})
                summary.update({f"{set_name}:scene:{name}": round(m["rmse"], 2) for name, m in result.get("scene", {}).items() if isinstance(m, dict) and "rmse" in m})
            print(f"    {record.get('error') or summary}  peak RSS {peak_rss_mb():.0f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
