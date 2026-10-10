"""Benchmark every downscaling baseline in `citycube.downscale` on
one real, live-fetched Sentinel-3 LST + Sentinel-2 predictor cube over
Berlin, per docs/downscaling.md's roadmap item 2 ("Benchmark OLS,
TsHARP/DisTrad, Random Forest and optional XGBoost on blocked
spatial-temporal holdouts") extended to also cover GWR and STARFM.

Regression models (OLS, TsHARP, Random Forest, XGBoost, GWR) are compared
the same way: fit on a blocked_spatiotemporal_split() training block,
evaluate with validate_downscaler() on the held-out block - a genuine
spatial+temporal holdout, not a random pixel split (docs/downscaling.md's
"Validation Protocol" is explicit that a random split overstates skill).

STARFM cannot be benchmarked the same way - it has no "fit on a training
cube" step, and evaluating it properly needs an independent fine-resolution
LST reference at two dates close enough together for one real coarse
change to have occurred between them, which did not happen to exist in
this run's real data. Instead it is sanity-checked by coarse-scale
conservation: feed it the best regression model's own downscaled map at
t0 as its "fine_t0" input, predict t1, reaggregate to the coarse grid, and
compare against the *real* observed coarse LST at t1 - the same
conservation check `validate_reaggregation` already applies to every other
model here, just applied to STARFM's output instead of a regression
model's.

A first real run (one week, 12 products/sensor) found GWR's rmse was not
comparable to the other models': its validation `n` was a tenth of theirs,
since a too-small bandwidth under this *spatial* holdout left most held-out
cells without a nearby training neighbor - `GWRDownscaler` correctly
reports "no local fit" there instead of extrapolating, but that made its
rmse computed on an easier subset. This version reports `coverage`
(fraction of the validation set's real observations any given model
actually produced a prediction for) alongside every model's rmse, and
benchmarks two GWR bandwidths side by side so that tradeoff is visible
instead of implicit. It also adds a couple of named, deliberately
different Random Forest/XGBoost configurations (not a full hyperparameter
search - just enough to tell whether one week's untuned-defaults result was
a tuning artifact or a real finding).

Usage:
    uv run --extra cdse --extra optical --extra ml python scripts/benchmark_downscalers.py

Environment variables:
    BENCHMARK_START, BENCHMARK_END         default 2026-08-01 / 2026-08-27
    BENCHMARK_MAX_PRODUCTS                 default 24
    BENCHMARK_RESOLUTION_M                 default 100
    BENCHMARK_OUTPUT                       default output/downscaler-benchmark
"""

from __future__ import annotations

import dataclasses
import os
import sys
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _liveguard import acquire  # noqa: E402
from citycube import (  # noqa: E402
    AnalysisRequest,
    AnalysisWorkflow,
    ClientConfig,
    blocked_spatiotemporal_split,
    fit_gwr_downscaler,
    fit_linear_downscaler,
    fit_random_forest_downscaler,
    fit_tsharp_downscaler,
    fit_xgboost_downscaler,
    fuse_starfm,
    get_city,
    validate_downscaler,
    validate_reaggregation,
)

START = os.getenv("BENCHMARK_START", "2026-08-01")
END = os.getenv("BENCHMARK_END", "2026-08-27")
MAX_PRODUCTS = int(os.getenv("BENCHMARK_MAX_PRODUCTS", "24"))
RESOLUTION_M = int(os.getenv("BENCHMARK_RESOLUTION_M", "100"))
OUTPUT_DIR = PROJECT_ROOT / os.getenv("BENCHMARK_OUTPUT", "output/downscaler-benchmark")
PREFERRED_PREDICTORS = ("NDVI", "NDBI", "EVI", "NDMI")


def fetch_training_cube():
    request = AnalysisRequest.for_city(
        get_city("Berlin"),
        START,
        END,
        sensors=("sentinel3", "sentinel2"),
        variables=("lst",),
        resolution_m=RESOLUTION_M,
        max_products_per_sensor=MAX_PRODUCTS,
    )
    # Daytime passes only (the relation these models learn is a daytime
    # one), and keep the raw archives already cached under OUTPUT_DIR: the
    # default "aoi_subset" retention would delete them after reading.
    request = dataclasses.replace(request, thermal_overpass="day", raw_retention="keep")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # max_workers=1: this machine froze twice when two live-fetch scripts
    # each ran with max_workers=2 concurrently - keep processing serial as
    # a safety margin, and never run this alongside another heavy script.
    result = AnalysisWorkflow(request).execute(OUTPUT_DIR, config=ClientConfig.from_env(), max_workers=1)
    return result.cube, result.predictors.get("sentinel2")


def select_predictors(cube, predictor_cube) -> list[str]:
    return [
        name
        for name in PREFERRED_PREDICTORS
        if name in cube
        and name in predictor_cube
        and bool(np.isfinite(cube[name].values).any())
        and bool(np.isfinite(predictor_cube[name].values).any())
    ]


def grid_spacing_m(cube) -> float:
    x = cube["x"].values
    return float(abs(x[1] - x[0])) if len(x) > 1 else float(RESOLUTION_M)


def run_regression_benchmark(cube, predictors: list[str]):
    train, validation = blocked_spatiotemporal_split(cube, validation_fraction=0.3, spatial_block_period=3)
    total_valid = int(np.isfinite(validation["lst"].values).sum())
    results: dict[str, dict] = {}
    models: dict[str, object] = {}

    def evaluate(name: str, fit_callable) -> None:
        try:
            model = fit_callable()
        except Exception as exc:  # noqa: BLE001 - report and continue, don't abort the whole benchmark
            print(f"  [{name}] unavailable/failed: {exc}")
            return
        metrics = validate_downscaler(model, validation)
        coverage = metrics["samples"] / total_valid if total_valid else float("nan")
        metrics["coverage"] = coverage
        results[name] = metrics
        models[name] = model
        print(f"  [{name}] rmse={metrics['rmse']:.3f} mae={metrics['mae']:.3f} bias={metrics['bias']:.3f} "
              f"corr={metrics['correlation']:.3f} n={metrics['samples']} coverage={coverage:.1%}")

    print(f"Training block: {train.sizes['time']} times; validation block: {validation.sizes['time']} times")
    print(f"Predictors: {predictors}")
    print(f"Total real observations in the validation block: {total_valid}")
    print("Fitting and evaluating (blocked spatial+temporal holdout):")

    evaluate("ols", lambda: fit_linear_downscaler(train, predictors=predictors, min_samples=50))
    if "NDVI" in predictors:
        evaluate("tsharp_ndvi", lambda: fit_tsharp_downscaler(train, predictor="NDVI", min_samples=50))

    # Two named configurations each, not a full search - enough to tell
    # whether the untuned defaults' result is a tuning artifact or real.
    evaluate(
        "random_forest_default",
        lambda: fit_random_forest_downscaler(train, predictors=predictors, min_samples=50),
    )
    evaluate(
        "random_forest_shallow",
        lambda: fit_random_forest_downscaler(train, predictors=predictors, n_estimators=100, min_samples=50),
    )
    evaluate(
        "xgboost_default",
        lambda: fit_xgboost_downscaler(train, predictors=predictors, min_samples=50),
    )
    evaluate(
        "xgboost_regularized",
        lambda: fit_xgboost_downscaler(
            train, predictors=predictors, n_estimators=100, max_depth=3, learning_rate=0.1, min_samples=50,
        ),
    )

    # Two bandwidths side by side: the original 5x grid spacing (from the
    # first real run, where it left 90% of held-out cells unsupported
    # under this spatial holdout) and a wider 15x, so the coverage tradeoff
    # is visible instead of a single, potentially-misleading number.
    spacing = grid_spacing_m(cube)
    for multiplier in (5, 15):
        evaluate(
            f"gwr_bandwidth_{multiplier}x",
            lambda multiplier=multiplier: fit_gwr_downscaler(
                train, predictors=predictors, bandwidth=multiplier * spacing,
                min_samples=50, min_local_samples=10, max_local_samples=150,
            ),
        )

    best_name = min(results, key=lambda name: results[name]["rmse"]) if results else None
    best_model = models.get(best_name) if best_name else None
    return results, best_name, best_model


def run_starfm_sanity_check(cube, predictor_cube, fallback_model) -> dict | None:
    """STARFM needs a fine_t0 map - since no independent fine LST reference
    happened to be available in this real run, use the best-performing
    regression model's own downscaled prediction at t0 as fine_t0 (an
    honest substitute, not a real independent reference - see module
    docstring), then check whether STARFM's t1 output reaggregates back to
    the *real* observed coarse LST at t1.

    `cube.time` is not chronologically sorted (an artifact of how per-orbit
    Sentinel-3 products get concatenated) and at least one real acquisition
    in a typical week has zero finite pixels (fully cloud-obscured) - a
    naive first/last-index pick can silently land on an empty observation.
    Sort first, then pick the first and last acquisitions that actually
    have any finite pixel at all.
    """

    if fallback_model is None:
        print("  No working regression model - STARFM sanity check skipped.")
        return None

    cube_sorted = cube.sortby("time")
    has_data = [t for t in range(cube_sorted.sizes["time"]) if np.isfinite(cube_sorted["lst"].isel(time=t).values).any()]
    if len(has_data) < 2:
        print("  Fewer than two real observations have any finite pixel - STARFM sanity check skipped.")
        return None
    t0_index, t1_index = has_data[0], has_data[-1]
    t0_time, t1_time = cube_sorted.time.values[t0_index], cube_sorted.time.values[t1_index]
    print(f"  Using real coarse observations t0={t0_time} -> t1={t1_time}")

    predictor_at_t0 = predictor_cube.reindex(time=[t0_time], method="nearest")
    fine_t0 = fallback_model.predict(predictor_at_t0)["lst_downscaled"].isel(time=0)
    coarse_t0 = cube_sorted["lst"].isel(time=t0_index)
    coarse_t1 = cube_sorted["lst"].isel(time=t1_index)

    fused = fuse_starfm(fine_t0, coarse_t0, coarse_t1, window_radius=5)
    metrics = validate_reaggregation(coarse_t1, fused["lst_downscaled"], tolerance=1.0)
    print(f"  [starfm] reaggregated-to-coarse vs real observed t1: rmse={metrics['rmse']:.3f} "
          f"mae={metrics['mae']:.3f} within_tolerance={metrics['within_tolerance']}")
    return metrics


def main() -> int:
    acquire("benchmark_downscalers.py")
    print(f"Fetching real Sentinel-3 + Sentinel-2 data over Berlin, {START} to {END}...")
    cube, predictor_cube = fetch_training_cube()
    if predictor_cube is None:
        print("No predictor cube returned - nothing to benchmark.")
        return 1
    print(f"Fused cube: {dict(cube.sizes)}; predictor cube: {dict(predictor_cube.sizes)}")

    predictors = select_predictors(cube, predictor_cube)
    if len(predictors) < 1:
        print(f"Not enough usable predictors: {predictors}")
        return 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results, best_name, best_model = run_regression_benchmark(cube, predictors)

    print(f"\nSTARFM sanity check (fine_t0 sourced from '{best_name}', the best regression model by rmse):")
    starfm_metrics = run_starfm_sanity_check(cube, predictor_cube, best_model)
    if starfm_metrics:
        results["starfm_sanity_check"] = starfm_metrics

    print("\n=== Summary (rmse, lower is better - coverage shows what fraction of the real")
    print("=== validation observations each model actually produced a prediction for) ===")
    for name, metrics in sorted(results.items(), key=lambda item: item[1].get("rmse", float("inf"))):
        coverage = metrics.get("coverage")
        coverage_str = f"{coverage:.1%}" if coverage is not None else "n/a"
        print(f"  {name:24s} rmse={metrics['rmse']:.3f}  coverage={coverage_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
