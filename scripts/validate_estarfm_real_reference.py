"""A real (partial) independent-reference validation of `fuse_estarfm`.

Mirrors `scripts/validate_starfm_real_reference.py`, but ESTARFM needs two
bracketing fine/coarse pairs (t0, t2), not STARFM's single pair - so this
finds two real Landsat 8/9 scenes over Berlin (`LandsatCatalog`/
`read_landsat_lst`) to use as `fine_t0`/`fine_t2` (real observations, not a
model's prediction), and a real Sentinel-3 coarse observation strictly
between them in time as `coarse_t1` - the value ESTARFM is asked to predict
a fine map for.

Same two-tier honesty as the STARFM script:
- The coarse-scale check (`validate_reaggregation` against the real
  observed Sentinel-3 LST at t1) is a real, quantifiable metric either way.
- A *full* independent-reference validation additionally needs a third real
  Landsat scene near t1 to compare ESTARFM's full-resolution output against
  directly. This script uses one if the search window produced one
  (`compare_to_reference`); if not, it says so explicitly rather than
  silently reporting only the coarse half as if it were complete.

Usage:
    uv run --extra cdse --extra optical --extra landsat python scripts/validate_estarfm_real_reference.py

Environment variables:
    ESTARFM_VALIDATION_START, ESTARFM_VALIDATION_END   default 2026-08-01 / 2026-09-15
    ESTARFM_VALIDATION_RESOLUTION_M                      default 100
    ESTARFM_VALIDATION_OUTPUT                            default output/estarfm-real-reference
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _liveguard import acquire  # noqa: E402
from sentinel_analysis import (  # noqa: E402
    AnalysisGrid,
    AnalysisRequest,
    AnalysisWorkflow,
    ClientConfig,
    LandsatCatalog,
    ProductRef,
    compare_to_reference,
    fuse_estarfm,
    get_city,
    read_landsat_lst,
    validate_reaggregation,
)

START = os.getenv("ESTARFM_VALIDATION_START", "2026-08-01")
END = os.getenv("ESTARFM_VALIDATION_END", "2026-09-15")
RESOLUTION_M = int(os.getenv("ESTARFM_VALIDATION_RESOLUTION_M", "100"))
OUTPUT_DIR = PROJECT_ROOT / os.getenv("ESTARFM_VALIDATION_OUTPUT", "output/estarfm-real-reference")
LANDSAT_REVISIT_TOLERANCE = np.timedelta64(4, "D")


def find_landsat_scenes(city):
    catalog = LandsatCatalog()
    refs = catalog.search(city.aoi, START, END, cloud_cover_max=30, limit=50)
    # Same-day scenes are adjacent WRS-2 tiles from one overpass, not two
    # independent observations - see validate_starfm_real_reference.py's
    # note on this. Keep only the first scene seen per calendar day.
    by_day: dict[np.datetime64, ProductRef] = {}
    for ref in sorted(refs, key=lambda r: r.start_datetime or ""):
        day = np.datetime64(ref.start_datetime, "D")
        by_day.setdefault(day, ref)
    refs = sorted(by_day.values(), key=lambda r: r.start_datetime or "")
    print(f"Found {len(refs)} usable Landsat scene-days (cloud_cover<=30) between {START} and {END}:")
    for ref in refs:
        print(f"  {ref.start_datetime}  {ref.product_id}")
    return refs


def fetch_sentinel3_cube(city):
    request = AnalysisRequest.for_city(
        city, START, END, sensors=("sentinel3",), variables=("lst",),
        resolution_m=1000, max_products_per_sensor=60,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # max_workers=1: this machine froze twice when two live-fetch scripts
    # each ran with max_workers=2 concurrently - keep processing serial as
    # a safety margin, and never run this alongside another heavy script
    # (see _liveguard.acquire() above and docs/history.md).
    result = AnalysisWorkflow(request).execute(OUTPUT_DIR, config=ClientConfig.from_env(), max_workers=1)
    cube = result.cube.sortby("time")
    has_data = [t for t in range(cube.sizes["time"]) if np.isfinite(cube["lst"].isel(time=t).values).any()]
    return cube.isel(time=has_data)


def nearest_coarse_index(cube, target_time):
    deltas = np.abs(cube.time.values - np.datetime64(target_time))
    return int(np.argmin(deltas))


def main() -> int:
    acquire("validate_estarfm_real_reference.py")
    city = get_city("Berlin")
    grid = AnalysisGrid.for_city(city, resolution_m=RESOLUTION_M)

    print("Searching for real Landsat scenes...")
    landsat_refs = find_landsat_scenes(city)
    if len(landsat_refs) < 2:
        print("Fewer than two usable Landsat scene-days found - ESTARFM needs a genuine")
        print("bracketing pair (t0, t2). Widen ESTARFM_VALIDATION_START/END and retry.")
        return 1
    ref_t0, ref_t2 = landsat_refs[0], landsat_refs[1]
    print("\nUsing Landsat scenes as the bracketing pair:")
    print(f"  t0: {ref_t0.product_id} ({ref_t0.start_datetime})")
    print(f"  t2: {ref_t2.product_id} ({ref_t2.start_datetime})")

    print("\nFetching real Sentinel-3 coarse LST over the same window...")
    cube = fetch_sentinel3_cube(city)
    print(f"Sentinel-3 cube: {dict(cube.sizes)} real (non-empty) observations")

    t0_bracket = np.datetime64(ref_t0.start_datetime)
    t2_bracket = np.datetime64(ref_t2.start_datetime)
    between = [
        t for t in range(cube.sizes["time"])
        if t0_bracket < cube.time.values[t] < t2_bracket
    ]
    if not between:
        print(f"No real Sentinel-3 observation falls strictly between t0 ({t0_bracket}) and")
        print(f"t2 ({t2_bracket}) in this window - nothing for ESTARFM to predict against.")
        print("Widen ESTARFM_VALIDATION_START/END, or this Landsat pair is too close together.")
        return 1
    # Prefer the observation closest to the pair's midpoint - the case
    # ESTARFM's local-slope interpolation is actually meant to handle.
    midpoint = t0_bracket + (t2_bracket - t0_bracket) / 2
    t1_index = min(between, key=lambda t: abs(cube.time.values[t] - midpoint))
    t1_time = cube.time.values[t1_index]
    print(f"\nt1 (real Sentinel-3 observation between t0 and t2): {t1_time}")

    coarse_t0 = cube["lst"].isel(time=nearest_coarse_index(cube, ref_t0.start_datetime))
    coarse_t2 = cube["lst"].isel(time=nearest_coarse_index(cube, ref_t2.start_datetime))
    coarse_t1 = cube["lst"].isel(time=t1_index)

    fine_t0 = read_landsat_lst(ref_t0, grid)["landsat_lst"].isel(time=0)
    fine_t2 = read_landsat_lst(ref_t2, grid)["landsat_lst"].isel(time=0)

    fused = fuse_estarfm(fine_t0, coarse_t0, fine_t2, coarse_t2, coarse_t1, window_radius=5)

    print("\n=== Coarse-scale conservation (real observed t1, quantifiable either way) ===")
    coarse_metrics = validate_reaggregation(coarse_t1, fused["lst_downscaled"], tolerance=1.0)
    print(f"  rmse={coarse_metrics['rmse']:.3f} mae={coarse_metrics['mae']:.3f} "
          f"within_tolerance={coarse_metrics['within_tolerance']}")

    print("\n=== Full-resolution independent-reference check (only possible if a third")
    print("=== real Landsat scene falls near t1) ===")
    used_days = {np.datetime64(ref_t0.start_datetime, "D"), np.datetime64(ref_t2.start_datetime, "D")}
    third_scene = next(
        (
            ref for ref in landsat_refs
            if np.datetime64(ref.start_datetime, "D") not in used_days
            and abs(np.datetime64(ref.start_datetime) - t1_time) <= LANDSAT_REVISIT_TOLERANCE
        ),
        None,
    )
    if third_scene is None:
        print(f"  No third Landsat scene-day within {LANDSAT_REVISIT_TOLERANCE} of t1 ({t1_time}) -")
        print("  only the coarse-scale check above is a real validation this run; widen the date")
        print("  range or rerun later for a chance at a genuine full-resolution comparison.")
        return 0

    print(f"  Found a real third Landsat scene near t1: {third_scene.product_id} ({third_scene.start_datetime})")
    reference_fine_t1 = read_landsat_lst(third_scene, grid)["landsat_lst"].isel(time=0)
    independent_metrics = compare_to_reference(fused["lst_downscaled"], reference_fine_t1, name="landsat_lst")
    print(f"  Full-resolution rmse={independent_metrics['rmse']:.3f} mae={independent_metrics['mae']:.3f} "
          f"n={independent_metrics['samples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
