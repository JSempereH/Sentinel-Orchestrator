"""A real (partial) independent-reference validation of `fuse_starfm`.

`scripts/benchmark_downscalers.py`'s STARFM check is a sanity check, not a
real validation: it feeds STARFM another model's own *modelled* map as
`fine_t0`, since no independent fine-resolution reference happened to exist
in that run's data window. This script closes that gap partway: it finds a
real Landsat 8/9 scene over Berlin (`LandsatCatalog`/`read_landsat_lst`,
already verified against real Planetary Computer data) and uses its actual
observed LST as `fine_t0` - a real fine-resolution observation, not a
model's prediction - paired with two real Sentinel-3 coarse LST dates.

This is still only a *partial* independent-reference validation:
- The coarse-scale check (`validate_reaggregation` against the real
  observed Sentinel-3 LST at t1) is a real, quantifiable metric either way.
- A *full* independent-reference validation additionally needs a second
  real fine-resolution scene near t1 to compare STARFM's full-resolution
  output against directly (not just its coarse aggregate). Landsat's ~8-day
  combined revisit (8/9 together) may or may not produce one within this
  script's search window - if it does, this script uses it
  (`validate_independent_reference`); if not, it says so explicitly rather
  than silently reporting only the coarse half as if it were complete.

Usage:
    uv run --extra cdse --extra optical --extra landsat python scripts/validate_starfm_real_reference.py

Environment variables:
    STARFM_VALIDATION_START, STARFM_VALIDATION_END   default 2026-08-01 / 2026-09-15
    STARFM_VALIDATION_RESOLUTION_M                    default 100
    STARFM_VALIDATION_OUTPUT                          default output/starfm-real-reference
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
    compare_to_reference,
    fuse_starfm,
    get_city,
    read_landsat_lst,
    validate_reaggregation,
)

START = os.getenv("STARFM_VALIDATION_START", "2026-08-01")
END = os.getenv("STARFM_VALIDATION_END", "2026-09-15")
RESOLUTION_M = int(os.getenv("STARFM_VALIDATION_RESOLUTION_M", "100"))
OUTPUT_DIR = PROJECT_ROOT / os.getenv("STARFM_VALIDATION_OUTPUT", "output/starfm-real-reference")
LANDSAT_REVISIT_TOLERANCE = np.timedelta64(4, "D")


def find_landsat_scenes(city):
    catalog = LandsatCatalog()
    refs = catalog.search(city.aoi, START, END, cloud_cover_max=30, limit=50)
    refs = sorted(refs, key=lambda r: r.start_datetime or "")
    print(f"Found {len(refs)} Landsat scenes (cloud_cover<=30) between {START} and {END}:")
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
    # a safety margin, and never run this alongside another heavy script.
    result = AnalysisWorkflow(request).execute(OUTPUT_DIR, config=ClientConfig.from_env(), max_workers=1)
    cube = result.cube.sortby("time")
    has_data = [t for t in range(cube.sizes["time"]) if np.isfinite(cube["lst"].isel(time=t).values).any()]
    return cube.isel(time=has_data)


def nearest_coarse_time(cube, target_time):
    deltas = np.abs(cube.time.values - np.datetime64(target_time))
    return int(np.argmin(deltas)), cube.time.values[int(np.argmin(deltas))]


def main() -> int:
    acquire("validate_starfm_real_reference.py")
    city = get_city("Berlin")
    grid = AnalysisGrid.for_city(city, resolution_m=RESOLUTION_M)

    print("Searching for real Landsat scenes...")
    landsat_refs = find_landsat_scenes(city)
    if not landsat_refs:
        print("No usable Landsat scenes found in this window - widen STARFM_VALIDATION_START/END and retry.")
        return 1
    reference_ref = landsat_refs[0]
    print(f"\nUsing Landsat scene as fine_t0: {reference_ref.product_id} ({reference_ref.start_datetime})")

    print("\nFetching real Sentinel-3 coarse LST over the same window...")
    cube = fetch_sentinel3_cube(city)
    print(f"Sentinel-3 cube: {dict(cube.sizes)} real (non-empty) observations")
    if cube.sizes["time"] < 2:
        print("Fewer than two real Sentinel-3 observations - cannot pick a t0/t1 pair.")
        return 1

    landsat_dataset = read_landsat_lst(reference_ref, grid)
    fine_t0 = landsat_dataset["landsat_lst"].isel(time=0)

    t0_index, t0_time = nearest_coarse_time(cube, reference_ref.start_datetime)
    later_indices = [t for t in range(cube.sizes["time"]) if cube.time.values[t] > t0_time]
    if not later_indices:
        print("No real Sentinel-3 observation exists after the Landsat date in this window.")
        return 1
    t1_index = later_indices[-1]
    t1_time = cube.time.values[t1_index]
    print(f"\nt0 (Sentinel-3, nearest to Landsat): {t0_time}")
    print(f"t1 (Sentinel-3, latest available):    {t1_time}")

    coarse_t0 = cube["lst"].isel(time=t0_index)
    coarse_t1 = cube["lst"].isel(time=t1_index)

    fused = fuse_starfm(fine_t0, coarse_t0, coarse_t1, window_radius=5)

    print("\n=== Coarse-scale conservation (real observed t1, quantifiable either way) ===")
    coarse_metrics = validate_reaggregation(coarse_t1, fused["lst_downscaled"], tolerance=1.0)
    print(f"  rmse={coarse_metrics['rmse']:.3f} mae={coarse_metrics['mae']:.3f} "
          f"within_tolerance={coarse_metrics['within_tolerance']}")

    print("\n=== Full-resolution independent-reference check (only possible if a second")
    print("=== real Landsat scene falls near t1) ===")
    reference_date = np.datetime64(reference_ref.start_datetime, "D")
    later_landsat = [
        ref for ref in landsat_refs
        if ref.product_id != reference_ref.product_id
        # Same calendar day as fine_t0's own scene almost certainly means an
        # *adjacent WRS-2 tile from the same overpass* (confirmed live: two
        # scenes 24s apart, consecutive path/row, covering a different
        # footprint), not a later observation of this AOI - comparing
        # against one gave a nonsense ~34 K rmse the first time this ran.
        # Require a genuinely different day, not just a different product_id.
        and np.datetime64(ref.start_datetime, "D") != reference_date
        and abs(np.datetime64(ref.start_datetime) - t1_time) <= LANDSAT_REVISIT_TOLERANCE
    ]
    if not later_landsat:
        print("  No second Landsat scene (on a genuinely different day than fine_t0's own)")
        print(f"  within {LANDSAT_REVISIT_TOLERANCE} of t1 ({t1_time}) in this window -")
        print("  only the coarse-scale check above is a real validation this run; widen the date")
        print("  range or rerun later for a chance at a genuine full-resolution comparison.")
        return 0

    reference_t1 = later_landsat[0]
    print(f"  Found a real second Landsat scene near t1: {reference_t1.product_id} ({reference_t1.start_datetime})")
    reference_dataset = read_landsat_lst(reference_t1, grid)
    reference_fine_t1 = reference_dataset["landsat_lst"].isel(time=0)
    independent_metrics = compare_to_reference(fused["lst_downscaled"], reference_fine_t1, name="landsat_lst")
    print(f"  Full-resolution rmse={independent_metrics['rmse']:.3f} mae={independent_metrics['mae']:.3f} "
          f"n={independent_metrics['samples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
