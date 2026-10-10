"""Real-scene validation of the cloud-native readers (`stac_cog`, `pc_rtc`).

`read_s2_l2a_cog` and `read_s1_rtc_cog` were unit-tested only against
synthetic local GeoTIFFs. This script checks them against real data:

1. Sentinel-2 `stac_cog` vs `cdse_safe`: one Berlin L2A scene already cached
   by `scripts/benchmark_downscalers.py` (the full SAFE archive) is read
   through `read_s2_l2a` + `AnalysisWorkflow._to_grid`, and the *same*
   acquisition is read in place from Planetary Computer's STAC COGs through
   `read_s2_l2a_cog`. Both land on the same Berlin grid (20 m, where pixels
   should agree almost exactly, and 100 m) and are compared band by band.
2. Sentinel-1 `pc_rtc`: no SNAP/HyP3 output exists on this machine to compare
   against, so this is a physical-plausibility check of one real Planetary
   Computer RTC scene: units/nodata metadata, linear gamma0 in the expected
   range, VH below VV, and open water (from the Sentinel-2 SCL water class)
   much darker than land.

Signed asset URLs carry SAS tokens and are never printed.

Usage:
    uv run --extra optical --extra landsat python scripts/validate_cloud_native_real_reference.py
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _liveguard import acquire  # noqa: E402
from citycube import (  # noqa: E402
    AnalysisGrid,
    Sentinel1RTCSTACCatalog,
    Sentinel2STACCatalog,
    get_city,
    read_s1_rtc_cog,
    read_s2_l2a,
    read_s2_l2a_cog,
)
from citycube.workflow.adapters import to_grid  # noqa: E402

SAFE = Path(os.getenv(
    "CLOUD_NATIVE_VALIDATION_SAFE",
    "output/downscaler-benchmark/downloads/sentinel2/S2B_MSIL2A_20260810T101019_N0512_R022_T33UUU_20260810T135525.SAFE.zip",
))
OUTPUT = Path(os.getenv("CLOUD_NATIVE_VALIDATION_OUTPUT", "output/cloud-native-validation"))
S1_START = os.getenv("CLOUD_NATIVE_VALIDATION_S1_START", "2026-08-05")
S1_END = os.getenv("CLOUD_NATIVE_VALIDATION_S1_END", "2026-08-15")
BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def compare(safe, cog, valid) -> dict[str, dict[str, float]]:
    stats = {}
    for band in BANDS:
        a = safe[band].isel(time=0).values
        b = cog[band].isel(time=0).values
        mask = valid & np.isfinite(a) & np.isfinite(b)
        diff = b[mask] - a[mask]
        stats[band] = {
            "pixels": int(mask.sum()),
            "safe_mean": float(a[mask].mean()),
            "cog_mean": float(b[mask].mean()),
            "bias": float(diff.mean()),
            "rmse": float(np.sqrt((diff**2).mean())),
            "max_abs": float(np.abs(diff).max()),
            "corr": float(np.corrcoef(a[mask], b[mask])[0, 1]),
        }
    return stats


def find_matching_item(scene_name: str, city_aoi):
    # S2B_MSIL2A_20260810T101019_N0512_R022_T33UVU_... -> date, orbit, tile
    parts = scene_name.split("_")
    sensing, orbit, tile = parts[2], parts[4], parts[5]
    day = f"{sensing[:4]}-{sensing[4:6]}-{sensing[6:8]}"
    candidates = Sentinel2STACCatalog().search(city_aoi, day, day, limit=50)
    matches = [c for c in candidates if sensing in c.product_id and orbit in c.product_id and tile in c.product_id]
    if not matches:
        raise SystemExit(f"No Planetary Computer item matches {sensing}/{orbit}/{tile}; candidates: {[c.product_id for c in candidates]}")
    return matches[0]


def validate_sentinel2(city, report: dict) -> object:
    item = find_matching_item(SAFE.name, city.aoi)
    report["s2_stac_item"] = item.product_id
    report["s2_processing_baseline"] = item.metadata["properties"].get("s2:processing_baseline")
    band_meta = {band: {k: v for k, v in item.metadata["assets"][band].items() if k not in {"href", "alternate"}} for band in ("B04", "SCL")}
    report["s2_asset_metadata"] = json.loads(json.dumps(band_meta, default=str))

    extract_dir = OUTPUT / "safe-extract"
    try:
        safe_native = read_s2_l2a(SAFE, aoi=city.aoi, extract_dir=extract_dir)
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)
    print(f"SAFE read: {dict(safe_native.sizes)}  peak RSS {peak_rss_mb():.0f} MB")

    results = {}
    scl_100m = None
    for resolution in (20, 100):
        grid = AnalysisGrid.for_city(city, resolution_m=resolution)
        safe = to_grid(safe_native, grid)
        cog = read_s2_l2a_cog(item, grid, bands=(*BANDS, "SCL"))
        safe_valid = safe["valid_mask"].isel(time=0).values.astype(bool)
        cog_valid = cog["valid_mask"].isel(time=0).values.astype(bool)
        valid = safe_valid & cog_valid
        safe_scl, cog_scl = safe["SCL"].isel(time=0).values, cog["SCL"].isel(time=0).values
        both_scl = np.isfinite(safe_scl) & np.isfinite(cog_scl)
        results[f"{resolution}m"] = {
            "grid": [grid.height, grid.width],
            "footprint_safe": float(np.isfinite(safe["B04"].isel(time=0).values).mean()),
            "footprint_cog": float(np.isfinite(cog["B04"].isel(time=0).values).mean()),
            "valid_both": int(valid.sum()),
            "valid_safe_only": int((safe_valid & ~cog_valid).sum()),
            "valid_cog_only": int((cog_valid & ~safe_valid).sum()),
            "scl_agreement": float((safe_scl[both_scl] == cog_scl[both_scl]).mean()),
            "bands": compare(safe, cog, valid),
        }
        print(f"S2 {resolution} m compared  peak RSS {peak_rss_mb():.0f} MB")
        if resolution == 100:
            scl_100m = safe["SCL"].isel(time=0).values
    report["s2_comparison"] = results
    return scl_100m


def validate_sentinel1(city, water_scl, report: dict) -> None:
    items = Sentinel1RTCSTACCatalog().search(city.aoi, S1_START, S1_END, limit=20)
    report["s1_items_found"] = len(items)
    if not items:
        report["s1"] = "no sentinel-1-rtc items found"
        return
    item = items[0]
    report["s1_item"] = item.product_id
    report["s1_asset_metadata"] = json.loads(json.dumps(
        {name: {k: v for k, v in asset.items() if k not in {"href", "alternate"}} for name, asset in item.metadata["assets"].items() if name in {"vv", "vh"}},
        default=str,
    ))
    grid = AnalysisGrid.for_city(city, resolution_m=100)
    cube = read_s1_rtc_cog(item, grid)
    vv = cube["gamma0_VV"].isel(time=0).values
    vh = cube["gamma0_VH"].isel(time=0).values
    finite = np.isfinite(vv) & np.isfinite(vh)
    water = finite & (water_scl == 6)
    land = finite & np.isin(water_scl, (4, 5))
    report["s1_stats"] = {
        "valid_fraction": float(finite.mean()),
        "vv_median": float(np.median(vv[finite])),
        "vh_median": float(np.median(vh[finite])),
        "vv_p01_p99": [float(np.percentile(vv[finite], 1)), float(np.percentile(vv[finite], 99))],
        "negative_or_zero_fraction": float((vv[finite] <= 0).mean()),
        "vh_below_vv_fraction": float((vh[finite] < vv[finite]).mean()),
        "water_pixels": int(water.sum()),
        "vv_water_median": float(np.median(vv[water])) if water.any() else None,
        "vv_land_median": float(np.median(vv[land])) if land.any() else None,
    }
    print(f"S1 read  peak RSS {peak_rss_mb():.0f} MB")


def main() -> int:
    acquire("validate_cloud_native_real_reference.py")
    if not SAFE.exists():
        raise SystemExit(f"Cached SAFE not found: {SAFE}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    city = get_city("Berlin")
    report: dict = {"safe": SAFE.name}
    water_scl = validate_sentinel2(city, report)
    validate_sentinel1(city, water_scl, report)
    report["peak_rss_mb"] = peak_rss_mb()
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
