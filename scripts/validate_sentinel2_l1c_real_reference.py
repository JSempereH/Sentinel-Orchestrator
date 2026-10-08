"""Real-product validation of the new Sentinel-2 L1C reader.

`read_s2_l1c`/`Sentinel2L1CReadConfig`/`Sentinel2Catalog(product_type=...)`
were added to support methane point-source detection (Varon et al. 2024,
Nat. Commun. s41467-024-47754-y), which needs top-of-atmosphere reflectance
rather than the L2A surface reflectance `read_s2_l2a` already reads. Their
unit tests only cover the CDSE `$filter` construction against a fake
session - this script finds and downloads one real L1C product through
CDSE and confirms `read_s2_l1c` parses its actual SAFE layout (no SCL, no
R10m/R20m/R60m resolution triplicates - a genuinely different structure
from L2A, not just a flag).

Usage:
    uv run --extra cdse --extra optical python scripts/validate_sentinel2_l1c_real_reference.py

Environment variables:
    SENTINEL2_L1C_VALIDATION_CITY          default Berlin
    SENTINEL2_L1C_VALIDATION_START/END     default 2026-08-01 / 2026-09-01
    SENTINEL2_L1C_VALIDATION_CLOUD_COVER_MAX  default 25
    SENTINEL2_L1C_VALIDATION_OUTPUT        default output/sentinel2-l1c-real-reference
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
    AssetCache,
    CDSEDownloader,
    ClientConfig,
    SENTINEL2_L1C_PRODUCT_TYPE,
    Sentinel2Catalog,
    get_city,
    read_s2_l1c,
    select_product_refs,
)

CITY = os.getenv("SENTINEL2_L1C_VALIDATION_CITY", "Berlin")
START = os.getenv("SENTINEL2_L1C_VALIDATION_START", "2026-08-01")
END = os.getenv("SENTINEL2_L1C_VALIDATION_END", "2026-09-01")
CLOUD_COVER_MAX = float(os.getenv("SENTINEL2_L1C_VALIDATION_CLOUD_COVER_MAX", "25"))
OUTPUT = Path(os.getenv("SENTINEL2_L1C_VALIDATION_OUTPUT", "output/sentinel2-l1c-real-reference"))


def main() -> int:
    acquire("validate_sentinel2_l1c_real_reference.py")

    config = ClientConfig.from_env()
    aoi = get_city(CITY).aoi

    catalog = Sentinel2Catalog(config, product_type=SENTINEL2_L1C_PRODUCT_TYPE)
    products = catalog.search(aoi, START, END, cloud_cover_max=CLOUD_COVER_MAX, limit=20)
    selected = select_product_refs(products, limit=1, max_cloud_cover=CLOUD_COVER_MAX)
    if not selected:
        print(f"No L1C product found for {CITY} between {START} and {END} with cloud_cover_max={CLOUD_COVER_MAX}", file=sys.stderr)
        return 1
    product = selected[0]
    print(f"Selected: {product.name} (cloud_cover={product.cloud_cover}, online={product.online})")

    downloader = CDSEDownloader(config, cache=AssetCache(OUTPUT / "cache"))
    archive = downloader.download(product, OUTPUT / "downloads")
    print(f"Downloaded: {archive} ({archive.stat().st_size / 1e6:.1f} MB)")

    dataset = read_s2_l1c(archive, aoi=aoi)
    print(f"product_type: {dataset.attrs['product_type']}")
    print(f"reflectance_scale: {dataset.attrs['reflectance_scale']}")
    print(f"shape: y={dataset.sizes['y']} x={dataset.sizes['x']}")
    print(f"bands: {sorted(dataset.data_vars)}")
    assert "SCL" not in dataset.data_vars, "L1C must not carry an L2A-only SCL band"

    out_of_range = []
    for band in sorted(dataset.data_vars):
        values = dataset[band].values
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            print(f"  {band}: no finite pixels in this AOI")
            continue
        low, high, mean = float(finite.min()), float(finite.max()), float(finite.mean())
        print(f"  {band}: min={low:.4f} max={high:.4f} mean={mean:.4f} finite_px={finite.size}")
        if low < -0.05 or high > 1.5:
            out_of_range.append(band)

    if out_of_range:
        print(f"WARNING: reflectance out of the physically plausible [0, ~1.2] range for: {out_of_range}", file=sys.stderr)
    print("Real Sentinel-2 L1C read: OK" if not out_of_range else "Real Sentinel-2 L1C read: completed with warnings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
