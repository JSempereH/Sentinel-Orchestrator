"""Sentinel-2 L2A acquisition and SAFE reading utilities.

The optical adapter is optional because JPEG2000 support adds a heavier
dependency. Install it with ``uv sync --extra optical``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import re
import tempfile
from typing import Any
import zipfile

import numpy as np
import requests
import xarray as xr

from ..catalog import ProductRef, _product_ref
from ..config import AOI, ClientConfig
from ..http import http_session
from ..metadata import apply_variable_contract


SENTINEL2_COLLECTION = "SENTINEL-2"
SENTINEL2_PRODUCT_TYPE = "S2MSI2A"
S2_BANDS_10M = ("B02", "B03", "B04", "B08")
S2_BANDS_20M = ("B05", "B06", "B07", "B8A", "B11", "B12")
S2_SCL_CLASSES = {
    0: "no_data",
    1: "saturated_or_defective",
    2: "dark_features_or_shadows",
    3: "cloud_shadow",
    4: "vegetation",
    5: "bare_soils",
    6: "water",
    7: "unclassified",
    8: "cloud_medium_probability",
    9: "cloud_high_probability",
    10: "thin_cirrus",
    11: "snow_or_ice",
}


@dataclass(frozen=True)
class Sentinel2ReadConfig:
    """Options for reading a Sentinel-2 L2A granule."""

    bands: tuple[str, ...] = S2_BANDS_10M + ("B11", "B12", "SCL")
    target_resolution_m: int = 20
    scale_reflectance: bool = True
    observation_time: str | np.datetime64 | None = None


def _require_rasterio():
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.transform import from_origin
        from rasterio.warp import reproject
    except ImportError as exc:
        raise RuntimeError("Install sentinel-analysis[optical] to read Sentinel-2 JPEG2000 data") from exc
    return rasterio, Resampling, from_origin, reproject


def _safe_root(path: str | Path, extract_dir: str | Path | None = None) -> Path:
    path = Path(path)
    if path.is_dir():
        return path
    if path.suffix.lower() not in {".zip", ".safe"}:
        raise FileNotFoundError(f"Expected a Sentinel-2 SAFE directory or ZIP archive: {path}")
    destination = Path(extract_dir) if extract_dir else Path(tempfile.mkdtemp(prefix="sentinel2-l2a-"))
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        archive.extractall(destination)
    safe_dirs = list(destination.glob("*.SAFE"))
    if len(safe_dirs) == 1:
        return safe_dirs[0]
    manifests = list(destination.rglob("MTD_MSIL2A.xml"))
    if len(manifests) == 1:
        return manifests[0].parent
    raise ValueError(f"Could not identify one Sentinel-2 SAFE product under {destination}")


def _band_file(root: Path, band: str, resolution: int) -> Path:
    if band == "SCL":
        candidates = sorted(root.rglob("*_SCL_20m.jp2"))
    else:
        candidates = sorted(root.rglob(f"*_{band}_{resolution}m.jp2"))
        if not candidates:
            candidates = sorted(root.rglob(f"*_{band}_*m.jp2"))
    if not candidates:
        raise FileNotFoundError(f"Could not find Sentinel-2 band {band} under {root}")
    return candidates[0]


def _target_grid(source, resolution: int, from_origin, bounds=None):
    bounds = bounds or source.bounds
    width = max(1, int(round((bounds.right - bounds.left) / resolution)))
    height = max(1, int(round((bounds.top - bounds.bottom) / resolution)))
    return from_origin(bounds.left, bounds.top, resolution, resolution), width, height


def _observation_time(root: Path, configured: str | np.datetime64 | None) -> np.datetime64:
    if configured is not None:
        return np.datetime64(configured, "ns")
    match = re.search(r"20\d{6}T\d{6}", root.name)
    if not match:
        raise ValueError("Provide observation_time when the Sentinel-2 SAFE name has no acquisition timestamp")
    value = match.group(0)
    formatted = f"{value[:4]}-{value[4:6]}-{value[6:8]}T{value[9:11]}:{value[11:13]}:{value[13:15]}"
    return np.datetime64(formatted, "ns")


def _read_band(path: Path, target_transform, width: int, height: int, reproject, Resampling, *, categorical: bool, clip_bounds=None):
    rasterio, _, _, _ = _require_rasterio()
    with rasterio.open(path) as source:
        source_window = rasterio.windows.from_bounds(*clip_bounds, transform=source.transform) if clip_bounds is not None else None
        source_array = source.read(1, window=source_window).astype(np.float32) if source_window is not None else source.read(1).astype(np.float32)
        nodata = source.nodata
        if nodata is not None:
            source_array[source_array == nodata] = np.nan
        destination = np.full((height, width), np.nan, dtype=np.float32)
        source_transform = source.window_transform(source_window) if source_window is not None else source.transform
        reproject(
            source=source_array,
            destination=destination,
            src_transform=source_transform,
            src_crs=source.crs,
            dst_transform=target_transform,
            dst_crs=source.crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.nearest if categorical else Resampling.bilinear,
        )
        return destination, source.crs


def read_s2_l2a(
    path: str | Path,
    *,
    config: Sentinel2ReadConfig | None = None,
    extract_dir: str | Path | None = None,
    aoi: AOI | None = None,
) -> xr.Dataset:
    """Read one Sentinel-2 L2A granule into a common xarray Dataset."""

    rasterio, Resampling, from_origin, reproject = _require_rasterio()
    config = config or Sentinel2ReadConfig()
    root = _safe_root(path, extract_dir)
    first_path = _band_file(root, config.bands[0], config.target_resolution_m)
    with rasterio.open(first_path) as source:
        clip_bounds = None
        if aoi is not None:
            from pyproj import Transformer

            transformer = Transformer.from_crs("EPSG:4326", source.crs, always_xy=True)
            xs, ys = transformer.transform(
                [aoi.west, aoi.east, aoi.east, aoi.west],
                [aoi.south, aoi.south, aoi.north, aoi.north],
            )
            requested = (min(xs), min(ys), max(xs), max(ys))
            clip_bounds = type(source.bounds)(
                max(source.bounds.left, requested[0]),
                max(source.bounds.bottom, requested[1]),
                min(source.bounds.right, requested[2]),
                min(source.bounds.top, requested[3]),
            )
            if clip_bounds.left >= clip_bounds.right or clip_bounds.bottom >= clip_bounds.top:
                raise ValueError(f"Sentinel-2 product does not overlap AOI: {aoi}")
        target_transform, width, height = _target_grid(source, config.target_resolution_m, from_origin, clip_bounds)
        crs = source.crs

    variables: dict[str, tuple[tuple[str, str], np.ndarray]] = {}
    for band in config.bands:
        source_path = _band_file(root, band, config.target_resolution_m)
        array, band_crs = _read_band(
            source_path,
            target_transform,
            width,
            height,
            reproject,
            Resampling,
            categorical=band == "SCL",
            clip_bounds=clip_bounds,
        )
        if str(band_crs) != str(crs):
            raise ValueError(f"Band {band} uses CRS {band_crs}, expected {crs}")
        if band != "SCL" and config.scale_reflectance:
            array = array / 10000.0
        variables[band] = (("y", "x"), array)

    x = target_transform.c + (np.arange(width) + 0.5) * target_transform.a
    y = target_transform.f + (np.arange(height) + 0.5) * target_transform.e
    dataset = xr.Dataset(variables, coords={"x": x, "y": y}).expand_dims(
        time=[_observation_time(root, config.observation_time)]
    )
    dataset.attrs.update({
        "product_name": root.name,
        "sensor": "Sentinel-2 MSI",
        "product_type": SENTINEL2_PRODUCT_TYPE,
        "crs": crs.to_string(),
        "resolution_m": config.target_resolution_m,
        "reflectance_scale": "1/10000" if config.scale_reflectance else "raw",
    })
    if "SCL" in dataset:
        dataset["SCL"].attrs.update({"flag_values": list(S2_SCL_CLASSES), "flag_meanings": " ".join(S2_SCL_CLASSES.values())})
        dataset["valid_mask"] = scl_valid_mask(dataset["SCL"])
    return apply_variable_contract(dataset, sensor="Sentinel-2", product=SENTINEL2_PRODUCT_TYPE, source=str(root))


def scl_valid_mask(scl: xr.DataArray) -> xr.DataArray:
    """Return the standard clear-land/water mask from SCL classes."""

    invalid = (0, 1, 2, 3, 8, 9, 10, 11)
    mask = xr.ones_like(scl, dtype=bool)
    for value in invalid:
        mask &= scl != value
    return mask.rename("valid_mask")


def sentinel2_indices(dataset: xr.Dataset) -> xr.Dataset:
    """Compute common Sentinel-2 spectral indices from reflectance bands."""

    required = {"B02", "B03", "B04", "B08", "B11", "B12"}
    missing = sorted(required.difference(dataset.data_vars))
    if missing:
        raise KeyError(f"Missing bands for Sentinel-2 indices: {missing}")

    def ratio(numerator: xr.DataArray, denominator: xr.DataArray) -> xr.DataArray:
        return (numerator / denominator.where(denominator != 0))

    result = dataset.copy()
    for name in result.data_vars:
        if isinstance(name, str) and (name.startswith("B") or name == "SCL"):
            result[name].attrs.setdefault("units", "1")
    result["NDVI"] = ratio(dataset["B08"] - dataset["B04"], dataset["B08"] + dataset["B04"])
    result["EVI"] = 2.5 * ratio(
        dataset["B08"] - dataset["B04"],
        dataset["B08"] + 6 * dataset["B04"] - 7.5 * dataset["B02"] + 1,
    )
    result["NDMI"] = ratio(dataset["B08"] - dataset["B11"], dataset["B08"] + dataset["B11"])
    result["NDBI"] = ratio(dataset["B11"] - dataset["B08"], dataset["B11"] + dataset["B08"])
    result["MNDWI"] = ratio(dataset["B03"] - dataset["B11"], dataset["B03"] + dataset["B11"])
    for name in ("NDVI", "EVI", "NDMI", "NDBI", "MNDWI"):
        result[name].attrs.update({"long_name": f"Sentinel-2 {name}", "units": "1"})
    return apply_variable_contract(
        result,
        sensor="Sentinel-2",
        product=SENTINEL2_PRODUCT_TYPE,
        source=result.attrs.get("source", ""),
    )


def sentinel2_coverage(dataset: xr.Dataset) -> xr.Dataset:
    """Return per-acquisition clear-pixel coverage diagnostics."""

    if "time" not in dataset.dims:
        raise ValueError("Sentinel-2 coverage requires a time dimension")
    if "valid_mask" in dataset:
        valid = dataset["valid_mask"]
    else:
        first = next(name for name in dataset.data_vars if name != "SCL")
        valid = xr.apply_ufunc(np.isfinite, dataset[first])
    return xr.Dataset({
        "clear_pixel_count": valid.sum(dim=("y", "x")),
        "clear_fraction": valid.mean(dim=("y", "x")),
    })


def compose_s2(
    datasets: xr.Dataset,
    *,
    method: str = "median",
    min_observations: int = 1,
) -> xr.Dataset:
    """Build a cloud-aware temporal Sentinel-2 composite."""

    if "time" not in datasets.dims:
        raise ValueError("Sentinel-2 compositing requires a time dimension")
    if method not in {"median", "mean"}:
        raise ValueError("method must be 'median' or 'mean'")
    if min_observations < 1:
        raise ValueError("min_observations must be positive")
    if "valid_mask" in datasets:
        valid = datasets["valid_mask"]
    else:
        first = next(name for name in datasets.data_vars if name != "SCL")
        valid = xr.apply_ufunc(np.isfinite, datasets[first])
    variables: dict[str, xr.DataArray] = {}
    for name, value in datasets.data_vars.items():
        if name in {"valid_mask", "SCL"} or value.dtype.kind not in "fiu":
            continue
        masked = value.where(valid)
        variables[name] = getattr(masked, method)(dim="time", skipna=True)
    count = valid.sum(dim="time")
    result = xr.Dataset(variables)
    result["clear_observation_count"] = count
    result["valid_mask"] = count >= min_observations
    result.attrs.update({
        **datasets.attrs,
        "composite_sensor": "Sentinel-2",
        "composite_method": method,
        "composite_min_observations": min_observations,
        "composite_observations": datasets.sizes["time"],
    })
    return result


class Sentinel2Catalog:
    """Search Sentinel-2 L2A products through the CDSE OData catalogue."""

    def __init__(self, config: ClientConfig | None = None, *, session: requests.Session | None = None):
        self.config = config
        self.session = session or http_session()
        self.catalog_url = config.catalog_url if config else "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        self.download_url = config.download_url if config else "https://download.dataspace.copernicus.eu/odata/v1/Products"

    def search(
        self,
        aoi: AOI,
        start: str | date | datetime,
        end: str | date | datetime,
        *,
        cloud_cover_max: float | None = None,
        limit: int = 100,
    ) -> list:
        """Return online Sentinel-2 L2A products intersecting an AOI."""

        def iso(value: str | date | datetime, *, end_of_day: bool = False) -> str:
            date_only = False
            if isinstance(value, datetime):
                parsed = value
            elif isinstance(value, date):
                parsed = datetime.combine(value, datetime.min.time())
                date_only = True
            else:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                date_only = "T" not in value and " " not in value
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            if end_of_day and date_only:
                parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
            return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

        start_iso, end_iso = iso(start), iso(end, end_of_day=True)
        filters = [
            f"Collection/Name eq '{SENTINEL2_COLLECTION}'",
            "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq "
            f"'productType' and att/OData.CSC.StringAttribute/Value eq '{SENTINEL2_PRODUCT_TYPE}')",
            f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi.as_wkt()}')",
            f"ContentDate/Start lt {end_iso}",
            f"ContentDate/End gt {start_iso}",
            "Online eq true",
        ]
        products: list[ProductRef] = []
        url: str | None = self.catalog_url
        params: dict[str, Any] | None = {
            "$filter": " and ".join(filters),
            "$expand": "Attributes",
            "$orderby": "ContentDate/Start asc",
            "$top": min(limit, 1000),
        }
        while url and len(products) < limit:
            response = self.session.get(url, params=params, timeout=120)
            response.raise_for_status()
            payload = response.json()
            for item in payload.get("value", []):
                product = _product_ref(item, self.download_url)
                if cloud_cover_max is None or float(product.metadata.get("attributes", {}).get("cloudCover", 101)) <= cloud_cover_max:
                    products.append(product)
                    if len(products) >= limit:
                        break
            url = payload.get("@odata.nextLink")
            params = None
        return products
