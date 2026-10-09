"""Sentinel-5P/TROPOMI trace-gas observations and atmospheric composition."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import requests
import xarray as xr

from ..catalog import ProductRef, _product_ref
from ..config import AOI, ClientConfig
from ..cube import validate_observation_set
from ..http import http_session
from ..metadata import apply_variable_contract


SENTINEL5P_COLLECTION = "SENTINEL-5P"
SENTINEL5P_PRODUCT_TYPES = {
    "NO2": "L2__NO2___",
    "CO": "L2__CO____",
    "SO2": "L2__SO2___",
    "O3": "L2__O3____",
    "CH4": "L2__CH4___",
    "HCHO": "L2__HCHO__",
    "AER_AI": "L2__AER_AI",
}
SENTINEL5P_VARIABLES = {
    "NO2": "nitrogendioxide_tropospheric_column",
    "CO": "carbonmonoxide_total_column",
    "SO2": "sulfurdioxide_total_vertical_column",
    "O3": "ozone_total_vertical_column",
    "CH4": "methane_mixing_ratio",
    "HCHO": "formaldehyde_tropospheric_vertical_column",
    "AER_AI": "aerosol_index_354_388",
}


@dataclass(frozen=True)
class Sentinel5PReadConfig:
    """Quality and variable selection for one TROPOMI L2 product."""

    gas: str = "NO2"
    qa_threshold: float = 0.75
    observation_time: str | np.datetime64 | None = None
    chunks: Mapping[str, int] | str | None = None

    def __post_init__(self) -> None:
        gas = self.gas.upper()
        if gas not in SENTINEL5P_VARIABLES:
            raise ValueError(f"Unsupported Sentinel-5P product: {self.gas}")
        if not 0 <= self.qa_threshold <= 1:
            raise ValueError("qa_threshold must be between 0 and 1")


def _iso_datetime(value: str | date | datetime, *, end: bool = False) -> str:
    date_only = isinstance(value, date) and not isinstance(value, datetime)
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, datetime.min.time())
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        date_only = "T" not in value and " " not in value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if end and date_only:
        parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class Sentinel5PCatalog:
    """Search Sentinel-5P L2 trace-gas products in CDSE OData."""

    def __init__(self, config: ClientConfig | None = None, *, session: requests.Session | None = None):
        self.session = session or http_session()
        self.catalog_url = config.catalog_url if config else "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        self.download_url = config.download_url if config else "https://download.dataspace.copernicus.eu/odata/v1/Products"

    def search(
        self,
        aoi: AOI,
        start: str | date | datetime,
        end: str | date | datetime,
        *,
        gas: str = "NO2",
        limit: int = 100,
    ) -> list[ProductRef]:
        gas = gas.upper()
        try:
            product_type = SENTINEL5P_PRODUCT_TYPES[gas]
        except KeyError as exc:
            raise ValueError(f"Unsupported Sentinel-5P product: {gas}") from exc
        start_iso, end_iso = _iso_datetime(start), _iso_datetime(end, end=True)
        filters = [
            f"Collection/Name eq '{SENTINEL5P_COLLECTION}'",
            "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq "
            f"'productType' and att/OData.CSC.StringAttribute/Value eq '{product_type}')",
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
            products.extend(
                _product_ref(item, self.download_url, default_product_type=product_type)
                for item in payload.get("value", [])
            )
            products = products[:limit]
            url = payload.get("@odata.nextLink")
            params = None
        return deduplicate_orbits(products)


_S5P_ORBIT = re.compile(r"_\d{8}T\d{6}_\d{8}T\d{6}_(\d{5})_")
_S5P_STREAM_PRIORITY = {"RPRO": 0, "OFFL": 1, "NRTI": 2}


def deduplicate_orbits(products: list[ProductRef]) -> list[ProductRef]:
    """Keep one product per orbit, preferring reprocessed, then offline,
    then near-real-time processing.

    CDSE returns both the NRTI granules and the OFFL orbit file of the same
    overpass; using both duplicates every observation (and downloads the
    same orbit twice).
    """

    best: dict[str, ProductRef] = {}
    passthrough: list[ProductRef] = []
    for product in products:
        match = _S5P_ORBIT.search(product.name)
        if not match:
            passthrough.append(product)
            continue
        orbit = match.group(1)
        stream = product.name[4:8]
        current = best.get(orbit)
        if current is None or _S5P_STREAM_PRIORITY.get(stream, 9) < _S5P_STREAM_PRIORITY.get(current.name[4:8], 9):
            best[orbit] = product
    return [*best.values(), *passthrough]


def _open_product(path: str | Path, *, chunks: Mapping[str, int] | str | None = None) -> xr.Dataset:
    """Open the PRODUCT group used by TROPOMI L2 NetCDF files."""

    try:
        return xr.open_dataset(path, group="PRODUCT", chunks=chunks)
    except (OSError, ValueError):
        return xr.open_dataset(path, chunks=chunks)


def _observation_time(path: Path, configured: str | np.datetime64 | None) -> np.datetime64:
    if configured is not None:
        return np.datetime64(configured, "ns")
    for pattern in (r"20\d{6}T\d{6}", r"20\d{6}"):
        match = next(iter(re.finditer(pattern, path.name)), None)
        if match:
            value = match.group(0)
            value = (
                f"{value[:4]}-{value[4:6]}-{value[6:8]}T{value[9:11]}:{value[11:13]}:{value[13:15]}"
                if "T" in value and len(value) > 8
                else f"{value[:4]}-{value[4:6]}-{value[6:8]}T00:00:00"
            )
            return np.datetime64(value, "ns")
    raise ValueError("Provide observation_time when the Sentinel-5P filename has no acquisition date")


def read_s5p_l2(
    path: str | Path,
    *,
    config: Sentinel5PReadConfig | None = None,
    aoi: AOI | None = None,
    aoi_margin_deg: float = 0.1,
) -> xr.Dataset:
    """Read a Sentinel-5P L2 gas product as a quality-filtered swath.

    The result intentionally keeps latitude/longitude and an observation
    dimension. It is not silently treated as a regular raster; call
    :func:`grid_s5p` when an analysis grid is required.

    With ``aoi``, only observations within the AOI plus ``aoi_margin_deg``
    are kept and loaded (an OFFL product covers a whole orbit, ~600 MB, of
    which a city needs a few hundred pixels), and the observation index is
    flattened so the result can be written to NetCDF.
    """

    config = config or Sentinel5PReadConfig()
    gas = config.gas.upper()
    variable_name = SENTINEL5P_VARIABLES[gas]
    source = _open_product(path, chunks=config.chunks)
    if variable_name not in source:
        source.close()
        raise KeyError(f"Variable {variable_name!r} not found in {path}")
    value = source[variable_name]
    qa = source["qa_value"] if "qa_value" in source else xr.ones_like(value, dtype=float)
    latitude = source["latitude"]
    longitude = source["longitude"]
    spatial_dims = [dim for dim in value.dims if dim != "time"]
    if not spatial_dims:
        raise ValueError("Sentinel-5P product has no scanline/ground_pixel dimensions")
    result = xr.Dataset({
        gas: value.stack(observation=spatial_dims),
        "qa_value": qa.stack(observation=[dim for dim in qa.dims if dim != "time"]),
        "latitude": latitude.stack(observation=[dim for dim in latitude.dims if dim != "time"]),
        "longitude": longitude.stack(observation=[dim for dim in longitude.dims if dim != "time"]),
    })
    if aoi is not None:
        lat_values = np.asarray(result["latitude"].values).reshape(-1, result.sizes["observation"])[0]
        lon_values = np.asarray(result["longitude"].values).reshape(-1, result.sizes["observation"])[0]
        inside = (
            (lat_values >= aoi.south - aoi_margin_deg) & (lat_values <= aoi.north + aoi_margin_deg)
            & (lon_values >= aoi.west - aoi_margin_deg) & (lon_values <= aoi.east + aoi_margin_deg)
        )
        # deep copy so the crop owns its memory instead of viewing the whole orbit's arrays
        result = result.isel(observation=np.flatnonzero(inside)).reset_index("observation", drop=True).load().copy(deep=True)
        source.close()
    observation_time = _observation_time(Path(path), config.observation_time)
    if "time" not in result.dims:
        result = result.expand_dims(time=[observation_time])
    else:
        result = result.assign_coords(time=np.repeat(observation_time, result.sizes["time"]))
    valid = (
        (result["qa_value"] >= config.qa_threshold)
        & np.isfinite(result[gas])
        & np.isfinite(result["latitude"])
        & np.isfinite(result["longitude"])
    )
    result[gas] = result[gas].where(valid)
    result["valid_mask"] = valid
    result.attrs.update({
        "sensor": "Sentinel-5P TROPOMI",
        "product_type": SENTINEL5P_PRODUCT_TYPES[gas],
        "gas": gas,
        "qa_threshold": config.qa_threshold,
        "analysis_shape": "swath",
        "source": str(path),
    })
    result[gas].attrs.update({"long_name": f"Sentinel-5P {gas} column", "source_variable": variable_name})
    validate_observation_set(result, required_variables=(gas,))
    return apply_variable_contract(result, sensor="Sentinel-5P", product=SENTINEL5P_PRODUCT_TYPES[gas], source=str(path))


# Half the diagonal of a TROPOMI nadir pixel (5.5 x 3.5 km since 2019).
TROPOMI_FOOTPRINT_RADIUS_KM = 3.5


def _fill_footprints(grid: np.ndarray, lat_centres: np.ndarray, lon_centres: np.ndarray,
                     lat: np.ndarray, lon: np.ndarray, values: np.ndarray, radius_km: float) -> None:
    """Fill empty cells in place with the nearest pixel within ``radius_km``."""

    empty = np.isnan(grid)
    if not empty.any() or values.size == 0:
        return
    rows, cols = np.nonzero(empty)
    # Local equirectangular kilometres: accurate to well under 1 % over a swath's few kilometres.
    scale = np.cos(np.radians(np.nanmean(lat)))
    cells = np.column_stack([lat_centres[rows] * 110.574, lon_centres[cols] * 111.320 * scale])
    pixels = np.column_stack([lat * 110.574, lon * 111.320 * scale])
    try:
        from scipy.spatial import cKDTree

        distance, nearest = cKDTree(pixels).query(cells, distance_upper_bound=radius_km)
        found = np.isfinite(distance)
    except ImportError:  # small AOIs only: one distance per cell and pixel
        squared = ((cells[:, None, :] - pixels[None, :, :]) ** 2).sum(-1)
        nearest = squared.argmin(1)
        found = squared[np.arange(len(cells)), nearest] <= radius_km**2
    grid[rows[found], cols[found]] = values[nearest[found]]


def grid_s5p(
    dataset: xr.Dataset,
    *,
    resolution_deg: float = 0.01,
    aoi: AOI | None = None,
    footprint_radius_km: float | None = TROPOMI_FOOTPRINT_RADIUS_KM,
) -> xr.Dataset:
    """Grid a quality-filtered TROPOMI swath onto a regular EPSG:4326 grid.

    Cells containing pixel centres get their mean. A TROPOMI pixel covers
    several kilometres, far more than one 0.01 degree cell, so binning
    centres alone left most cells empty (a fully valid Berlin overpass
    filled 3.5 % of the grid). With ``footprint_radius_km`` the remaining
    cells take the nearest valid pixel whose centre lies within that
    distance, i.e. inside its footprint; ``None`` keeps centres only.
    """

    if resolution_deg <= 0:
        raise ValueError("resolution_deg must be positive")
    if footprint_radius_km is not None and footprint_radius_km <= 0:
        raise ValueError("footprint_radius_km must be positive")
    gas = str(dataset.attrs.get("gas", "NO2"))
    validate_observation_set(dataset, required_variables=(gas,))
    values = np.asarray(dataset[gas].values)
    lat = np.asarray(dataset["latitude"].values)
    lon = np.asarray(dataset["longitude"].values)
    if values.ndim == 1:
        values, lat, lon = values[None, :], lat[None, :], lon[None, :]
    valid = np.isfinite(values) & np.isfinite(lat) & np.isfinite(lon)
    valid &= np.asarray(dataset["valid_mask"].values, dtype=bool)
    if aoi is not None:
        # Pixels centred just outside the AOI still cover its edge.
        pad_lat = (footprint_radius_km or 0.0) / 110.574
        pad_lon = pad_lat / max(np.cos(np.radians((aoi.south + aoi.north) / 2)), 0.1)
        valid &= (lat >= aoi.south - pad_lat) & (lat <= aoi.north + pad_lat) & (lon >= aoi.west - pad_lon) & (lon <= aoi.east + pad_lon)
    if not valid.any():
        raise ValueError("Sentinel-5P swath has no valid observations in the requested AOI")
    if aoi is not None:
        lat_min, lat_max = aoi.south, aoi.north
        lon_min, lon_max = aoi.west, aoi.east
    else:
        lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
        lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
    y_edges = np.arange(np.floor(lat_min / resolution_deg) * resolution_deg, lat_max + resolution_deg, resolution_deg)
    x_edges = np.arange(np.floor(lon_min / resolution_deg) * resolution_deg, lon_max + resolution_deg, resolution_deg)
    if len(y_edges) < 2 or len(x_edges) < 2:
        raise ValueError("Swath does not cover at least one output grid cell")
    output = np.full((values.shape[0], len(y_edges) - 1, len(x_edges) - 1), np.nan, dtype=np.float32)
    for index in range(values.shape[0]):
        mask = valid[index]
        sums, _, _ = np.histogram2d(lat[index][mask], lon[index][mask], bins=(y_edges, x_edges), weights=values[index][mask])
        counts, _, _ = np.histogram2d(lat[index][mask], lon[index][mask], bins=(y_edges, x_edges))
        output[index] = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
        if footprint_radius_km is not None:
            _fill_footprints(output[index], (y_edges[:-1] + y_edges[1:]) / 2, (x_edges[:-1] + x_edges[1:]) / 2,
                             lat[index][mask], lon[index][mask], values[index][mask], footprint_radius_km)
    result = xr.Dataset(
        {gas: (("time", "y", "x"), output)},
        coords={
            "time": dataset.time,
            "y": (y_edges[:-1] + y_edges[1:]) / 2,
            "x": (x_edges[:-1] + x_edges[1:]) / 2,
        },
        attrs={**dataset.attrs, "analysis_shape": "regular_grid", "crs": "EPSG:4326", "resolution_deg": resolution_deg},
    )
    result["valid_mask"] = np.isfinite(result[gas])
    result.attrs["footprint_radius_km"] = footprint_radius_km if footprint_radius_km is not None else "none (pixel centres only)"
    return apply_variable_contract(
        result,
        sensor="Sentinel-5P",
        product=str(dataset.attrs.get("product_type", "Sentinel-5P")),
        source=dataset.attrs.get("source", ""),
    )
