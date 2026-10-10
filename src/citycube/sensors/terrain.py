"""Terrain predictors for thermal downscaling: elevation, slope, aspect and
solar illumination.

Surface temperature falls with elevation (roughly the environmental lapse
rate) and depends on how directly the sun hits a slope at acquisition time.
ESA's Sen-ET sharpening chain feeds exactly these to its Data Mining
Sharpener alongside Sentinel-2 reflectance (elevation plus the cosine of the
solar incidence angle on the tilted surface); without them a model has to
explain a mountain city's thermal pattern from vegetation indices alone.

The DEM is Copernicus GLO-30, read in place from Planetary Computer's
cloud-optimized GeoTIFFs (no account needed to search; assets are signed with
the ``landsat`` extra's ``planetary-computer``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from ..cube import AnalysisGrid
from ..stac import STACCatalog
from .cog import read_cog_to_grid, sign_href

COP_DEM_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"
COP_DEM_COLLECTION = "cop-dem-glo-30"


def _grid_lonlat_bounds(grid: AnalysisGrid) -> tuple[float, float, float, float]:
    from pyproj import Transformer

    return Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True).transform_bounds(*grid.bounds, densify_pts=21)


def load_dem(grid: AnalysisGrid, *, catalog: Any | None = None, collection: str = COP_DEM_COLLECTION) -> xr.DataArray:
    """Elevation (m) on ``grid``, area-averaged from every intersecting DEM tile."""

    catalog = catalog or STACCatalog(COP_DEM_ENDPOINT)
    items = catalog.search(collections=[collection], bbox=list(_grid_lonlat_bounds(grid)), limit=100)
    if not items:
        raise ValueError(f"No {collection} tiles intersect grid {grid.grid_id}")
    elevation = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
    for item in items:
        asset = item.assets.get("data") or next(iter(item.assets.values()))
        tile = read_cog_to_grid(sign_href(asset["href"]), grid, resampling="average")
        elevation = np.where(np.isfinite(elevation), elevation, tile)
    return xr.DataArray(
        elevation, dims=("y", "x"), coords={"y": grid.y, "x": grid.x}, name="elevation",
        attrs={"units": "m", "long_name": "surface elevation", "source": collection, "crs": grid.crs},
    )


def slope_aspect(elevation: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Slope (degrees from horizontal) and aspect (degrees clockwise from
    north, the direction the slope faces) on a projected, metric grid."""

    x = elevation["x"].values
    y = elevation["y"].values
    dz_dy, dz_dx = np.gradient(elevation.values.astype(np.float64), y, x)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    # Downslope direction is -(dz_dx, dz_dy); its compass bearing is the aspect.
    aspect = (np.degrees(np.arctan2(-dz_dx, -dz_dy)) + 360.0) % 360.0
    coords = {"y": elevation.y, "x": elevation.x}
    return (
        xr.DataArray(slope.astype(np.float32), dims=("y", "x"), coords=coords, name="slope", attrs={"units": "degree"}),
        xr.DataArray(aspect.astype(np.float32), dims=("y", "x"), coords=coords, name="aspect", attrs={"units": "degree"}),
    )


def solar_position(time: np.datetime64, latitude: np.ndarray, longitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solar zenith and azimuth (degrees) with NOAA's general solar position
    equations (~0.5 degree accuracy, ample for illumination predictors)."""

    timestamp = np.datetime64(time, "s").astype("datetime64[s]").astype(object)
    day_of_year = timestamp.timetuple().tm_yday
    hours = timestamp.hour + timestamp.minute / 60 + timestamp.second / 3600
    gamma = 2 * np.pi / 365 * (day_of_year - 1 + (hours - 12) / 24)
    equation_of_time = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma) - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    declination = (
        0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma) - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma) - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)
    )
    true_solar_minutes = hours * 60 + equation_of_time + 4 * np.asarray(longitude)
    hour_angle = np.radians(true_solar_minutes / 4 - 180)
    lat = np.radians(np.asarray(latitude))
    cos_zenith = np.sin(lat) * np.sin(declination) + np.cos(lat) * np.cos(declination) * np.cos(hour_angle)
    zenith = np.arccos(np.clip(cos_zenith, -1, 1))
    azimuth = np.degrees(np.arctan2(
        np.sin(hour_angle),
        np.cos(hour_angle) * np.sin(lat) - np.tan(declination) * np.cos(lat),
    )) + 180.0
    return np.degrees(zenith), azimuth % 360.0


def terrain_predictors(static: xr.Dataset, times, *, crs: str) -> xr.Dataset:
    """Static terrain plus the cosine of the solar incidence angle on each
    tilted pixel at every time in ``times``.

    ``static`` holds ``elevation``, ``slope`` and ``aspect`` (see
    ``terrain_static``). ``cos_incidence`` is the illumination predictor
    Sen-ET feeds its sharpener: 1 for a surface facing the sun, cos(solar
    zenith) for flat ground, negative for self-shaded slopes.
    """

    from pyproj import Transformer

    xx, yy = np.meshgrid(static["x"].values, static["y"].values)
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(xx, yy)
    slope = np.radians(static["slope"].values)
    aspect = np.radians(static["aspect"].values)
    layers = []
    for time in np.atleast_1d(np.asarray(times, dtype="datetime64[ns]")):
        zenith, azimuth = solar_position(time, lat, lon)
        zenith, azimuth = np.radians(zenith), np.radians(azimuth)
        layers.append(np.cos(zenith) * np.cos(slope) + np.sin(zenith) * np.sin(slope) * np.cos(azimuth - aspect))
    time_coord = np.atleast_1d(np.asarray(times, dtype="datetime64[ns]"))
    result = static.expand_dims(time=time_coord).copy()
    result["cos_incidence"] = (("time", "y", "x"), np.asarray(layers, dtype=np.float32))
    result["cos_incidence"].attrs.update({"units": "1", "long_name": "cosine of the solar incidence angle on the terrain"})
    result.attrs["crs"] = crs
    return result


def terrain_static(grid: AnalysisGrid, *, catalog: Any | None = None) -> xr.Dataset:
    """Elevation, slope and aspect on ``grid`` (no time dimension)."""

    elevation = load_dem(grid, catalog=catalog)
    slope, aspect = slope_aspect(elevation)
    return xr.Dataset({"elevation": elevation, "slope": slope, "aspect": aspect}, attrs={"crs": grid.crs, "grid_id": grid.grid_id})
