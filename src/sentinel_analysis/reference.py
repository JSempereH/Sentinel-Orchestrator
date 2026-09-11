"""Collocation and validation helpers for stations and reference datasets."""

from __future__ import annotations


import xarray as xr

from .cube import validate_cube
from .validation import compare_to_reference


def collocate_stations(
    cube: xr.Dataset,
    stations: xr.Dataset,
    *,
    variable: str,
    station_variable: str,
) -> dict[str, dict[str, float | int | str]]:
    """Collocate a regular cube with station observations by nearest pixel/time.

    Stations must contain `station`, `time`, `latitude` and `longitude`.
    Spatial coordinates are transformed to the cube CRS before nearest lookup.
    """

    validate_cube(cube, required_variables=(variable,), require_time=True)
    required = {"station", "time", "latitude", "longitude", station_variable}
    missing = sorted(required.difference(stations.dims).difference(stations.coords).difference(stations.data_vars))
    if missing:
        raise ValueError(f"Station dataset is missing: {missing}")
    if "station" not in stations.dims or "time" not in stations.dims:
        raise ValueError("Station dataset requires station and time dimensions")
    transformer = None
    if cube.attrs.get("crs") != "EPSG:4326":
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", cube.attrs["crs"], always_xy=True)
    reports: dict[str, dict[str, float | int | str]] = {}
    for station_index, station_name in enumerate(stations.station.values):
        longitude = float(stations.longitude.isel(station=station_index).item())
        latitude = float(stations.latitude.isel(station=station_index).item())
        if transformer:
            longitude, latitude = transformer.transform(longitude, latitude)
        estimate = cube[variable].sel(x=longitude, y=latitude, method="nearest")
        observed = stations[station_variable].isel(station=station_index)
        estimate, observed = xr.align(estimate, observed, join="inner")
        reports[str(station_name)] = compare_to_reference(estimate, observed, name=variable)
    return reports
