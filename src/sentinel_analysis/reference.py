"""Collocation and validation helpers for stations and reference datasets."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .cube import validate_cube
from .validation import compare_to_reference


def collocate_stations(
    cube: xr.Dataset,
    stations: xr.Dataset,
    *,
    variable: str,
    station_variable: str,
    temporal_tolerance: np.timedelta64 = np.timedelta64(1, "h"),
) -> dict[str, dict[str, float | int | str]]:
    """Collocate a regular cube with station observations by nearest pixel and time.

    Stations must contain `station`, `time`, `latitude` and `longitude`.
    Spatial coordinates are transformed to the cube CRS before the nearest
    pixel lookup. Each cube time is paired with the station's nearest
    observation within ``temporal_tolerance`` (satellite and station clocks
    never coincide exactly). A station outside the cube, or with no
    observation close enough in time, is reported with ``samples`` 0 and a
    ``note`` instead of failing the whole collocation.
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
    half_x = float(abs(cube.x.values[1] - cube.x.values[0])) / 2 if cube.x.size > 1 else 0.0
    half_y = float(abs(cube.y.values[1] - cube.y.values[0])) / 2 if cube.y.size > 1 else 0.0
    station_table = stations.sortby("time").drop_duplicates("time")
    reports: dict[str, dict[str, float | int | str]] = {}
    for station_index, station_name in enumerate(station_table.station.values):
        longitude = float(station_table.longitude.isel(station=station_index).item())
        latitude = float(station_table.latitude.isel(station=station_index).item())
        if transformer:
            longitude, latitude = transformer.transform(longitude, latitude)
        inside = (float(cube.x.min()) - half_x <= longitude <= float(cube.x.max()) + half_x
                  and float(cube.y.min()) - half_y <= latitude <= float(cube.y.max()) + half_y)
        if not inside:
            reports[str(station_name)] = {"variable": variable, "samples": 0, "note": "station outside the cube"}
            continue
        estimate = cube[variable].sel(x=longitude, y=latitude, method="nearest")
        observed = station_table[station_variable].isel(station=station_index).reindex(time=estimate.time, method="nearest", tolerance=str(temporal_tolerance.astype("timedelta64[s]").astype(int)) + "s")
        try:
            reports[str(station_name)] = compare_to_reference(estimate, observed, name=variable)
        except ValueError:
            reports[str(station_name)] = {"variable": variable, "samples": 0, "note": f"no observation within {temporal_tolerance} of a cube time"}
    return reports
