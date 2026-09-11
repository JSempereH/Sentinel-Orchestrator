"""Geolocation and conservative binning for Sentinel-3 LST swaths."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import xarray as xr

from ...cube import AnalysisGrid, CubeValidationError
from ...metadata import apply_variable_contract


def _bin_mean(values: np.ndarray, rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.int32)
    np.add.at(sums, (rows[finite], cols[finite]), values[finite])
    np.add.at(counts, (rows[finite], cols[finite]), 1)
    result = np.full(shape, np.nan, dtype=np.float32)
    np.divide(sums, counts, out=result, where=counts > 0)
    return result, counts


def _bin_flags(values: np.ndarray, rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    # Flag variables such as cloud_flags are commonly float on disk (CF
    # conventions promote integer bitmasks to float to represent missing
    # values as NaN), which numpy's bitwise ufuncs reject outright. Values
    # reaching here are already finite (see the `finite` filter below), so
    # it is always safe to accumulate them as integers.
    output = np.zeros(shape, dtype=np.int64)
    finite = np.isfinite(values)
    for row, col, value in zip(rows[finite], cols[finite], values[finite]):
        output[row, col] |= int(value)
    return output


def _require_single_geolocation(dataset: xr.Dataset, n_time: int) -> None:
    """Both binning methods use one geolocation grid for every time step in
    ``dataset``. That is only valid for a single Sentinel-3 acquisition (or
    several that happen to share identical swath geometry): each raw
    acquisition has its own instrument geometry, so silently reusing one
    acquisition's latitude/longitude for the others would mis-georeference
    every slice but the first, with no error raised. Georeference each
    acquisition individually and concatenate the regridded, common-grid
    results instead of concatenating raw acquisitions before regridding.
    """

    if n_time <= 1:
        return
    for name in ("latitude", "longitude"):
        values = dataset[name]
        if "time" not in values.dims:
            raise CubeValidationError(
                f"grid_l2_lst received {n_time} time steps but a single shared "
                f"{name!r} geolocation array - georeference each Sentinel-3 "
                "acquisition individually (e.g. one grid_l2_lst call per archive) "
                "before concatenating the regridded results along time, rather "
                "than concatenating raw acquisitions before regridding."
            )
        first = values.isel(time=0).values
        if not np.array_equal(values.values, np.broadcast_to(first, values.values.shape), equal_nan=True):
            raise CubeValidationError(
                f"grid_l2_lst received {n_time} time steps with differing {name!r} "
                "geolocation - georeference each Sentinel-3 acquisition "
                "individually before concatenating the regridded results along "
                "time, rather than concatenating raw acquisitions before regridding."
            )


def _grid_point_l2_lst(
    dataset: xr.Dataset,
    grid: AnalysisGrid,
    *,
    flag_variables: Iterable[str] = ("exception_flags", "cloud_flags"),
) -> xr.Dataset:
    """Project Sentinel-3 geolocation and bin native observations to a grid.

    This is point binning, not image interpolation. The output exposes the
    observation count and explicitly records that pixel footprints were not
    reconstructed from the instrument geometry.
    """

    required = {"lst", "latitude", "longitude"}
    missing = sorted(required.difference(dataset.data_vars))
    if missing:
        raise CubeValidationError(f"Sentinel-3 LST geolocation requires variables: {missing}")
    if "time" not in dataset.dims:
        dataset = dataset.expand_dims(time=["unknown"])
    _require_single_geolocation(dataset, dataset.sizes["time"])
    latitude = dataset["latitude"]
    longitude = dataset["longitude"]
    if "time" in latitude.dims:
        latitude = latitude.isel(time=0)
    if "time" in longitude.dims:
        longitude = longitude.isel(time=0)
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise RuntimeError("Install pyproj to georeference Sentinel-3 LST") from exc
    transformer = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    latitude_values, longitude_values = np.broadcast_arrays(latitude.values, longitude.values)
    projected_x, projected_y = transformer.transform(longitude_values, latitude_values)
    left, bottom, right, top = grid.bounds
    projected_x = np.asarray(projected_x)
    projected_y = np.asarray(projected_y)
    finite_coordinates = np.isfinite(projected_x) & np.isfinite(projected_y)
    cols = np.zeros(projected_x.shape, dtype=int)
    rows = np.zeros(projected_y.shape, dtype=int)
    cols[finite_coordinates] = np.floor((projected_x[finite_coordinates] - left) / grid.resolution[0]).astype(int)
    rows[finite_coordinates] = np.floor((top - projected_y[finite_coordinates]) / grid.resolution[1]).astype(int)
    inside = finite_coordinates & (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
    rows = rows[inside].ravel()
    cols = cols[inside].ravel()
    shape = (grid.height, grid.width)
    output: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    observation_count = np.zeros((dataset.sizes["time"], grid.height, grid.width), dtype=np.int32)
    flag_names = set(flag_variables)
    for name, value in dataset.data_vars.items():
        if name in {"latitude", "longitude"} or not {"y", "x"}.issubset(value.dims):
            continue
        layers = value.transpose("time", "y", "x").values
        binned_layers: list[np.ndarray] = []
        for index, layer in enumerate(layers):
            flattened = np.asarray(layer).ravel()[inside.ravel()]
            if name in flag_names:
                binned = _bin_flags(flattened, rows, cols, shape)
            else:
                binned, counts = _bin_mean(flattened, rows, cols, shape)
                if name == "lst":
                    observation_count[index] = counts
            binned_layers.append(binned)
        output[name] = (("time", "y", "x"), np.stack(binned_layers))
    # rows are already north-up because the target row index is top-origin.
    result = xr.Dataset(
        output,
        coords={"time": dataset.time, "y": grid.y, "x": grid.x},
        attrs={**dataset.attrs, "grid_id": grid.grid_id, "crs": grid.crs, "resolution_m": grid.resolution_m},
    )
    result["lst_observation_count"] = (("time", "y", "x"), observation_count)
    result["valid_mask"] = np.isfinite(result["lst"]) & (result["lst_observation_count"] > 0)
    result.attrs.update({
        "analysis_shape": "regular_grid",
        "geolocation_method": "projected_point_binning",
        "pixel_footprints_reconstructed": False,
    })
    return apply_variable_contract(result, sensor="Sentinel-3", product="SL_2_LST", aggregation_method="area_weighted", source=str(dataset.attrs.get("source", "")))


def _corner_grid(values: np.ndarray) -> np.ndarray:
    """Estimate curvilinear pixel corners from geolocation centres."""

    height, width = values.shape
    corners = np.empty((height + 1, width + 1), dtype=float)
    if height > 1 and width > 1:
        corners[1:-1, 1:-1] = (
            values[:-1, :-1] + values[1:, :-1] + values[:-1, 1:] + values[1:, 1:]
        ) / 4
    top = values[0] + (values[0] - values[1]) / 2 if height > 1 else values[0]
    bottom = values[-1] + (values[-1] - values[-2]) / 2 if height > 1 else values[-1]
    left = values[:, 0] + (values[:, 0] - values[:, 1]) / 2 if width > 1 else values[:, 0]
    right = values[:, -1] + (values[:, -1] - values[:, -2]) / 2 if width > 1 else values[:, -1]
    corners[0, 1:-1] = (top[:-1] + top[1:]) / 2
    corners[-1, 1:-1] = (bottom[:-1] + bottom[1:]) / 2
    corners[1:-1, 0] = (left[:-1] + left[1:]) / 2
    corners[1:-1, -1] = (right[:-1] + right[1:]) / 2
    corners[0, 0] = top[0] - (top[1] - top[0]) / 2 if width > 1 else top[0]
    corners[0, -1] = top[-1] + (top[-1] - top[-2]) / 2 if width > 1 else top[-1]
    corners[-1, 0] = bottom[0] - (bottom[1] - bottom[0]) / 2 if width > 1 else bottom[0]
    corners[-1, -1] = bottom[-1] + (bottom[-1] - bottom[-2]) / 2 if width > 1 else bottom[-1]
    return corners


def _grid_area_l2_lst(dataset: xr.Dataset, grid: AnalysisGrid) -> xr.Dataset:
    """Area-weighted aggregation using reconstructed geolocation footprints."""

    try:
        from shapely.geometry import Polygon, box
        from shapely.strtree import STRtree
    except ImportError as exc:
        raise RuntimeError("Install sentinel-analysis[geo] for Sentinel-3 footprint aggregation") from exc
    required = {"lst", "latitude", "longitude"}
    missing = sorted(required.difference(dataset.data_vars))
    if missing:
        raise CubeValidationError(f"Sentinel-3 LST geolocation requires variables: {missing}")
    if "time" not in dataset.dims:
        dataset = dataset.expand_dims(time=["unknown"])
    _require_single_geolocation(dataset, dataset.sizes["time"])
    latitude = dataset["latitude"].isel(time=0) if "time" in dataset["latitude"].dims else dataset["latitude"]
    longitude = dataset["longitude"].isel(time=0) if "time" in dataset["longitude"].dims else dataset["longitude"]
    lat_values, lon_values = np.broadcast_arrays(latitude.values, longitude.values)
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise RuntimeError("Install pyproj to georeference Sentinel-3 LST") from exc
    transformer = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    px, py = transformer.transform(lon_values, lat_values)
    px, py = np.asarray(px), np.asarray(py)
    corner_x, corner_y = _corner_grid(px), _corner_grid(py)
    source_polygons: list[tuple[int, int, Polygon]] = []
    for row in range(px.shape[0]):
        for col in range(px.shape[1]):
            corners = [
                (corner_x[row, col], corner_y[row, col]),
                (corner_x[row, col + 1], corner_y[row, col + 1]),
                (corner_x[row + 1, col + 1], corner_y[row + 1, col + 1]),
                (corner_x[row + 1, col], corner_y[row + 1, col]),
            ]
            polygon = Polygon(corners)
            if polygon.is_valid and polygon.area > 0:
                source_polygons.append((row, col, polygon))
    target_polygons = [
        box(grid.bounds[0] + col * grid.resolution[0], grid.bounds[1] + row * grid.resolution[1],
            grid.bounds[0] + (col + 1) * grid.resolution[0], grid.bounds[1] + (row + 1) * grid.resolution[1])
        for row in range(grid.height) for col in range(grid.width)
    ]
    tree = STRtree(target_polygons)
    target_cell_area = grid.resolution[0] * grid.resolution[1]
    continuous = {
        name: value for name, value in dataset.data_vars.items()
        if name not in {"latitude", "longitude", "valid_mask"} and {"y", "x"}.issubset(value.dims)
    }
    flag_names = {name for name in continuous if name.endswith("flags")}
    output = {name: np.full((dataset.sizes["time"], grid.height, grid.width), np.nan, dtype=np.float32) for name in continuous if name not in flag_names}
    # Flag variables such as cloud_flags are commonly float on disk (CF
    # conventions promote integer bitmasks to float to represent missing
    # values as NaN), which numpy's bitwise ufuncs reject outright - always
    # accumulate as integers regardless of the source dtype (see below,
    # values are cast to int right before the bitwise-or, after the
    # isfinite check that already guarantees a real flag value).
    flags = {name: np.zeros((dataset.sizes["time"], grid.height, grid.width), dtype=np.int64) for name in continuous if name in flag_names}
    coverage = np.zeros((dataset.sizes["time"], grid.height, grid.width), dtype=np.float32)
    valid_count = np.zeros_like(coverage, dtype=np.int32)
    invalid_count = np.zeros_like(coverage, dtype=np.int32)
    footprint_count = np.zeros_like(coverage, dtype=np.int32)
    weights = {name: np.zeros_like(coverage, dtype=np.float64) for name in output}
    sums = {name: np.zeros_like(coverage, dtype=np.float64) for name in output}
    for source_row, source_col, polygon in source_polygons:
        for target_index in tree.query(polygon, predicate="intersects"):
            target_row, target_col = divmod(int(target_index), grid.width)
            intersection_area = polygon.intersection(target_polygons[int(target_index)]).area
            if intersection_area <= max(1e-8, target_cell_area * 1e-10):
                continue
            weight = intersection_area / target_cell_area
            coverage[:, target_row, target_col] = np.minimum(1, coverage[:, target_row, target_col] + weight)
            footprint_count[:, target_row, target_col] += 1
            for time_index in range(dataset.sizes["time"]):
                for name, value in continuous.items():
                    cell_value = value.isel(time=time_index).values[source_row, source_col]
                    if name in flag_names:
                        if np.isfinite(cell_value):
                            flags[name][time_index, target_row, target_col] |= int(cell_value)
                    elif np.isfinite(cell_value):
                        sums[name][time_index, target_row, target_col] += float(cell_value) * weight
                        weights[name][time_index, target_row, target_col] += weight
                        if name == "lst":
                            valid_count[time_index, target_row, target_col] += 1
                    elif name == "lst":
                        invalid_count[time_index, target_row, target_col] += 1
    for name in output:
        np.divide(sums[name], weights[name], out=output[name], where=weights[name] > 0)
    result = xr.Dataset(
        {name: (("time", "y", "x"), values) for name, values in output.items()},
        coords={"time": dataset.time, "y": grid.y, "x": grid.x},
        attrs={**dataset.attrs, "grid_id": grid.grid_id, "crs": grid.crs, "resolution_m": grid.resolution_m},
    )
    for name, values in flags.items():
        result[name] = (("time", "y", "x"), values)
    result["coverage_fraction"] = (("time", "y", "x"), coverage)
    result["lst_observation_count"] = (("time", "y", "x"), valid_count)
    result["lst_invalid_observation_count"] = (("time", "y", "x"), invalid_count)
    result["source_footprint_count"] = (("time", "y", "x"), footprint_count)
    result["no_observation_mask"] = result["coverage_fraction"] == 0
    result["valid_mask"] = np.isfinite(result["lst"]) & (result["lst_observation_count"] > 0)
    result.attrs.update({
        "analysis_shape": "regular_grid",
        "geolocation_method": "area_weighted_reconstructed_footprints",
        "pixel_footprints_reconstructed": True,
        "coverage_is_capped_at_one": True,
    })
    return apply_variable_contract(result, sensor="Sentinel-3", product="SL_2_LST", aggregation_method="area_weighted", source=str(dataset.attrs.get("source", "")))


def grid_l2_lst(
    dataset: xr.Dataset,
    grid: AnalysisGrid,
    *,
    method: str = "area",
    flag_variables: Iterable[str] = ("exception_flags", "cloud_flags"),
) -> xr.Dataset:
    """Georeference LST using area footprints or explicit point fallback."""

    if method == "area":
        try:
            return _grid_area_l2_lst(dataset, grid)
        except ImportError:
            raise
    if method == "point":
        return _grid_point_l2_lst(dataset, grid, flag_variables=flag_variables)
    raise ValueError("method must be 'area' or 'point'")
