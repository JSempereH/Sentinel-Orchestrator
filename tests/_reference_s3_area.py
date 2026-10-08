"""Verbatim copy of the original per-pixel ``_grid_area_l2_lst``.

Kept only as a slow but trusted reference: the vectorized implementation in
``sensors/sentinel3/georeference.py`` must reproduce it exactly.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from sentinel_analysis.cube import AnalysisGrid, CubeValidationError
from sentinel_analysis.metadata import apply_variable_contract
from sentinel_analysis.sensors.sentinel3.georeference import _corner_grid, _require_single_geolocation


def reference_grid_area_l2_lst(dataset: xr.Dataset, grid: AnalysisGrid) -> xr.Dataset:
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
