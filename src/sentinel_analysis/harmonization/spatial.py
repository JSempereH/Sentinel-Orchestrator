"""Spatial harmonization for regular analysis grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import xarray as xr

from ..cube import CubeValidationError, validate_cube


@dataclass(frozen=True)
class SpatialResamplingPolicy:
    """Variable-specific spatial semantics for harmonization."""

    methods: Mapping[str, str] | None = None

    def for_variable(self, name: str, *, source_resolution: float, target_resolution: float, categorical: bool) -> str:
        explicit = (self.methods or {}).get(name)
        if explicit:
            return explicit
        if categorical or name.endswith("mask") or name.endswith("flags"):
            return "nearest"
        return "mean" if source_resolution < target_resolution else "nearest"


def harmonize_spatial(
    target: xr.Dataset,
    source: xr.Dataset,
    *,
    policy: SpatialResamplingPolicy | None = None,
) -> xr.Dataset:
    """Place a regular source cube on the target grid.

    Same-CRS grids use coordinate interpolation. Reprojection between CRSs is
    delegated to rasterio, preserving nearest-neighbour semantics for masks and
    categorical fields.
    """

    validate_cube(target, require_time=True)
    validate_cube(source, require_time=True)
    policy = policy or SpatialResamplingPolicy()
    target_crs = target.attrs.get("crs")
    source_crs = source.attrs.get("crs")
    target_resolution = float(abs(target.x.values[1] - target.x.values[0]))
    source_resolution = float(abs(source.x.values[1] - source.x.values[0]))
    if target_crs == source_crs:
        result = xr.Dataset(attrs={**source.attrs, "crs": target_crs, "grid_id": target.attrs.get("grid_id", "")})
        template = next(iter(target.data_vars.values()))
        for name, value in source.data_vars.items():
            if not {"time", "y", "x"}.issubset(value.dims):
                continue
            method = policy.for_variable(name, source_resolution=source_resolution, target_resolution=target_resolution, categorical=value.dtype.kind == "b")
            if method == "mean" and source_resolution < target_resolution:
                from ..downscale import reaggregate_to_target
                result[name] = reaggregate_to_target(value, template)
            else:
                result[name] = value.reindex(x=target.x, y=target.y, method="nearest")
        return result

    try:
        from rasterio.transform import from_origin
        from rasterio.warp import reproject
        from rasterio.enums import Resampling
    except ImportError as exc:
        raise RuntimeError("Install sentinel-analysis[optical] or sentinel-analysis[sar] for CRS reprojection") from exc
    if target_crs is None or source_crs is None:
        raise CubeValidationError("Both cubes must declare CRS before reprojection")
    if target.x.size < 2 or target.y.size < 2 or source.x.size < 2 or source.y.size < 2:
        raise CubeValidationError("Reprojection requires regular grids with at least two coordinates per axis")

    def transform(data: xr.DataArray, *, method: str) -> xr.DataArray:
        dx = float(abs(source.x.values[1] - source.x.values[0]))
        dy = float(abs(source.y.values[1] - source.y.values[0]))
        target_dx = float(abs(target.x.values[1] - target.x.values[0]))
        target_dy = float(abs(target.y.values[1] - target.y.values[0]))
        source_transform = from_origin(float(source.x.values.min() - dx / 2), float(source.y.values.max() + dy / 2), dx, dy)
        target_transform = from_origin(float(target.x.values.min() - target_dx / 2), float(target.y.values.max() + target_dy / 2), target_dx, target_dy)
        values = data.transpose("time", "y", "x").values
        output = np.full((values.shape[0], target.y.size, target.x.size), np.nan, dtype=np.float32)
        for index, layer in enumerate(values):
            reproject(
                source=layer.astype(np.float32),
                destination=output[index],
                src_transform=source_transform,
                src_crs=source_crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                src_nodata=np.nan,
                dst_nodata=np.nan,
                resampling={"nearest": Resampling.nearest, "mean": Resampling.average}.get(method, Resampling.bilinear),
            )
        return xr.DataArray(output, dims=("time", "y", "x"), coords={"time": data.time, "y": target.y, "x": target.x}, name=data.name, attrs=data.attrs)

    result = xr.Dataset(attrs={**source.attrs, "crs": target_crs, "grid_id": target.attrs.get("grid_id", "")})
    for name, value in source.data_vars.items():
        if set(("time", "y", "x")).issubset(value.dims):
            method = policy.for_variable(name, source_resolution=source_resolution, target_resolution=target_resolution, categorical=value.dtype.kind == "b")
            result[name] = transform(value, method=method)
    return result
