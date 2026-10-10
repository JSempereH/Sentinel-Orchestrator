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


def _restore_boolean(source_dtype: np.dtype, regridded: xr.DataArray) -> xr.DataArray:
    """Keep boolean masks boolean after regridding.

    Regridding marks target cells outside the source footprint as NaN, which
    turns a boolean mask into float - and ``NaN.astype(bool)`` is True, so a
    later cast would silently mark every no-data cell as valid. Cells with
    no source data are not valid: fill them with False.
    """

    if source_dtype.kind != "b":
        return regridded
    return regridded.fillna(0).astype(bool)


def _spacing(cube: xr.Dataset) -> tuple[float, float]:
    """Cell size (dx, dy) of a regular grid.

    A small AOI can fall inside a single row or column of a coarse source
    (one 0.25 degree ERA5 row over a city): that axis then borrows the other
    axis' spacing, or the declared ``resolution_deg``.
    """

    def step(values: np.ndarray) -> float | None:
        return float(abs(values[1] - values[0])) if values.size > 1 else None

    dx, dy = step(cube.x.values), step(cube.y.values)
    declared = cube.attrs.get("resolution_deg")
    dx = dx if dx is not None else (dy if dy is not None else declared)
    dy = dy if dy is not None else (dx if dx is not None else declared)
    if dx is None or dy is None:
        raise CubeValidationError("A one-cell grid needs a 'resolution_deg' attribute to be regridded")
    return float(dx), float(dy)


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
    target_resolution = _spacing(target)[0]
    source_resolution = _spacing(source)[0]
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
                # Bound the nearest-neighbour search to half a source cell per
                # axis so target cells outside the source footprint become
                # missing instead of repeating the source's edge values.
                source_dy = float(abs(source.y.values[1] - source.y.values[0])) if source.y.size > 1 else source_resolution
                source_dtype = value.dtype
                value = value.sortby("x").sortby("y")
                value = value.reindex(x=target.x, method="nearest", tolerance=source_resolution / 2 * (1 + 1e-9))
                result[name] = _restore_boolean(source_dtype, value.reindex(y=target.y, method="nearest", tolerance=source_dy / 2 * (1 + 1e-9)))
        return result

    try:
        from rasterio.transform import from_origin
        from rasterio.warp import reproject
        from rasterio.enums import Resampling
    except ImportError as exc:
        raise RuntimeError("Install citycube[optical] or citycube[sar] for CRS reprojection") from exc
    if target_crs is None or source_crs is None:
        raise CubeValidationError("Both cubes must declare CRS before reprojection")
    if target.x.size < 2 or target.y.size < 2:
        raise CubeValidationError("Reprojection requires a target grid with at least two coordinates per axis")
    # The transforms below place row 0 at the *top* (max y) and column 0 at
    # the left, so the arrays must be north-up and west-to-east. Sources with
    # ascending y - every grid_s5p output, latitude-ascending ERA5/CAMS files
    # - were otherwise warped upside down.
    source = source.sortby("x").sortby("y", ascending=False)

    def transform(data: xr.DataArray, *, method: str) -> xr.DataArray:
        dx, dy = _spacing(source)
        target_dx, target_dy = _spacing(target)
        source_transform = from_origin(float(source.x.values.min() - dx / 2), float(source.y.values.max() + dy / 2), dx, dy)
        target_transform = from_origin(float(target.x.values.min() - target_dx / 2), float(target.y.values.max() + target_dy / 2), target_dx, target_dy)
        resampling = {"nearest": Resampling.nearest, "mean": Resampling.average}.get(method, Resampling.bilinear)
        shape = (target.y.size, target.x.size)

        def warp(values: np.ndarray) -> np.ndarray:
            layers = values.reshape((-1, *values.shape[-2:]))
            output = np.full((layers.shape[0], *shape), np.nan, dtype=np.float32)
            for index, layer in enumerate(layers):
                reproject(
                    source=layer.astype(np.float32),
                    destination=output[index],
                    src_transform=source_transform,
                    src_crs=source_crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    src_nodata=np.nan,
                    dst_nodata=np.nan,
                    resampling=resampling,
                )
            return output.reshape((*values.shape[:-2], *shape))

        # apply_ufunc keeps dask-backed inputs lazy, warping one time chunk
        # per task instead of materializing the whole source cube.
        warped = xr.apply_ufunc(
            warp,
            data.transpose("time", "y", "x"),
            input_core_dims=[["y", "x"]],
            output_core_dims=[["y_target", "x_target"]],
            dask="parallelized",
            output_dtypes=[np.float32],
            dask_gufunc_kwargs={"output_sizes": {"y_target": shape[0], "x_target": shape[1]}},
            keep_attrs=True,
        )
        return warped.rename({"y_target": "y", "x_target": "x"}).assign_coords(y=target.y.values, x=target.x.values).rename(data.name)

    result = xr.Dataset(attrs={**source.attrs, "crs": target_crs, "grid_id": target.attrs.get("grid_id", "")})
    for name, value in source.data_vars.items():
        if set(("time", "y", "x")).issubset(value.dims):
            method = policy.for_variable(name, source_resolution=source_resolution, target_resolution=target_resolution, categorical=value.dtype.kind == "b")
            result[name] = _restore_boolean(value.dtype, transform(value, method=method))
    return result
