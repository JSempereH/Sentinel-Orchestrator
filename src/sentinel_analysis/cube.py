"""Contracts for analysis-ready, multi-sensor city cubes."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Iterable

import numpy as np
import xarray as xr
from affine import Affine

from .cities import CitySpec


class CubeValidationError(ValueError):
    """Raised when a Dataset cannot be used as an analysis cube."""


@dataclass(frozen=True)
class AnalysisGrid:
    """A snapped, north-up regular analysis grid.

    Bounds are expressed in the grid CRS as ``(left, bottom, right, top)``.
    Coordinates are pixel centres and ``transform`` maps pixel corners to the
    CRS. Keeping these values together prevents two datasets from sharing an
    identifier while using different pixel origins.
    """

    grid_id: str
    crs: str
    bounds: tuple[float, float, float, float]
    resolution: tuple[float, float]
    width: int
    height: int
    transform: Affine
    city_id: str | None = None

    @property
    def resolution_m(self) -> float:
        """Return the x resolution for compatibility with city metadata."""

        return self.resolution[0]

    @property
    def x(self) -> np.ndarray:
        left, _, _, _ = self.bounds
        return left + (np.arange(self.width) + 0.5) * self.resolution[0]

    @property
    def y(self) -> np.ndarray:
        _, _, _, top = self.bounds
        return top - (np.arange(self.height) + 0.5) * self.resolution[1]

    @classmethod
    def from_bounds(
        cls,
        bounds: tuple[float, float, float, float],
        *,
        crs: str,
        resolution_m: int | float,
        grid_id: str | None = None,
        city_id: str | None = None,
    ) -> "AnalysisGrid":
        """Create a grid after snapping its extent to pixel boundaries."""

        left, bottom, right, top = bounds
        if not left < right or not bottom < top:
            raise ValueError("Grid bounds must be (left, bottom, right, top) with positive area")
        resolution = float(resolution_m)
        if resolution <= 0:
            raise ValueError("Grid resolution must be positive")
        snapped_left = floor(left / resolution) * resolution
        snapped_bottom = floor(bottom / resolution) * resolution
        snapped_right = ceil(right / resolution) * resolution
        snapped_top = ceil(top / resolution) * resolution
        width = max(1, int(round((snapped_right - snapped_left) / resolution)))
        height = max(1, int(round((snapped_top - snapped_bottom) / resolution)))
        snapped_bounds = (snapped_left, snapped_bottom, snapped_right, snapped_top)
        transform = Affine(resolution, 0, snapped_left, 0, -resolution, snapped_top)
        identifier = grid_id or f"{crs}:{resolution:g}m:{snapped_left:g},{snapped_bottom:g},{snapped_right:g},{snapped_top:g}"
        return cls(identifier, crs, snapped_bounds, (resolution, resolution), width, height, transform, city_id)

    @classmethod
    def for_city(cls, city: CitySpec, *, resolution_m: int | None = None) -> "AnalysisGrid":
        resolution = resolution_m or city.target_resolution_m
        try:
            from pyproj import Transformer
        except ImportError as exc:
            raise RuntimeError("Install pyproj to build a projected analysis grid") from exc
        transformer = Transformer.from_crs("EPSG:4326", city.crs, always_xy=True)
        corners = [
            transformer.transform(x, y)
            for x, y in (
                (city.aoi.west, city.aoi.south),
                (city.aoi.west, city.aoi.north),
                (city.aoi.east, city.aoi.south),
                (city.aoi.east, city.aoi.north),
            )
        ]
        projected_bounds = (
            min(x for x, _ in corners),
            min(y for _, y in corners),
            max(x for x, _ in corners),
            max(y for _, y in corners),
        )
        return cls.from_bounds(
            projected_bounds,
            crs=city.crs,
            resolution_m=resolution,
            grid_id=f"{city.city_id}:{city.crs}:{resolution}m",
            city_id=city.city_id,
        )


# Existing public name remains a readable alias while new code can use the
# domain-specific AnalysisGrid name.
GridSpec = AnalysisGrid


def validate_cube(
    dataset: xr.Dataset,
    *,
    required_variables: Iterable[str] = (),
    require_time: bool = True,
) -> xr.Dataset:
    """Validate and return a Dataset using the common cube contract."""

    required_dims = {"y", "x"}
    if require_time:
        required_dims.add("time")
    missing_dims = sorted(required_dims.difference(dataset.dims).difference(dataset.coords))
    if missing_dims:
        raise CubeValidationError(f"Cube is missing dimensions or coordinates: {missing_dims}")
    missing_variables = sorted(set(required_variables).difference(dataset.data_vars))
    if missing_variables:
        raise CubeValidationError(f"Cube is missing variables: {missing_variables}")
    if "crs" not in dataset.attrs and "spatial_ref" not in dataset:
        raise CubeValidationError("Cube must declare a CRS in attrs['crs'] or a spatial_ref variable")
    return dataset


def validate_observation_set(
    dataset: xr.Dataset,
    *,
    required_variables: Iterable[str] = (),
) -> xr.Dataset:
    """Validate a geolocated swath/table before rasterizing it."""

    if "observation" not in dataset.dims:
        raise CubeValidationError("Observation set must contain an 'observation' dimension")
    missing_coordinates = {"latitude", "longitude"}.difference(dataset.coords).difference(dataset.data_vars)
    if missing_coordinates:
        raise CubeValidationError(f"Observation set is missing coordinates: {sorted(missing_coordinates)}")
    missing_variables = sorted(set(required_variables).difference(dataset.data_vars))
    if missing_variables:
        raise CubeValidationError(f"Observation set is missing variables: {missing_variables}")
    return dataset


def tag_cube(
    dataset: xr.Dataset,
    *,
    source: str,
    grid: GridSpec | None = None,
    processing_version: str = "unknown",
) -> xr.Dataset:
    """Attach stable provenance metadata without changing data variables."""

    result = dataset.copy()
    result.attrs.update({
        "source": source,
        "processing_version": processing_version,
        "analysis_ready": True,
    })
    if grid:
        result.attrs.update({
            "grid_id": grid.grid_id,
            "crs": grid.crs,
            "resolution_m": grid.resolution_m,
            "city_id": grid.city_id or "",
        })
    return result
