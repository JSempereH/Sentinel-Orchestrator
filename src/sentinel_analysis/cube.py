"""Contracts for analysis-ready, multi-sensor city cubes."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Iterable

import numpy as np
import xarray as xr
from affine import Affine

from .cities import CitySpec
from .config import AOI


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
        return cls.from_bounds(
            projected_bounds(city.aoi, city.crs),
            crs=city.crs,
            resolution_m=resolution,
            grid_id=f"{city.city_id}:{city.crs}:{resolution}m",
            city_id=city.city_id,
        )

    @classmethod
    def for_aoi(
        cls,
        aoi: AOI,
        *,
        resolution_m: int | float,
        crs: str | None = None,
        grid_id: str | None = None,
    ) -> "AnalysisGrid":
        """Snap a projected grid around any lon/lat AOI.

        ``crs`` defaults to the UTM zone of the AOI centre, so arbitrary areas
        get a metric grid without a preset ``CitySpec``.
        """

        crs = crs or utm_crs((aoi.west + aoi.east) / 2, (aoi.south + aoi.north) / 2)
        return cls.from_bounds(projected_bounds(aoi, crs), crs=crs, resolution_m=resolution_m, grid_id=grid_id)

    def to_geobox(self):
        """Return the equivalent ``odc.geo.geobox.GeoBox`` (for odc-stac/odc-geo).

        Requires the ``odc`` extra.
        """

        try:
            from odc.geo.geobox import GeoBox
        except ImportError as exc:
            raise RuntimeError("Install sentinel-analysis[odc] to convert grids to odc-geo GeoBoxes") from exc
        return GeoBox((self.height, self.width), self.transform, self.crs)

    @classmethod
    def from_geobox(cls, geobox, *, grid_id: str | None = None, city_id: str | None = None) -> "AnalysisGrid":
        """Build a grid from a north-up ``odc.geo`` GeoBox with square pixels."""

        transform = Affine(*tuple(geobox.affine)[:6])
        if transform.b != 0 or transform.d != 0 or transform.e >= 0 or abs(transform.a) != abs(transform.e):
            raise ValueError("Only north-up GeoBoxes with square pixels can become an AnalysisGrid")
        height, width = geobox.shape
        resolution = float(transform.a)
        left, top = float(transform.c), float(transform.f)
        bounds = (left, top - height * resolution, left + width * resolution, top)
        crs = str(geobox.crs)
        identifier = grid_id or f"{crs}:{resolution:g}m:{bounds[0]:g},{bounds[1]:g},{bounds[2]:g},{bounds[3]:g}"
        return cls(identifier, crs, bounds, (resolution, resolution), int(width), int(height), transform, city_id)

    def chips(self, chip_size_m: float, *, drop_partial: bool = True) -> list["AnalysisGrid"]:
        """Partition this grid into fixed-size, non-overlapping sub-grids.

        Unlike `from_bounds`, which snaps to an arbitrary AOI-shaped extent,
        this produces uniform patches (e.g. the 2.5x2.5 km chips fixed-size
        training/inference patches are built from). `chip_size_m` must be a
        whole multiple of the grid resolution. By default a chip that would
        be cut short by the parent grid's edge is dropped rather than
        returned undersized; pass ``drop_partial=False`` to keep it.
        """

        if chip_size_m <= 0:
            raise ValueError("chip_size_m must be positive")
        resolution = self.resolution[0]
        chip_pixels = chip_size_m / resolution
        if abs(chip_pixels - round(chip_pixels)) > 1e-6:
            raise ValueError(f"chip_size_m ({chip_size_m}) must be a whole multiple of the grid resolution ({resolution})")
        chip_pixels = int(round(chip_pixels))
        if chip_pixels < 1:
            raise ValueError("chip_size_m must be at least one pixel wide")

        left, _, _, top = self.bounds
        n_cols = self.width // chip_pixels if drop_partial else ceil(self.width / chip_pixels)
        n_rows = self.height // chip_pixels if drop_partial else ceil(self.height / chip_pixels)
        result: list[AnalysisGrid] = []
        for row in range(n_rows):
            for col in range(n_cols):
                chip_width = min(chip_pixels, self.width - col * chip_pixels)
                chip_height = min(chip_pixels, self.height - row * chip_pixels)
                chip_left = left + col * chip_pixels * resolution
                chip_top = top - row * chip_pixels * resolution
                chip_bounds = (chip_left, chip_top - chip_height * resolution, chip_left + chip_width * resolution, chip_top)
                result.append(
                    AnalysisGrid(
                        grid_id=f"{self.grid_id}:chip:{row}:{col}",
                        crs=self.crs,
                        bounds=chip_bounds,
                        resolution=self.resolution,
                        width=chip_width,
                        height=chip_height,
                        transform=Affine(resolution, 0, chip_left, 0, -resolution, chip_top),
                        city_id=self.city_id,
                    )
                )
        return result


def utm_crs(lon: float, lat: float) -> str:
    """Standard UTM zone CRS for a WGS84 lon/lat (6-degree zones, N/S hemisphere)."""

    zone = int((lon + 180) // 6) % 60 + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def projected_bounds(aoi: AOI, crs: str) -> tuple[float, float, float, float]:
    """Bounds of a lon/lat AOI in ``crs``, densifying edges so curved
    projected edges (e.g. a UTM box's bowed north edge) are fully enclosed."""

    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise RuntimeError("Install pyproj to build a projected analysis grid") from exc
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return transformer.transform_bounds(aoi.west, aoi.south, aoi.east, aoi.north, densify_pts=21)


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
