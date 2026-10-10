"""Size estimates and limits that reject a request before it exhausts memory.

Every prepared cube is held in memory (dask is only used for regridding), so
the size of a run is roughly AOI cells x products x variables per sensor. A
request far beyond what the machine can hold used to fail only after
downloading, or freeze the machine; checking the estimate first turns that
into an immediate, explained error.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .request import AnalysisRequest, _as_instant

# Prepared cubes are float64 after harmonization.
BYTES_PER_VALUE = 8
# Approximate data variables each sensor's prepared cube holds (bands,
# indices, masks and quality flags), measured on real runs.
VARIABLES_PER_SENSOR = {
    "sentinel1": 6,
    "sentinel2": 14,
    "sentinel3": 6,
    "sentinel5p": 3,
    "landsat": 3,
    "ecostress": 3,
}
# elevation, slope, aspect, cos_incidence.
TERRAIN_VARIABLES = 4
# lst_downscaled(_raw), uncertainty, support, correction.
DOWNSCALED_VARIABLES = 5
# Time steps per day of each gridded auxiliary source, as delivered.
AUXILIARY_STEPS_PER_DAY = {"era5": 24, "cams": 8}
# Interpolated value, station count, distance and validity flag.
OPENAQ_GRID_VARIABLES = 4
_EARTH_RADIUS_KM = 6371.0088


class RequestTooLargeError(ValueError):
    """The request exceeds a configured size limit."""


@dataclass(frozen=True)
class RequestEstimate:
    """Order-of-magnitude size of a request's in-memory cubes.

    ``estimated_bytes`` assumes every sensor finds ``max_products_per_sensor``
    products and counts the prepared cubes only; peak memory during
    processing is typically up to about twice that.
    """

    aoi_km2: float
    cells: dict[str, int]
    estimated_bytes: int

    @property
    def estimated_gb(self) -> float:
        return self.estimated_bytes / 1e9

    def to_dict(self) -> dict:
        return {"aoi_km2": round(self.aoi_km2, 1), "cells": dict(self.cells), "estimated_gb": round(self.estimated_gb, 3)}


@dataclass(frozen=True)
class RequestLimits:
    """Upper bounds for a request; ``None`` disables a bound."""

    max_aoi_km2: float | None = None
    max_products_per_sensor: int | None = None
    max_estimated_gb: float | None = 16.0


# The library default only guards memory: AOI and product counts are a
# deployment decision (the worker sets its own from its environment).
DEFAULT_REQUEST_LIMITS = RequestLimits()


def aoi_area_km2(request: AnalysisRequest) -> float:
    """Area of the request's WGS84 bounding box on a spherical Earth.

    The bounding box, not a polygon AOI's own area, because every cube is a
    rectangular grid over the box: that is what memory and storage pay for.
    """

    aoi = request.aoi
    width = math.radians(aoi.east - aoi.west)
    height = abs(math.sin(math.radians(aoi.north)) - math.sin(math.radians(aoi.south)))
    return _EARTH_RADIUS_KM**2 * width * height


def estimate_request(request: AnalysisRequest) -> RequestEstimate:
    """Estimate the cells and in-memory size of ``request``'s cubes."""

    area = aoi_area_km2(request)
    products = request.max_products_per_sensor
    thermal = request.thermal_grid or request.grid
    predictor = request.predictor_grid or request.grid
    cells: dict[str, int] = {}
    total = 0
    for sensor in request.sensors:
        if sensor == "sentinel5p":
            aoi = request.aoi
            count = math.ceil((aoi.east - aoi.west) / request.s5p_resolution_deg) * math.ceil((aoi.north - aoi.south) / request.s5p_resolution_deg)
        else:
            grid = thermal if sensor == "sentinel3" else predictor
            # Without a grid the run fails before reading data; assume 1 km.
            count = grid.width * grid.height if grid is not None else math.ceil(area)
        cells[sensor] = count
        total += count * products * VARIABLES_PER_SENSOR[sensor]
    fine_cells = predictor.width * predictor.height if predictor is not None else 0
    if request.terrain_predictors:
        total += fine_cells * TERRAIN_VARIABLES
    if request.downscale is not None:
        total += fine_cells * products * DOWNSCALED_VARIABLES
    days = (_as_instant(request.end, end_of_day=True) - _as_instant(request.start)).days + 1
    thermal_cells = thermal.width * thermal.height if thermal is not None else math.ceil(area)
    for spec in request.auxiliary:
        variables = max(1, len(spec.variables))
        if spec.provider in AUXILIARY_STEPS_PER_DAY:
            total += days * AUXILIARY_STEPS_PER_DAY[spec.provider] * thermal_cells * variables
        elif spec.provider == "openaq":
            total += fine_cells * products * OPENAQ_GRID_VARIABLES * variables
    return RequestEstimate(aoi_km2=area, cells=cells, estimated_bytes=total * BYTES_PER_VALUE)


def check_request(request: AnalysisRequest, limits: RequestLimits = DEFAULT_REQUEST_LIMITS) -> RequestEstimate:
    """Return the request's estimate, or raise ``RequestTooLargeError`` listing every exceeded limit."""

    estimate = estimate_request(request)
    problems = []
    if limits.max_aoi_km2 is not None and estimate.aoi_km2 > limits.max_aoi_km2:
        problems.append(f"AOI is {estimate.aoi_km2:,.0f} km2 (limit {limits.max_aoi_km2:,.0f} km2)")
    if limits.max_products_per_sensor is not None and request.max_products_per_sensor > limits.max_products_per_sensor:
        problems.append(f"max_products_per_sensor is {request.max_products_per_sensor} (limit {limits.max_products_per_sensor})")
    if limits.max_estimated_gb is not None and estimate.estimated_gb > limits.max_estimated_gb:
        problems.append(
            f"estimated cube size is {estimate.estimated_gb:,.1f} GB (limit {limits.max_estimated_gb:,.1f} GB); "
            "reduce the AOI, the period or max_products_per_sensor, or coarsen the predictor grid"
        )
    if problems:
        raise RequestTooLargeError("Request too large: " + "; ".join(problems))
    return estimate
