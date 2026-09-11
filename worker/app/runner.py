"""Wraps sentinel_analysis execution and persists the result to disk.

AnalysisWorkflow.execute() returns an AnalysisResult holding only in-memory
xarray Datasets - it never writes cubes to disk itself. This module is the
part of the platform responsible for turning that in-memory result into
files a client can download.

It also fills a gap between what the platform's UI naturally has (an
arbitrary drawn AOI + a target resolution in metres) and what
AnalysisRequest needs (a projected AnalysisGrid). sentinel_analysis itself
only derives a grid automatically for one of its five preset CitySpecs
(AnalysisRequest.for_city); for any other AOI, something has to pick a
projected CRS and snap a grid to it. That's done here, the same way
AnalysisGrid.for_city does it internally (reproject the AOI corners to a
metric CRS, then AnalysisGrid.from_bounds) - just with a computed UTM zone
instead of a preset one.
"""

from pathlib import Path
from typing import Callable

from sentinel_analysis.config import AOI, ClientConfig
from sentinel_analysis.cube import AnalysisGrid
from sentinel_analysis.providers import AuxiliarySpec
from sentinel_analysis.storage import write_zarr
from sentinel_analysis.workflow.request import AnalysisRequest
from sentinel_analysis.workflow.runner import AnalysisWorkflow

from .config import settings

ProgressCallback = Callable[[str, int, int], None]


def _utm_epsg(lon: float, lat: float) -> str:
    """Standard UTM zone for a WGS84 lon/lat (6-degree zones, N/S hemisphere)."""

    zone = int((lon + 180) // 6) % 60 + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def _projected_bounds(aoi: AOI, crs: str) -> tuple[float, float, float, float]:
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    corners = [
        transformer.transform(x, y)
        for x, y in (
            (aoi.west, aoi.south),
            (aoi.west, aoi.north),
            (aoi.east, aoi.south),
            (aoi.east, aoi.north),
        )
    ]
    return (
        min(x for x, _ in corners),
        min(y for _, y in corners),
        max(x for x, _ in corners),
        max(y for _, y in corners),
    )


def build_request(request_dict: dict) -> AnalysisRequest:
    """Build an AnalysisRequest from a job payload.

    Two shapes are accepted:
    - A full AnalysisRequest.to_dict() shape (pre-built grid/thermal_grid/
      predictor_grid) - used as-is via AnalysisRequest.from_dict, for
      advanced/direct callers.
    - The simpler {aoi, resolution_m} shape this worker's own frontend
      (`worker/frontend/`) sends for an arbitrary drawn AOI - a projected
      AnalysisGrid is derived here (see module docstring) before
      constructing the request.
    """

    if any(key in request_dict for key in ("grid", "thermal_grid", "predictor_grid")):
        return AnalysisRequest.from_dict(request_dict)

    payload = dict(request_dict)
    aoi = AOI(**payload["aoi"])
    resolution_m = float(payload.get("resolution_m", 100))
    thermal_resolution_m = float(payload.get("thermal_resolution_m", max(1000.0, resolution_m)))

    crs = _utm_epsg((aoi.west + aoi.east) / 2, (aoi.south + aoi.north) / 2)
    bounds = _projected_bounds(aoi, crs)
    predictor_grid = AnalysisGrid.from_bounds(bounds, crs=crs, resolution_m=resolution_m)
    thermal_grid = AnalysisGrid.from_bounds(bounds, crs=crs, resolution_m=thermal_resolution_m)

    return AnalysisRequest(
        aoi=aoi,
        start=payload["start"],
        end=payload["end"],
        sensors=tuple(payload.get("sensors", ("sentinel3",))),
        variables=tuple(payload.get("variables", ())),
        target_sensor=payload.get("target_sensor"),
        grid=predictor_grid,
        predictor_grid=predictor_grid,
        thermal_grid=thermal_grid,
        s5p_resolution_deg=payload.get("s5p_resolution_deg", 0.01),
        s2_cloud_cover_max=payload.get("s2_cloud_cover_max"),
        max_products_per_sensor=payload.get("max_products_per_sensor", 100),
        s2_composite_method=payload.get("s2_composite_method"),
        s2_min_observations=payload.get("s2_min_observations", 1),
        sentinel1_backend=payload.get("sentinel1_backend", "snap"),
        auxiliary=tuple(AuxiliarySpec.from_dict(item) for item in payload.get("auxiliary", ())),
    )


def execute_and_persist(
    job_id: str,
    request_dict: dict,
    *,
    output_root: Path,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Run one AnalysisRequest end to end, writing its cubes under
    output_root/{job_id}/result/. Returns that result directory."""

    request = build_request(request_dict)
    workflow = AnalysisWorkflow(request)
    work_dir = output_root / job_id / "work"
    result = workflow.execute(
        work_dir,
        config=ClientConfig.from_env(),
        max_workers=settings.max_download_workers,
        progress=progress_cb,
    )

    result_dir = output_root / job_id / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    write_zarr(result.cube, result_dir / "cube.zarr")
    if result.thermal_cube is not None:
        write_zarr(result.thermal_cube, result_dir / "thermal_cube.zarr")
    if result.predictor_cube is not None:
        write_zarr(result.predictor_cube, result_dir / "predictor_cube.zarr")
    if result.auxiliary:
        for name, dataset in result.auxiliary.items():
            write_zarr(dataset, result_dir / "auxiliary" / f"{name}.zarr")
    return result_dir
