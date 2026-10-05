"""Wraps sentinel_analysis execution and persists the result to disk.

AnalysisWorkflow.execute() returns an AnalysisResult holding only in-memory
xarray Datasets - it never writes cubes to disk itself. This module is the
part of the platform responsible for turning that in-memory result into
files a client can download.

It also fills a gap between what the platform's UI naturally has (an
arbitrary drawn AOI + a target resolution in metres) and what
AnalysisRequest needs (a projected AnalysisGrid): AnalysisGrid.for_aoi
snaps a grid in the AOI's UTM zone.
"""

from pathlib import Path
from typing import Callable

from sentinel_analysis.config import AOI, ClientConfig
from sentinel_analysis.cube import AnalysisGrid
from sentinel_analysis.providers import AuxiliarySpec
from sentinel_analysis.workflow.request import AnalysisRequest
from sentinel_analysis.workflow.runner import AnalysisWorkflow

from .config import settings

ProgressCallback = Callable[[str, int, int], None]


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

    predictor_grid = AnalysisGrid.for_aoi(aoi, resolution_m=resolution_m)
    thermal_grid = AnalysisGrid.for_aoi(aoi, resolution_m=thermal_resolution_m, crs=predictor_grid.crs)

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
        sentinel2_source=payload.get("sentinel2_source", "cdse_safe"),
        raw_retention=payload.get("raw_retention", "aoi_subset"),
        terrain_predictors=bool(payload.get("terrain_predictors", False)),
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

    return result.save(output_root / job_id / "result")
