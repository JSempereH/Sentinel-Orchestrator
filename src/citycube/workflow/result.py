"""Result and provenance of a completed analysis workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import xarray as xr

from .plan import WorkflowPlan


@dataclass(frozen=True)
class AnalysisResult:
    cube: xr.Dataset
    plan: WorkflowPlan
    provenance: dict[str, Any]
    thermal_cube: xr.Dataset | None = None
    # Fine-resolution cubes per sensor, each on its own acquisition times.
    predictors: dict[str, xr.Dataset] = field(default_factory=dict)
    auxiliary: dict[str, xr.Dataset] | None = None
    terrain: xr.Dataset | None = None
    downscaled: xr.Dataset | None = None

    def save(self, directory: str | Path) -> Path:
        """Write every cube as Zarr plus the request and provenance as JSON.

        Layout: ``cube.zarr``, ``thermal_cube.zarr``, ``predictors/<sensor>.zarr``,
        ``terrain.zarr``, ``downscaled.zarr``, ``auxiliary/<name>.zarr``, ``request.json`` and
        ``provenance.json`` (each cube only when present).
        """

        from ..storage import write_zarr

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        write_zarr(self.cube, directory / "cube.zarr")
        if self.thermal_cube is not None:
            write_zarr(self.thermal_cube, directory / "thermal_cube.zarr")
        for sensor, dataset in self.predictors.items():
            write_zarr(dataset, directory / "predictors" / f"{sensor}.zarr")
        if self.terrain is not None:
            write_zarr(self.terrain, directory / "terrain.zarr")
        if self.downscaled is not None:
            write_zarr(self.downscaled, directory / "downscaled.zarr")
        for name, dataset in (self.auxiliary or {}).items():
            write_zarr(dataset, directory / "auxiliary" / f"{name}.zarr")
        self.plan.request.save_json(directory / "request.json")
        (directory / "provenance.json").write_text(json.dumps(self.provenance, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return directory
