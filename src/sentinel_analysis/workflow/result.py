"""Result and provenance of a completed analysis workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import xarray as xr

from .plan import WorkflowPlan


@dataclass(frozen=True)
class AnalysisResult:
    cube: xr.Dataset
    plan: WorkflowPlan
    provenance: dict[str, Any]
    thermal_cube: xr.Dataset | None = None
    predictor_cube: xr.Dataset | None = None
    auxiliary: dict[str, xr.Dataset] | None = None
