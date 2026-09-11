"""Declarative request for a multisensor analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from ..cities import CitySpec
from ..config import AOI
from ..cube import AnalysisGrid
from ..providers import AUXILIARY_PROVIDERS, AuxiliarySpec


SUPPORTED_SENSORS = ("sentinel1", "sentinel2", "sentinel3", "sentinel5p", "landsat", "ecostress")


@dataclass(frozen=True)
class AnalysisRequest:
    """Inputs and policies shared by all sensor workflows."""

    aoi: AOI
    start: str | date | datetime
    end: str | date | datetime
    sensors: tuple[str, ...] = ("sentinel3",)
    variables: tuple[str, ...] = ()
    target_sensor: str | None = None
    grid: AnalysisGrid | None = None
    thermal_grid: AnalysisGrid | None = None
    predictor_grid: AnalysisGrid | None = None
    temporal_tolerance: np.timedelta64 = np.timedelta64(3, "D")
    temporal_tolerances: Mapping[str, np.timedelta64] | None = None
    s5p_resolution_deg: float = 0.01
    s2_cloud_cover_max: float | None = None
    max_products_per_sensor: int = 100
    s2_composite_method: str | None = None
    s2_min_observations: int = 1
    sentinel1_backend: str = "snap"
    auxiliary: tuple[AuxiliarySpec, ...] = ()

    def __post_init__(self) -> None:
        sensors = tuple(sensor.lower() for sensor in self.sensors)
        if not sensors:
            raise ValueError("At least one sensor is required")
        unknown = sorted(set(sensors).difference(SUPPORTED_SENSORS))
        if unknown:
            raise ValueError(f"Unsupported sensors: {unknown}")
        if self.target_sensor and self.target_sensor.lower() not in sensors:
            raise ValueError("target_sensor must be one of sensors")
        if str(self.start) > str(self.end):
            raise ValueError("Analysis start must not be after end")
        if self.s5p_resolution_deg <= 0:
            raise ValueError("s5p_resolution_deg must be positive")
        if self.s2_cloud_cover_max is not None and not 0 <= self.s2_cloud_cover_max <= 100:
            raise ValueError("s2_cloud_cover_max must be between 0 and 100")
        if self.max_products_per_sensor <= 0:
            raise ValueError("max_products_per_sensor must be positive")
        if self.s2_composite_method not in {None, "mean", "median"}:
            raise ValueError("s2_composite_method must be None, 'mean' or 'median'")
        if self.s2_min_observations < 1:
            raise ValueError("s2_min_observations must be positive")
        if self.sentinel1_backend not in {"s1ard", "snap", "hyp3_rtc"}:
            raise ValueError("sentinel1_backend must be 's1ard', 'snap' or 'hyp3_rtc'")
        auxiliary = tuple(value if isinstance(value, AuxiliarySpec) else AuxiliarySpec.from_dict(value) for value in self.auxiliary)
        providers = [value.provider for value in auxiliary]
        if len(providers) != len(set(providers)):
            raise ValueError("Only one auxiliary spec per provider is supported")
        if any(value.provider not in AUXILIARY_PROVIDERS for value in auxiliary):
            raise ValueError("Unsupported auxiliary provider")
        object.__setattr__(self, "sensors", sensors)
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "temporal_tolerances", dict(self.temporal_tolerances or {}))
        object.__setattr__(self, "auxiliary", auxiliary)
        if self.target_sensor:
            object.__setattr__(self, "target_sensor", self.target_sensor.lower())

    @classmethod
    def for_city(
        cls,
        city: CitySpec,
        start: str | date | datetime,
        end: str | date | datetime,
        *,
        sensors: Iterable[str] = ("sentinel1", "sentinel2", "sentinel3"),
        variables: Iterable[str] = (),
        resolution_m: int | None = None,
        max_products_per_sensor: int = 100,
        auxiliary: Iterable[AuxiliarySpec] = (),
    ) -> "AnalysisRequest":
        resolution = resolution_m or city.target_resolution_m
        predictor_grid = AnalysisGrid.for_city(city, resolution_m=resolution)
        thermal_grid = AnalysisGrid.for_city(city, resolution_m=max(1000, resolution))
        return cls(
            aoi=city.aoi,
            start=start,
            end=end,
            sensors=tuple(sensors),
            variables=tuple(variables),
            grid=predictor_grid,
            predictor_grid=predictor_grid,
            thermal_grid=thermal_grid,
            max_products_per_sensor=max_products_per_sensor,
            auxiliary=tuple(auxiliary),
        )

    def tolerance_for(self, sensor: str) -> np.timedelta64:
        """Return the configured temporal tolerance for one sensor."""

        return (self.temporal_tolerances or {}).get(sensor.lower(), self.temporal_tolerance)

    @staticmethod
    def _grid_dict(grid: AnalysisGrid | None) -> dict | None:
        if grid is None:
            return None
        return {
            "grid_id": grid.grid_id,
            "crs": grid.crs,
            "bounds": list(grid.bounds),
            "resolution_m": grid.resolution_m,
            "city_id": grid.city_id,
        }

    def to_dict(self) -> dict:
        """Serialize the request without embedding data or credentials."""

        return {
            "aoi": {"west": self.aoi.west, "south": self.aoi.south, "east": self.aoi.east, "north": self.aoi.north},
            "start": str(self.start),
            "end": str(self.end),
            "sensors": list(self.sensors),
            "variables": list(self.variables),
            "target_sensor": self.target_sensor,
            "grid": self._grid_dict(self.grid),
            "thermal_grid": self._grid_dict(self.thermal_grid),
            "predictor_grid": self._grid_dict(self.predictor_grid),
            "temporal_tolerance_days": float(self.temporal_tolerance / np.timedelta64(1, "D")),
            "temporal_tolerances_days": {key: float(value / np.timedelta64(1, "D")) for key, value in (self.temporal_tolerances or {}).items()},
            "s5p_resolution_deg": self.s5p_resolution_deg,
            "s2_cloud_cover_max": self.s2_cloud_cover_max,
            "max_products_per_sensor": self.max_products_per_sensor,
            "s2_composite_method": self.s2_composite_method,
            "s2_min_observations": self.s2_min_observations,
            "sentinel1_backend": self.sentinel1_backend,
            "auxiliary": [value.to_dict() for value in self.auxiliary],
        }

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, value: dict) -> "AnalysisRequest":
        def grid(data: dict | None) -> AnalysisGrid | None:
            if data is None:
                return None
            return AnalysisGrid.from_bounds(tuple(data["bounds"]), crs=data["crs"], resolution_m=data["resolution_m"], grid_id=data.get("grid_id"), city_id=data.get("city_id"))
        return cls(
            aoi=AOI(**value["aoi"]),
            start=value["start"],
            end=value["end"],
            sensors=tuple(value.get("sensors", ("sentinel3",))),
            variables=tuple(value.get("variables", ())),
            target_sensor=value.get("target_sensor"),
            grid=grid(value.get("grid")),
            thermal_grid=grid(value.get("thermal_grid")),
            predictor_grid=grid(value.get("predictor_grid")),
            temporal_tolerance=np.timedelta64(int(value.get("temporal_tolerance_days", 3) * 86400), "s"),
            temporal_tolerances={key: np.timedelta64(int(days * 86400), "s") for key, days in value.get("temporal_tolerances_days", {}).items()},
            s5p_resolution_deg=value.get("s5p_resolution_deg", 0.01),
            s2_cloud_cover_max=value.get("s2_cloud_cover_max"),
            max_products_per_sensor=value.get("max_products_per_sensor", 100),
            s2_composite_method=value.get("s2_composite_method"),
            s2_min_observations=value.get("s2_min_observations", 1),
            sentinel1_backend=value.get("sentinel1_backend", "snap"),
            auxiliary=tuple(AuxiliarySpec.from_dict(item) for item in value.get("auxiliary", ())),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "AnalysisRequest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
