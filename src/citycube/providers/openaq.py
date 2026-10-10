"""OpenAQ v3 station discovery and measurement downloads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import requests
import xarray as xr
from dotenv import load_dotenv

from ..cache import AssetCache
from ..config import AOI
from ..cube import AnalysisGrid, CubeValidationError
from ..http import http_session
from .base import AuxiliaryArtifact, AuxiliaryProviderError, AuxiliarySpec, artifact_from_path, as_date, request_key


OPENAQ_VARIABLES: Mapping[str, Mapping[str, str]] = {
    "NO2": {"api": "no2", "units": "ug/m3"},
    "O3": {"api": "o3", "units": "ug/m3"},
    "CO": {"api": "co", "units": "ug/m3"},
    "PM2.5": {"api": "pm25", "units": "ug/m3"},
    "PM10": {"api": "pm10", "units": "ug/m3"},
    "SO2": {"api": "so2", "units": "ug/m3"},
}


@dataclass(frozen=True)
class OpenAQConfig:
    base_url: str = "https://api.openaq.org/v3"
    api_key: str | None = None
    page_limit: int = 1000
    max_locations: int = 100

    @classmethod
    def from_env(cls) -> "OpenAQConfig":
        load_dotenv()
        return cls(
            base_url=os.getenv("OPENAQ_BASE_URL", "https://api.openaq.org/v3").rstrip("/"),
            api_key=os.getenv("OPENAQ_API_KEY"),
            page_limit=int(os.getenv("OPENAQ_PAGE_LIMIT", "1000")),
            max_locations=int(os.getenv("OPENAQ_MAX_LOCATIONS", "100")),
        )


@dataclass(frozen=True)
class OpenAQInterpolationConfig:
    """Guardrails for converting station observations into a grid."""

    method: str = "idw"
    power: float = 2.0
    max_distance_m: float = 10000.0
    min_neighbors: int = 3
    max_neighbors: int = 12
    chunk_size: int = 100000

    def __post_init__(self) -> None:
        if self.method != "idw":
            raise ValueError("OpenAQ interpolation method must be 'idw'")
        if self.power <= 0:
            raise ValueError("OpenAQ IDW power must be positive")
        if self.max_distance_m <= 0:
            raise ValueError("OpenAQ maximum interpolation distance must be positive")
        if self.min_neighbors < 1:
            raise ValueError("OpenAQ minimum neighbors must be positive")
        if self.max_neighbors < self.min_neighbors:
            raise ValueError("OpenAQ maximum neighbors must not be below the minimum")
        if self.chunk_size < 1:
            raise ValueError("OpenAQ interpolation chunk size must be positive")

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> "OpenAQInterpolationConfig":
        """Build interpolation settings from JSON-safe auxiliary options."""

        return cls(
            method=str(options.get("interpolation_method", "idw")).lower(),
            power=float(options.get("interpolation_power", 2.0)),
            max_distance_m=float(options.get("interpolation_max_distance_m", 10000.0)),
            min_neighbors=int(options.get("interpolation_min_neighbors", 3)),
            max_neighbors=int(options.get("interpolation_max_neighbors", 12)),
            chunk_size=int(options.get("interpolation_chunk_size", 100000)),
        )


class OpenAQProvider:
    """Fetch station metadata and observations, preserving station provenance."""

    name = "openaq"

    def __init__(self, config: OpenAQConfig | None = None, *, session: requests.Session | None = None):
        self.config = config or OpenAQConfig.from_env()
        self.session = session or http_session()
        # Proactive OpenAQ rate-limit throttling (60/min on the free tier -
        # see x-ratelimit-* response headers). Without this, a multisensor
        # request over many locations/sensors burns through the budget and
        # relies entirely on http_session()'s Retry-After-based 429 retries,
        # which works but wastes rejected requests. `_rate_limit_reset_at` is
        # a wall-clock deadline computed from the header's countdown-in-
        # seconds value (confirmed via OpenAQ's docs: "x-ratelimit-reset: 60"
        # means "resets in 60 seconds", not a Unix timestamp).
        self._rate_limit_remaining: int | None = None
        self._rate_limit_reset_at: float | None = None

    @staticmethod
    def variables_for(spec: AuxiliarySpec) -> tuple[str, ...]:
        return spec.variables or ("NO2", "O3", "CO", "PM2.5", "PM10")

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.config.api_key} if self.config.api_key else {}

    def _throttle_if_needed(self) -> None:
        if self._rate_limit_remaining is None or self._rate_limit_remaining > 1:
            return
        if self._rate_limit_reset_at is None:
            return
        wait_s = self._rate_limit_reset_at - time.monotonic()
        if wait_s > 0:
            time.sleep(min(wait_s, 65.0))

    def _record_rate_limit(self, headers: Mapping[str, str]) -> None:
        remaining = headers.get("x-ratelimit-remaining")
        reset = headers.get("x-ratelimit-reset")
        if remaining is not None:
            try:
                self._rate_limit_remaining = int(remaining)
            except ValueError:
                pass
        if reset is not None:
            try:
                self._rate_limit_reset_at = time.monotonic() + int(reset)
            except ValueError:
                pass

    def _get(self, endpoint: str, params: Mapping[str, Any]) -> dict[str, Any]:
        self._throttle_if_needed()
        response = self.session.get(f"{self.config.base_url}/{endpoint.lstrip('/')}", params=dict(params), headers=self._headers(), timeout=60)
        self._record_rate_limit(getattr(response, "headers", {}) or {})
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise AuxiliaryProviderError(f"OpenAQ request failed ({response.status_code}): {response.text[:300]}") from exc
        return response.json()

    def _all_pages(self, endpoint: str, params: dict[str, Any], result_key: str, *, max_items: int | None = None) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        page = 1
        while True:
            current = dict(params)
            current.update({"limit": self.config.page_limit, "page": page})
            payload = self._get(endpoint, current)
            values.extend(payload.get("results", payload.get(result_key, [])))
            meta = payload.get("meta", {})
            found = meta.get("found")
            found_count = int(found) if isinstance(found, int) or (isinstance(found, str) and found.isdigit()) else None
            if not payload.get("results", payload.get(result_key)) or (found_count is not None and len(values) >= found_count) or (max_items is not None and len(values) >= max_items):
                break
            page += 1
        return values[:max_items] if max_items is not None else values

    def build_request(self, aoi: AOI, start: str, end: str, *, variables: tuple[str, ...]) -> dict[str, Any]:
        unknown = sorted(set(variables).difference(OPENAQ_VARIABLES))
        if unknown:
            raise ValueError(f"Unsupported OpenAQ parameters: {unknown}")
        return {
            "bbox": [aoi.west, aoi.south, aoi.east, aoi.north],
            "datetime_from": f"{as_date(start).isoformat()}T00:00:00Z",
            "datetime_to": f"{as_date(end).isoformat()}T23:59:59Z",
            "parameters": [OPENAQ_VARIABLES[name]["api"] for name in variables],
        }

    def download(self, aoi: AOI, start: str, end: str, spec: AuxiliarySpec, output_dir: str | Path) -> AuxiliaryArtifact:
        if spec.options:
            config = OpenAQConfig(
                base_url=str(spec.options.get("base_url", self.config.base_url)).rstrip("/"),
                api_key=self.config.api_key,
                page_limit=int(spec.options.get("page_limit", self.config.page_limit)),
                max_locations=int(spec.options.get("max_locations", self.config.max_locations)),
            )
            provider = OpenAQProvider(config, session=self.session)
            return provider.download(aoi, start, end, AuxiliarySpec(spec.provider, spec.variables, spec.dataset), output_dir)
        variables = self.variables_for(spec)
        request = self.build_request(aoi, start, end, variables=variables)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset = spec.dataset or "openaq-v3"
        key = request_key(self.name, dataset, request)
        path = output_dir / f"{self.name}-{key.rsplit(':', 1)[-1]}.json"
        cache = AssetCache(output_dir / ".cache")
        if cache.valid(key, path):
            return artifact_from_path(path, provider=self.name, dataset=dataset, request=request, cache=cache, manifest_path=output_dir / "manifest.json")
        location_params = {"bbox": ",".join(str(value) for value in request["bbox"])}
        locations = self._all_pages("locations", location_params, "locations", max_items=self.config.max_locations)
        measurements: list[dict[str, Any]] = []
        wanted_parameters = set(request["parameters"])
        for location in locations:
            location_id = location.get("id")
            sensors = [
                sensor for sensor in location.get("sensors", [])
                if sensor.get("id") is not None
                and sensor.get("parameter", {}).get("name") in wanted_parameters
            ]
            for sensor in sensors:
                sensor_id = sensor["id"]
                params = {
                    "datetime_from": request["datetime_from"],
                    "datetime_to": request["datetime_to"],
                }
                sensor_measurements = self._all_pages(
                    f"sensors/{sensor_id}/measurements",
                    params,
                    "measurements",
                )
                for measurement in sensor_measurements:
                    measurement["location_id"] = location_id
                    measurement.setdefault("parameter", sensor.get("parameter", {}))
                measurements.extend(sensor_measurements)
        payload = {
            "provider": self.name,
            "api": f"{self.config.base_url}",
            "request": request,
            "locations": locations,
            "measurements": measurements,
        }
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
        return artifact_from_path(path, provider=self.name, dataset=dataset, request=request, cache=cache, manifest_path=output_dir / "manifest.json")

    def open(self, path: str | Path, *, dataset: str = "openaq-v3") -> xr.Dataset:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        locations = {item.get("id"): item for item in payload.get("locations", [])}
        measurements = payload.get("measurements", [])
        station_ids = sorted({item.get("location_id") for item in measurements if item.get("location_id") in locations})
        times = sorted({self._measurement_time(item) for item in measurements})
        if not station_ids or not times:
            return xr.Dataset(coords={"station": [], "time": [], "latitude": ("station", []), "longitude": ("station", [])}, attrs={"source": self.name, "product": dataset, "crs": "EPSG:4326", "analysis_shape": "station_table", "metadata_contract": "citycube-v1"})
        station_index = {value: index for index, value in enumerate(station_ids)}
        time_index = {value: index for index, value in enumerate(times)}
        values: dict[str, np.ndarray] = {}
        units: dict[str, str] = {}
        for item in measurements:
            parameter = str(item.get("parameter", {}).get("name", item.get("parameter", ""))).lower()
            name = next((canonical for canonical, info in OPENAQ_VARIABLES.items() if info["api"] == parameter), parameter.upper())
            values.setdefault(name, np.full((len(station_ids), len(times)), np.nan, dtype=float))
            parameter_info = item.get("parameter", {}) if isinstance(item.get("parameter", {}), Mapping) else {}
            units[name] = str(item.get("unit") or parameter_info.get("units") or OPENAQ_VARIABLES.get(name, {}).get("units", "unknown"))
            location_id = item.get("location_id")
            timestamp = self._measurement_time(item)
            if location_id in station_index and item.get("value") is not None:
                values[name][station_index[location_id], time_index[timestamp]] = float(item["value"])
        data_vars = {name: (("station", "time"), array, {"units": units[name], "sensor": self.name, "product": dataset, "aggregation_method": "native", "variable_role": "reference"}) for name, array in values.items()}
        return xr.Dataset(
            data_vars,
            coords={
                "station": [str(value) for value in station_ids],
                "time": np.asarray(times, dtype="datetime64[ns]"),
                "latitude": ("station", [float(locations[value]["coordinates"]["latitude"]) for value in station_ids]),
                "longitude": ("station", [float(locations[value]["coordinates"]["longitude"]) for value in station_ids]),
            },
            attrs={"source": self.name, "product": dataset, "crs": "EPSG:4326", "analysis_shape": "station_table", "processing_version": "citycube-auxiliary-v1", "metadata_contract": "citycube-v1"},
        )

    @staticmethod
    def interpolate_to_grid(
        stations: xr.Dataset,
        grid: AnalysisGrid,
        target_times: Iterable[np.datetime64],
        *,
        temporal_tolerance: np.timedelta64 = np.timedelta64(3, "D"),
        config: OpenAQInterpolationConfig | None = None,
    ) -> xr.Dataset:
        """Interpolate station observations onto a guarded analysis grid.

        Inverse-distance weighting is evaluated only where enough observations
        fall within the configured radius. The returned diagnostics make sparse
        station support and local disagreement explicit instead of extrapolating
        a visually complete but unsupported surface.
        """

        config = config or OpenAQInterpolationConfig()
        if stations.attrs.get("analysis_shape") != "station_table":
            raise CubeValidationError("OpenAQ interpolation requires a station table")
        if not {"station", "time"}.issubset(stations.dims) or not {"latitude", "longitude"}.issubset(stations.coords):
            raise CubeValidationError("OpenAQ station table is missing station/time coordinates")
        if stations.sizes.get("station", 0) == 0 or stations.sizes.get("time", 0) == 0:
            return xr.Dataset(
                coords={"time": np.asarray(tuple(target_times), dtype="datetime64[ns]"), "y": grid.y, "x": grid.x},
                attrs={**stations.attrs, "crs": grid.crs, "grid_id": grid.grid_id, "analysis_shape": "regular_grid"},
            )

        try:
            from pyproj import CRS, Transformer
        except ImportError as exc:  # pragma: no cover - pyproj is a core dependency
            raise RuntimeError("Install pyproj to interpolate OpenAQ stations") from exc

        target_crs = CRS.from_user_input(grid.crs)
        if not target_crs.is_projected:
            raise CubeValidationError("OpenAQ interpolation requires a projected target CRS")
        unit_factor = float(target_crs.axis_info[0].unit_conversion_factor or 1.0)
        station_x, station_y = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True).transform(
            np.asarray(stations.longitude.values, dtype=float),
            np.asarray(stations.latitude.values, dtype=float),
        )
        mesh_x, mesh_y = np.meshgrid(grid.x, grid.y)
        distances = np.sqrt(
            (np.asarray(station_x)[:, None] - mesh_x.ravel()[None, :]) ** 2
            + (np.asarray(station_y)[:, None] - mesh_y.ravel()[None, :]) ** 2
        ) * unit_factor
        station_times = np.asarray(stations.time.values, dtype="datetime64[ns]")
        output_times = np.asarray(tuple(target_times), dtype="datetime64[ns]")
        if output_times.ndim != 1:
            raise ValueError("OpenAQ target times must be one-dimensional")

        variable_names = [
            str(name)
            for name, value in stations.data_vars.items()
            if {"station", "time"}.issubset(value.dims) and value.dtype.kind in "fiu"
        ]
        output: dict[str, np.ndarray] = {}
        for name in variable_names:
            output[name] = np.full((len(output_times), grid.height, grid.width), np.nan, dtype=np.float32)
            output[f"{name}_station_count"] = np.zeros((len(output_times), grid.height, grid.width), dtype=np.float32)
            output[f"{name}_interpolation_uncertainty"] = np.full((len(output_times), grid.height, grid.width), np.nan, dtype=np.float32)
            output[f"{name}_interpolation_valid"] = np.zeros((len(output_times), grid.height, grid.width), dtype=bool)

        tolerance = temporal_tolerance.astype("timedelta64[ns]")
        flat_cells = distances.shape[1]
        for time_index, target_time in enumerate(output_times):
            for name in variable_names:
                raw_values = np.asarray(stations[name].values, dtype=float)
                station_values = np.full(raw_values.shape[0], np.nan, dtype=float)
                for station_index in range(raw_values.shape[0]):
                    available = np.isfinite(raw_values[station_index])
                    if not available.any():
                        continue
                    deltas = np.abs(station_times - target_time).astype("timedelta64[ns]").astype("float64")
                    deltas[~available] = np.inf
                    nearest_index = int(np.argmin(deltas))
                    if deltas[nearest_index] <= tolerance.astype("float64"):
                        station_values[station_index] = raw_values[station_index, nearest_index]
                finite_station = np.isfinite(station_values)
                for start in range(0, flat_cells, config.chunk_size):
                    stop = min(start + config.chunk_size, flat_cells)
                    cell_distances = distances[:, start:stop].copy()
                    cell_distances[~finite_station, :] = np.inf
                    within = cell_distances <= config.max_distance_m
                    counts = within.sum(axis=0)
                    order = np.argsort(cell_distances, axis=0, kind="stable")[: config.max_neighbors]
                    nearest_distances = np.take_along_axis(cell_distances, order, axis=0)
                    nearest_values = station_values[order]
                    selected = np.isfinite(nearest_distances)
                    weights = np.zeros_like(nearest_distances, dtype=float)
                    zero_distance = selected & (nearest_distances == 0)
                    positive_distance = selected & ~zero_distance
                    weights[positive_distance] = 1.0 / np.power(nearest_distances[positive_distance], config.power)
                    zero_count = zero_distance.sum(axis=0)
                    weights[zero_distance] = 1.0
                    weight_sum = weights.sum(axis=0)
                    estimates = np.sum(weights * nearest_values, axis=0) / np.where(weight_sum > 0, weight_sum, 1.0)
                    spread = np.sqrt(np.sum(weights * (nearest_values - estimates) ** 2, axis=0) / np.where(weight_sum > 0, weight_sum, 1.0))
                    supported = counts >= config.min_neighbors
                    exact = zero_count > 0
                    estimates[exact] = np.sum(np.where(zero_distance, nearest_values, 0.0), axis=0)[exact] / zero_count[exact]
                    spread[exact] = 0.0
                    cell_values = output[name][time_index].ravel()
                    cell_uncertainty = output[f"{name}_interpolation_uncertainty"][time_index].ravel()
                    cell_values[start:stop][supported] = estimates[supported]
                    cell_uncertainty[start:stop][supported] = spread[supported]
                    output[f"{name}_station_count"][time_index].ravel()[start:stop] = counts.astype(np.float32)
                    output[f"{name}_interpolation_valid"][time_index].ravel()[start:stop] = supported

        data_vars: dict[str, tuple[tuple[str, str, str], np.ndarray, dict[str, Any]]] = {}
        source = str(stations.attrs.get("source", "openaq"))
        product = str(stations.attrs.get("product", "openaq-v3"))
        for name in variable_names:
            units = str(stations[name].attrs.get("units", "unknown"))
            common = {
                "units": units,
                "sensor": "openaq",
                "product": product,
                "aggregation_method": "inverse_distance_weighting",
                "variable_role": "feature",
                "source": source,
                "interpolation_method": config.method,
                "interpolation_power": config.power,
                "interpolation_max_distance_m": config.max_distance_m,
                "interpolation_min_neighbors": config.min_neighbors,
            }
            data_vars[name] = (("time", "y", "x"), output[name], common)
            data_vars[f"{name}_station_count"] = (("time", "y", "x"), output[f"{name}_station_count"], {**common, "units": "1", "variable_role": "diagnostic"})
            data_vars[f"{name}_interpolation_uncertainty"] = (("time", "y", "x"), output[f"{name}_interpolation_uncertainty"], {**common, "units": units, "variable_role": "diagnostic"})
            data_vars[f"{name}_interpolation_valid"] = (("time", "y", "x"), output[f"{name}_interpolation_valid"], {**common, "units": "1", "variable_role": "diagnostic"})
        return xr.Dataset(
            data_vars,
            coords={"time": output_times, "y": grid.y, "x": grid.x},
            attrs={
                **stations.attrs,
                "source": source,
                "product": product,
                "crs": grid.crs,
                "grid_id": grid.grid_id,
                "resolution_m": grid.resolution_m,
                "analysis_shape": "regular_grid",
                "interpolation_method": config.method,
                "interpolation_max_distance_m": config.max_distance_m,
                "interpolation_min_neighbors": config.min_neighbors,
            },
        )

    @staticmethod
    def validate_leave_one_station_out(
        stations: xr.Dataset,
        variable: str,
        *,
        temporal_tolerance: np.timedelta64 = np.timedelta64(3, "h"),
        config: OpenAQInterpolationConfig | None = None,
    ) -> dict[str, Any]:
        """Validate IDW by withholding each station in turn.

        Each held-out observation is predicted from the remaining stations at
        their nearest available time.  The returned per-station diagnostics
        expose coverage as well as error, avoiding a misleading score based
        only on easy, well-supported observations.
        """

        config = config or OpenAQInterpolationConfig()
        if variable not in stations:
            raise CubeValidationError(f"OpenAQ station table has no variable {variable!r}")
        if stations.attrs.get("analysis_shape") != "station_table":
            raise CubeValidationError("Leave-one-station-out requires a station table")
        values = np.asarray(stations[variable].values, dtype=float)
        times = np.asarray(stations.time.values, dtype="datetime64[ns]")
        latitudes = np.asarray(stations.latitude.values, dtype=float)
        longitudes = np.asarray(stations.longitude.values, dtype=float)
        if values.ndim != 2 or values.shape != (len(latitudes), len(times)):
            raise CubeValidationError(f"{variable!r} must have station and time dimensions")
        try:
            from pyproj import Transformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install pyproj to validate OpenAQ stations") from exc
        station_x, station_y = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(longitudes, latitudes)
        distances = np.sqrt((station_x[:, None] - station_x[None, :]) ** 2 + (station_y[:, None] - station_y[None, :]) ** 2)
        tolerance_ns = temporal_tolerance.astype("timedelta64[ns]").astype("float64")
        predictions: list[float] = []
        references: list[float] = []
        held_ids: list[str] = []
        per_station: dict[str, dict[str, Any]] = {}
        station_names = [str(value) for value in stations.station.values]

        for held_index, held_name in enumerate(station_names):
            station_predictions: list[float] = []
            station_references: list[float] = []
            for time_index, target_time in enumerate(times):
                reference_value = values[held_index, time_index]
                if not np.isfinite(reference_value):
                    continue
                nearest_values: list[tuple[float, float]] = []
                for source_index in range(len(station_names)):
                    if source_index == held_index:
                        continue
                    available = np.isfinite(values[source_index])
                    if not available.any():
                        continue
                    deltas = np.abs(times - target_time).astype("timedelta64[ns]").astype("float64")
                    deltas[~available] = np.inf
                    nearest_time = int(np.argmin(deltas))
                    if deltas[nearest_time] <= tolerance_ns and distances[held_index, source_index] <= config.max_distance_m:
                        nearest_values.append((distances[held_index, source_index], values[source_index, nearest_time]))
                nearest_values.sort(key=lambda item: item[0])
                nearest_values = nearest_values[: config.max_neighbors]
                if len(nearest_values) < config.min_neighbors:
                    continue
                distances_m = np.asarray([item[0] for item in nearest_values], dtype=float)
                source_values = np.asarray([item[1] for item in nearest_values], dtype=float)
                weights = 1.0 / np.maximum(distances_m, 1.0) ** config.power
                prediction = float(np.sum(weights * source_values) / np.sum(weights))
                station_predictions.append(prediction)
                station_references.append(float(reference_value))
            if station_predictions:
                station_prediction_array = np.asarray(station_predictions)
                station_reference_array = np.asarray(station_references)
                error = station_prediction_array - station_reference_array
                station_metrics: dict[str, Any] = {
                    "samples": len(error),
                    "coverage_fraction": len(error) / int(np.isfinite(values[held_index]).sum()),
                    "bias": float(error.mean()),
                    "mae": float(np.abs(error).mean()),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                }
                predictions.extend(station_predictions)
                references.extend(station_references)
                held_ids.extend([held_name] * len(error))
            else:
                station_metrics = {"samples": 0, "coverage_fraction": 0.0, "bias": float("nan"), "mae": float("nan"), "rmse": float("nan")}
            per_station[held_name] = station_metrics
        if not predictions:
            raise ValueError("No supported observations are available for leave-one-station-out validation")
        error = np.asarray(predictions) - np.asarray(references)
        return {
            "variable": variable,
            "method": "leave_one_station_out_idw",
            "samples": len(error),
            "stations_evaluated": len(per_station),
            "coverage_fraction": len(error) / int(np.isfinite(values).sum()),
            "bias": float(error.mean()),
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "per_station": per_station,
        }

    @staticmethod
    def _measurement_time(item: Mapping[str, Any]) -> str:
        date_value = item.get("datetime", item.get("date", {}))
        if not date_value:
            period = item.get("period", {})
            date_value = period.get("datetimeFrom", {}) if isinstance(period, Mapping) else {}
        if isinstance(date_value, Mapping):
            value = date_value.get("utc") or date_value.get("local")
        else:
            value = date_value
        if not value:
            raise AuxiliaryProviderError("OpenAQ measurement has no timestamp")
        parsed = str(value).replace("Z", "+00:00")
        timestamp = datetime.fromisoformat(parsed)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
        return timestamp.isoformat()
