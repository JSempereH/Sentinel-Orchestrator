"""End-to-end planning, discovery and local execution."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping
from threading import Lock

import xarray as xr

from ..cube import validate_cube
from ..catalog import select_product_refs
from ..cache import AssetCache
from ..config import ClientConfig
from ..download import CDSEDownloader
from ..fusion import TemporalMatch, align_features, fusion_quality
from ..metadata import ensure_compatible_units, validate_variable_contract
from ..sensors.sentinel3.georeference import grid_l2_lst
from ..sensors.sentinel5p import grid_s5p
from ..harmonization.spatial import harmonize_spatial
from ..sensors.terrain import terrain_predictors, terrain_static
from ..providers import AUXILIARY_PROVIDER_FACTORIES, AuxiliarySpec, OpenAQInterpolationConfig, OpenAQProvider
from .adapters import SENSOR_ADAPTERS, AcquisitionContext, combine_sentinel3, mosaic_temporal_tiles, to_grid
from .plan import WorkflowPlan, build_plan
from .request import AnalysisRequest
from .result import AnalysisResult

# Catalogue searches return results in date order, but products are ranked
# by quality (cloud cover, coverage) only afterwards. Searching just
# `max_products_per_sensor` items would keep the *earliest* N rather than
# the *best* N, so discovery scans up to this many candidates first.
DISCOVERY_SCAN_LIMIT = 1000

# Backwards-compatible private names (tests and notebooks import these).
_combine_sentinel3 = combine_sentinel3
_mosaic_temporal_tiles = mosaic_temporal_tiles


class AnalysisWorkflow:
    """Coordinate acquisition planning and fusion of prepared sensor cubes."""

    def __init__(self, request: AnalysisRequest):
        self.request = request
        self._plan = build_plan(request)

    @property
    def plan(self) -> WorkflowPlan:
        """Return the deterministic plan without network access."""

        return self._plan

    def discover(self) -> dict[str, list]:
        """Discover and rank products for every requested sensor."""

        limit = self.request.max_products_per_sensor
        scan_limit = max(limit, DISCOVERY_SCAN_LIMIT)
        return {
            sensor: select_product_refs(SENSOR_ADAPTERS[sensor].search(self.request, limit=scan_limit), limit=limit)
            for sensor in self.request.sensors
        }

    def discover_auxiliary(self) -> dict[str, AuxiliarySpec]:
        """Return configured auxiliary selections without making network calls."""

        return {spec.provider: spec for spec in self.request.auxiliary}

    @staticmethod
    def _auxiliary_provider(spec: AuxiliarySpec):
        try:
            return AUXILIARY_PROVIDER_FACTORIES[spec.provider]()
        except KeyError:
            raise ValueError(f"Unsupported auxiliary provider: {spec.provider}") from None

    def acquire_auxiliary(self, output_dir: str | Path) -> dict[str, xr.Dataset]:
        """Download and normalize configured auxiliary sources explicitly."""

        output_dir = Path(output_dir)
        result: dict[str, xr.Dataset] = {}
        for spec in self.request.auxiliary:
            provider = self._auxiliary_provider(spec)
            artifact = provider.download(
                self.request.aoi,
                str(self.request.start),
                str(self.request.end),
                spec,
                output_dir / spec.provider,
            )
            dataset = provider.open(artifact.path, dataset=artifact.dataset)
            dataset.attrs["auxiliary_artifact_path"] = str(artifact.path)
            dataset.attrs["auxiliary_manifest_path"] = str(artifact.manifest_path)
            result[spec.provider] = dataset
        return result

    def run(self, datasets: Mapping[str, xr.Dataset], *, terrain: xr.Dataset | None = None) -> AnalysisResult:
        """Fuse already downloaded/processed datasets on one target grid.

        ``terrain`` (static ``elevation``/``slope``/``aspect`` on a fine
        grid, see ``sensors.terrain.terrain_static``) adds terrain
        predictors: aggregated elevation, slope and the solar-illumination
        ``cos_incidence`` at each target acquisition time on the fused cube,
        and static elevation/slope on the predictor cube.

        Acquisition and sensor-specific preprocessing remain explicit because
        Sentinel-1, Sentinel-3 and Sentinel-5P require different external
        processors. This method is the deterministic local execution stage.
        """

        if not datasets:
            raise ValueError("run() requires at least one prepared sensor dataset")
        prepared: dict[str, xr.Dataset] = {}
        auxiliary: dict[str, xr.Dataset] = {}
        auxiliary_names = {spec.provider for spec in self.request.auxiliary}
        for sensor, dataset in datasets.items():
            if sensor not in self.request.sensors and sensor not in auxiliary_names:
                raise ValueError(f"Dataset sensor {sensor!r} is not present in the request")
            if "time" in dataset.dims:
                # Nearest-time matching needs monotonic time on both sides;
                # callers may pass cubes concatenated in catalogue order.
                dataset = dataset.sortby("time")
            if sensor in auxiliary_names:
                if dataset.attrs.get("analysis_shape") == "station_table":
                    auxiliary[sensor] = dataset
                    continue
                if dataset.attrs.get("analysis_shape") == "swath":
                    raise ValueError(f"Auxiliary dataset {sensor!r} must be a regular grid or station table")
                if self.request.predictor_grid is not None:
                    dataset = self._to_grid(dataset, self.request.predictor_grid)
                if dataset.attrs.get("metadata_contract"):
                    validate_variable_contract(dataset)
                auxiliary[sensor] = dataset
                continue
            if dataset.attrs.get("analysis_shape") == "swath":
                dataset = grid_s5p(dataset, resolution_deg=self.request.s5p_resolution_deg, aoi=self.request.aoi)
            elif sensor == "sentinel3" and {"latitude", "longitude"}.issubset(dataset.data_vars):
                thermal_grid = self.request.thermal_grid or self.request.grid
                if thermal_grid is None:
                    raise ValueError("A thermal_grid or grid is required to georeference Sentinel-3 LST")
                dataset = grid_l2_lst(dataset, thermal_grid)
            if sensor != "sentinel3" and self.request.predictor_grid is not None:
                dataset = self._to_grid(dataset, self.request.predictor_grid)
            if dataset.attrs.get("metadata_contract"):
                validate_variable_contract(dataset)
            prepared[sensor] = dataset

        if not prepared:
            raise ValueError("At least one prepared Sentinel dataset is required; auxiliary station tables are references only")
        target_sensor = self.request.target_sensor or ("sentinel3" if "sentinel3" in prepared else next(iter(prepared)))
        if target_sensor not in prepared:
            raise ValueError(f"Target sensor {target_sensor!r} has no prepared dataset")
        target = prepared[target_sensor].copy()
        validate_cube(target, require_time=True)
        if self.request.grid:
            target.attrs.setdefault("requested_grid_id", self.request.grid.grid_id)

        auxiliary_rasters: dict[str, xr.Dataset] = {}
        interpolation_grid = self.request.predictor_grid or self.request.grid
        if interpolation_grid is not None and "openaq" in auxiliary:
            openaq_spec = next(spec for spec in self.request.auxiliary if spec.provider == "openaq")
            auxiliary_rasters["openaq"] = OpenAQProvider.interpolate_to_grid(
                auxiliary["openaq"],
                interpolation_grid,
                target.time.values,
                temporal_tolerance=self.request.tolerance_for("openaq"),
                config=OpenAQInterpolationConfig.from_options(openaq_spec.options),
            )

        merged = target
        merge_inputs = {
            **prepared,
            **{f"auxiliary_{name}": dataset for name, dataset in auxiliary.items() if "time" in dataset.dims and "x" in dataset.dims and "y" in dataset.dims},
            **{f"auxiliary_{name}": dataset for name, dataset in auxiliary_rasters.items()},
        }
        for sensor, feature in merge_inputs.items():
            if sensor == target_sensor:
                continue
            if sensor.startswith("auxiliary_"):
                feature = feature.rename({name: f"{sensor}_{name}" for name in feature.data_vars})
            feature = harmonize_spatial(target, feature)
            ensure_compatible_units(merged, feature)
            conflicts = set(feature.data_vars).intersection(merged.data_vars)
            rename = {name: f"{sensor}_{name}" for name in conflicts if name in {"valid_mask", "qa_value"}}
            if rename:
                feature = feature.rename(rename)
            merged = align_features(
                merged,
                feature,
                match=TemporalMatch(self.request.tolerance_for(sensor)),
                feature_name=sensor,
            )
        merged = fusion_quality(merged)
        merged.attrs.update({
            "workflow": "sentinel_analysis",
            "workflow_sensors": list(prepared),
            "workflow_target_sensor": target_sensor,
            "workflow_temporal_tolerance": str(self.request.temporal_tolerance),
        })
        predictor_cube = self._merge_predictors(merge_inputs)
        if terrain is not None:
            merged, predictor_cube = self._add_terrain(merged, predictor_cube, terrain)
        return AnalysisResult(
            cube=merged,
            plan=self._plan,
            provenance={"sensors": list(prepared), "auxiliary": list(auxiliary), "auxiliary_rasters": list(auxiliary_rasters), "target_sensor": target_sensor, "variables": list(merged.data_vars)},
            thermal_cube=prepared.get("sentinel3"),
            predictor_cube=predictor_cube,
            auxiliary=auxiliary or None,
            terrain=terrain,
        )

    @staticmethod
    def _add_terrain(merged: xr.Dataset, predictor_cube: xr.Dataset | None, terrain: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset | None]:
        # cos_incidence is computed at fine resolution for each target time and
        # only then averaged: illumination is non-linear in slope/aspect, so
        # aggregating slope first would misstate it. Aspect is circular and
        # has no meaningful coarse mean, so it is not aggregated.
        fine = terrain_predictors(terrain, merged.time.values, crs=terrain.attrs["crs"])
        coarse = harmonize_spatial(merged, fine)
        merged = merged.assign({name: coarse[name] for name in ("elevation", "slope", "cos_incidence")})
        if predictor_cube is not None:
            static = terrain[["elevation", "slope"]].reindex_like(predictor_cube, method="nearest", tolerance=1e-6)
            predictor_cube = predictor_cube.assign({name: static[name].broadcast_like(predictor_cube["time"]) for name in ("elevation", "slope")})
        return merged, predictor_cube

    def _merge_predictors(self, prepared: Mapping[str, xr.Dataset]) -> xr.Dataset | None:
        """Keep a native predictor-grid product separate from thermal fusion."""

        predictors = {sensor: dataset for sensor, dataset in prepared.items() if sensor != "sentinel3"}
        if not predictors:
            return None
        target_sensor = next((sensor for sensor in ("sentinel2", "sentinel1", "sentinel5p") if sensor in predictors), next(iter(predictors)))
        merged = predictors[target_sensor]
        validate_cube(merged, require_time=True)
        for sensor, feature in predictors.items():
            if sensor == target_sensor:
                continue
            if sensor.startswith("auxiliary_"):
                feature = feature.rename({name: f"{sensor}_{name}" for name in feature.data_vars})
            feature = harmonize_spatial(merged, feature)
            ensure_compatible_units(merged, feature)
            conflicts = set(feature.data_vars).intersection(merged.data_vars)
            rename = {name: f"{sensor}_{name}" for name in conflicts if name in {"valid_mask", "qa_value"}}
            if rename:
                feature = feature.rename(rename)
            merged = xr.merge(
                [merged, feature],
                compat="no_conflicts",
                join="outer",
                combine_attrs="override",
            )
        return fusion_quality(merged)

    @staticmethod
    def _to_grid(dataset: xr.Dataset, grid) -> xr.Dataset:
        """Regrid a prepared predictor dataset to the requested predictor grid."""

        return to_grid(dataset, grid)

    def execute(
        self,
        output_dir: str | Path,
        *,
        config: ClientConfig | None = None,
        gpt: str = "gpt",
        max_workers: int = 1,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> AnalysisResult:
        """Discover, download, preprocess and fuse the requested sensors."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        # Lazy: a request for "landsat" alone needs no CDSE credentials at
        # all (it reads directly from Planetary Computer's signed URLs), so
        # ClientConfig.from_env() must not run unless a CDSE-backed sensor
        # is actually requested.
        # Locked because parallel downloads may request it concurrently.
        downloader_holder: list[CDSEDownloader] = []
        downloader_lock = Lock()

        def get_downloader() -> CDSEDownloader:
            with downloader_lock:
                if not downloader_holder:
                    downloader_holder.append(CDSEDownloader(config or ClientConfig.from_env(), cache=AssetCache(output_dir / "cache")))
                return downloader_holder[0]

        context = AcquisitionContext(
            request=self.request,
            output_dir=output_dir,
            downloader_factory=get_downloader,
            max_workers=max_workers,
            gpt=gpt,
            progress=progress,
        )
        prepared: dict[str, xr.Dataset] = {}
        for sensor, references in self.discover().items():
            unique_references = list({reference.product_id: reference for reference in references}.values())
            if not unique_references:
                continue
            dataset = SENSOR_ADAPTERS[sensor].acquire(unique_references, context)
            if dataset is not None:
                prepared[sensor] = dataset
        prepared.update(self.acquire_auxiliary(output_dir / "auxiliary"))
        terrain = None
        if self.request.terrain_predictors:
            terrain_grid = self.request.predictor_grid or self.request.grid
            if terrain_grid is None:
                raise ValueError("terrain_predictors needs a predictor_grid or grid")
            terrain = terrain_static(terrain_grid)
        return self.run(prepared, terrain=terrain)
