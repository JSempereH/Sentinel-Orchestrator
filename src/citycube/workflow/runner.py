"""End-to-end planning, discovery and local execution."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Mapping
from threading import Lock

import xarray as xr

from ..cube import validate_cube
from ..catalog import annotate_aoi_coverage, filter_overpass, select_product_refs
from ..cache import AssetCache
from ..config import ClientConfig
from ..download import CDSEDownloader
from ..version import build_info
from ..downscale.per_scene import downscale_per_scene
from ..fusion import TemporalMatch, align_features, fusion_quality
from ..metadata import ensure_compatible_units, validate_variable_contract
from ..sensors.sentinel3.georeference import grid_l2_lst
from ..zones import aoi_mask
from ..sensors.sentinel5p import grid_s5p
from ..harmonization.spatial import harmonize_spatial
from ..sensors.terrain import terrain_predictors, terrain_static
from ..providers import AUXILIARY_PROVIDER_FACTORIES, AuxiliarySpec, OpenAQInterpolationConfig, OpenAQProvider
from .adapters import SENSOR_ADAPTERS, AcquisitionContext, combine_sentinel3, mosaic_temporal_tiles, to_grid
from .limits import DEFAULT_REQUEST_LIMITS, RequestLimits, check_request
from .plan import WorkflowPlan, build_plan
from .request import OVERPASS_SENSORS, AnalysisRequest
from .result import AnalysisResult

# Catalogue searches return results in date order, but products are ranked
# by quality (cloud cover, coverage) only afterwards. Searching just
# `max_products_per_sensor` items would keep the *earliest* N rather than
# the *best* N, so discovery scans up to this many candidates first.
DISCOVERY_SCAN_LIMIT = 1000
# Candidates per requested product handed to the Sentinel-3 cloud probe.
PROBE_CANDIDATE_FACTOR = 3

# Backwards-compatible private names (tests and notebooks import these).
_combine_sentinel3 = combine_sentinel3
_mosaic_temporal_tiles = mosaic_temporal_tiles

logger = logging.getLogger(__name__)


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
        """Discover and rank products for every requested sensor.

        ``thermal_overpass`` is applied before ranking, so a day-only request
        keeps the best daytime products rather than losing slots (and
        downloads) to night passes.
        """

        limit = self.request.max_products_per_sensor
        scan_limit = max(limit, DISCOVERY_SCAN_LIMIT)
        aoi = self.request.aoi
        longitude = (aoi.west + aoi.east) / 2
        discovered = {}
        for sensor in self.request.sensors:
            products = SENSOR_ADAPTERS[sensor].search(self.request, limit=scan_limit)
            if sensor in OVERPASS_SENSORS:
                products = filter_overpass(products, longitude=longitude, overpass=self.request.thermal_overpass)
            # Granule-wide cloud cover says nothing about how much of the AOI
            # a product sees: drop slivers before they take a download slot.
            products = annotate_aoi_coverage(products, aoi)
            covering = [p for p in products if p.coverage is None or p.coverage >= self.request.min_aoi_coverage]
            # With a cloud probe, Sentinel-3 gets spare candidates to replace
            # the ones the probe rejects (see Sentinel3Adapter._select_clear).
            sensor_limit = limit * PROBE_CANDIDATE_FACTOR if sensor == "sentinel3" and self.request.min_clear_fraction > 0 else limit
            discovered[sensor] = select_product_refs(covering, limit=sensor_limit)
            logger.info("%s: %d candidate products, %d cover at least %.0f%% of the AOI, %d selected",
                        sensor, len(products), len(covering), 100 * self.request.min_aoi_coverage, len(discovered[sensor]))
        return discovered

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

    def run(
        self,
        datasets: Mapping[str, xr.Dataset],
        *,
        terrain: xr.Dataset | None = None,
        downscale_store: str | Path | None = None,
    ) -> AnalysisResult:
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
                # Reanalysis and model fields are 9-40 km; the thermal grid
                # already oversamples them. Regridding hourly fields to the
                # 100 m predictor grid multiplied their size by 100 for no
                # information.
                auxiliary_grid = self.request.thermal_grid or self.request.grid or self.request.predictor_grid
                if auxiliary_grid is not None:
                    dataset = self._to_grid(dataset, auxiliary_grid)
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
            "workflow": "citycube",
            "workflow_sensors": list(prepared),
            "workflow_target_sensor": target_sensor,
            "workflow_temporal_tolerance": str(self.request.temporal_tolerance),
        })
        # One fine-resolution cube per sensor, each on its own acquisition
        # times. A single cube outer-joined every sensor's times (and once
        # ERA5's hourly ones), so 78 % of a 6-sensor Berlin cube was NaN fill.
        predictors = {sensor: dataset for sensor, dataset in prepared.items() if sensor != "sentinel3"}
        if terrain is not None:
            merged = self._add_terrain(merged, terrain)
        provenance: dict[str, Any] = {"software": build_info(), "sensors": list(prepared), "auxiliary": list(auxiliary), "auxiliary_rasters": list(auxiliary_rasters), "target_sensor": target_sensor, "variables": list(merged.data_vars)}
        downscaled = None
        spec = self.request.downscale
        if spec is not None:
            if spec.predictor_sensor not in predictors:
                raise ValueError(f"downscale needs {spec.predictor_sensor} data at fine resolution, but it produced none")
            logger.info("Downscaling %d scenes per scene with %s", merged.sizes["time"], spec.model)
            downscaled = downscale_per_scene(
                merged,
                predictors[spec.predictor_sensor],
                target=spec.target,
                predictors=spec.predictors or None,
                model=spec.model,
                coarse_consistent=spec.coarse_consistent,
                min_samples=spec.min_samples,
                predictor_sensor=spec.predictor_sensor,
                terrain=terrain,
                model_options=spec.model_options,
                correction=spec.correction,
                mask_unobserved=spec.mask_unobserved,
                store=downscale_store,
                domain=aoi_mask(predictors[spec.predictor_sensor], self.request.aoi) if self.request.aoi.has_geometry else None,
            )
            provenance["downscaling"] = {
                key: downscaled.attrs[f"downscaling_{key}"] for key in ("protocol", "model", "predictors", "scenes", "skipped_scenes")
            }
        return AnalysisResult(
            cube=merged,
            plan=self._plan,
            provenance=provenance,
            thermal_cube=prepared.get("sentinel3"),
            predictors=predictors,
            auxiliary={**auxiliary, **{f"{name}_grid": raster for name, raster in auxiliary_rasters.items()}} or None,
            terrain=terrain,
            downscaled=downscaled,
        )

    @staticmethod
    def _add_terrain(merged: xr.Dataset, terrain: xr.Dataset) -> xr.Dataset:
        # cos_incidence is computed at fine resolution for each target time and
        # only then averaged: illumination is non-linear in slope/aspect, so
        # aggregating slope first would misstate it. Aspect is circular and
        # has no meaningful coarse mean, so it is not aggregated. The fine
        # static terrain stays in `AnalysisResult.terrain`, without a time axis.
        fine = terrain_predictors(terrain, merged.time.values, crs=terrain.attrs["crs"])
        coarse = harmonize_spatial(merged, fine)
        return merged.assign({name: coarse[name] for name in ("elevation", "slope", "cos_incidence")})

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
        limits: RequestLimits | None = DEFAULT_REQUEST_LIMITS,
    ) -> AnalysisResult:
        """Discover, download, preprocess and fuse the requested sensors.

        The request is first checked against ``limits`` (``None`` disables
        the check) so an oversized request fails before any download.
        Products that fail under ``on_product_error="skip"`` are listed in
        ``result.provenance["failed_products"]``.
        """

        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        if limits is not None:
            check_request(self.request, limits)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
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
        # Auxiliary sources first: they are small but their request errors
        # (an unavailable CAMS date, a missing licence) would otherwise only
        # surface after every satellite product had been downloaded.
        auxiliary = self.acquire_auxiliary(output_dir / "auxiliary")
        prepared: dict[str, xr.Dataset] = {}
        for sensor, references in self.discover().items():
            unique_references = list({reference.product_id: reference for reference in references}.values())
            if not unique_references:
                logger.warning("%s: no products found for this AOI and period", sensor)
                continue
            logger.info("%s: acquiring %d products", sensor, len(unique_references))
            dataset = SENSOR_ADAPTERS[sensor].acquire(unique_references, context)
            if dataset is not None:
                prepared[sensor] = dataset
                logger.info("%s: prepared %d acquisitions", sensor, dataset.sizes.get("time", 1))
        prepared.update(auxiliary)
        terrain = None
        if self.request.terrain_predictors:
            terrain_grid = self.request.predictor_grid or self.request.grid
            if terrain_grid is None:
                raise ValueError("terrain_predictors needs a predictor_grid or grid")
            terrain = terrain_static(terrain_grid)
        store = output_dir / "downscaled.zarr" if self.request.downscale is not None else None
        result = self.run(prepared, terrain=terrain, downscale_store=store)
        logger.info("Fused cube: %d times, %d variables", result.cube.sizes["time"], len(result.cube.data_vars))
        result.provenance["failed_products"] = list(context.failures)
        result.provenance["probed_out"] = list(context.probed_out)
        if context.failures:
            logger.warning("%d products were skipped after failing; see provenance['failed_products']", len(context.failures))
        return result
