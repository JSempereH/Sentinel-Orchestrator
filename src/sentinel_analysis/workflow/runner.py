"""End-to-end planning, discovery and local execution."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping
import zipfile

import xarray as xr
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging

from ..cube import validate_cube
from ..catalog import select_product_refs
from ..cache import AssetCache
from ..config import ClientConfig
from ..download import CDSEDownloader
from ..fusion import TemporalMatch, align_features, fusion_quality
from ..metadata import ensure_compatible_units, validate_variable_contract
from ..sensors.ecostress import EcostressCatalog, read_ecostress_lst
from ..sensors.landsat import LandsatCatalog, read_landsat_lst
from ..sensors.sentinel1 import Sentinel1Catalog, process_s1, process_s1_rtc, read_s1_ard, read_s1_grd, read_s1_rtc, sentinel1_indices
from ..sensors.sentinel2 import Sentinel2Catalog, compose_s2, read_s2_l2a, sentinel2_indices
from ..sensors.sentinel3.catalog import CDSECatalog, ProductQuery
from ..sensors.sentinel3.georeference import grid_l2_lst
from ..sensors.sentinel3.processing import apply_quality_mask, to_celsius
from ..sensors.sentinel3.pipeline import Sentinel3LST
from ..sensors.sentinel5p import Sentinel5PCatalog, grid_s5p
from ..harmonization.spatial import harmonize_spatial
from ..providers import AuxiliarySpec, CAMSProvider, CarbonMapperProvider, ERA5Provider, OpenAQInterpolationConfig, OpenAQProvider
from .plan import WorkflowPlan, build_plan
from .request import AnalysisRequest
from .result import AnalysisResult


def _mosaic_temporal_tiles(datasets: list[xr.Dataset]) -> list[xr.Dataset]:
    """Mosaic same-time tiles and retain one dataset per acquisition time."""

    grouped: dict[np.datetime64, list[xr.Dataset]] = {}
    for dataset in datasets:
        if "time" not in dataset.coords or dataset.sizes.get("time", 0) != 1:
            raise ValueError("Sentinel-2 acquisitions must contain exactly one time step")
        timestamp = np.datetime64(dataset.time.values[0], "ns")
        grouped.setdefault(timestamp, []).append(dataset)

    mosaics: list[xr.Dataset] = []
    for timestamp in sorted(grouped):
        candidates = grouped[timestamp]
        merged = candidates[0]
        for candidate in candidates[1:]:
            same_grid = (
                np.array_equal(merged.x.values, candidate.x.values)
                and np.array_equal(merged.y.values, candidate.y.values)
                and merged.attrs.get("crs") == candidate.attrs.get("crs")
            )
            if not same_grid:
                coverage = [
                    int(item["valid_mask"].sum().item()) if "valid_mask" in item else 0
                    for item in (merged, candidate)
                ]
                merged = merged if coverage[0] >= coverage[1] else candidate
                continue
            for name in candidate.data_vars:
                if name not in merged:
                    merged[name] = candidate[name]
                elif name == "valid_mask":
                    merged[name] = merged[name].fillna(0).astype(bool) | candidate[name].fillna(0).astype(bool)
                else:
                    merged[name] = merged[name].combine_first(candidate[name])
        mosaics.append(merged)
    return mosaics


def _combine_sentinel3(raw_datasets: list[xr.Dataset], grid) -> xr.Dataset:
    """Georeference each raw Sentinel-3 acquisition individually, then
    concatenate the regridded, common-grid results along time.

    Each acquisition has its own swath geometry (and often a different
    native pixel-array shape, since raw L2 LST products are not cropped to
    a fixed size). Concatenating raw acquisitions *before* regridding - as
    if they shared one geolocation - either crashes when their native
    shapes differ, or silently mis-georeferences every slice but the first
    when they happen to match (see grid_l2_lst's
    _require_single_geolocation guard). Regridding first sidesteps both:
    every result already shares the exact same grid.height x grid.width
    shape by construction, so the final concat is always safe.
    """

    if not raw_datasets:
        raise ValueError("At least one product path is required")
    georeferenced = [grid_l2_lst(apply_quality_mask(raw), grid) for raw in raw_datasets]
    if len(georeferenced) == 1:
        return georeferenced[0]
    return xr.concat(georeferenced, dim="time", data_vars="minimal", coords="minimal", compat="override")


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
        """Discover products for every requested sensor through CDSE."""

        products: dict[str, list] = {}
        for sensor in self.request.sensors:
            if sensor == "sentinel1":
                products[sensor] = Sentinel1Catalog().search(self.request.aoi, self.request.start, self.request.end, limit=self.request.max_products_per_sensor)
            elif sensor == "sentinel2":
                products[sensor] = Sentinel2Catalog().search(
                    self.request.aoi,
                    self.request.start,
                    self.request.end,
                    cloud_cover_max=self.request.s2_cloud_cover_max,
                    limit=self.request.max_products_per_sensor,
                )
            elif sensor == "sentinel3":
                products[sensor] = CDSECatalog().search(ProductQuery(self.request.aoi, self.request.start, self.request.end), limit=self.request.max_products_per_sensor)
            elif sensor == "sentinel5p":
                gas = next((value for value in self.request.variables if value.upper() in {"NO2", "SO2", "CO", "O3", "CH4", "HCHO", "AER_AI"}), "NO2")
                products[sensor] = Sentinel5PCatalog().search(self.request.aoi, self.request.start, self.request.end, gas=gas, limit=self.request.max_products_per_sensor)
            elif sensor == "landsat":
                products[sensor] = LandsatCatalog().search(self.request.aoi, self.request.start, self.request.end, limit=self.request.max_products_per_sensor)
            elif sensor == "ecostress":
                products[sensor] = EcostressCatalog().search(self.request.aoi, self.request.start, self.request.end, limit=self.request.max_products_per_sensor)
        return {
            sensor: select_product_refs(references, limit=self.request.max_products_per_sensor)
            for sensor, references in products.items()
        }

    def discover_auxiliary(self) -> dict[str, AuxiliarySpec]:
        """Return configured auxiliary selections without making network calls."""

        return {spec.provider: spec for spec in self.request.auxiliary}

    @staticmethod
    def _auxiliary_provider(spec: AuxiliarySpec):
        if spec.provider == "era5":
            return ERA5Provider()
        if spec.provider == "cams":
            return CAMSProvider()
        if spec.provider == "openaq":
            return OpenAQProvider()
        if spec.provider == "carbon_mapper":
            return CarbonMapperProvider()
        raise ValueError(f"Unsupported auxiliary provider: {spec.provider}")

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

    def run(self, datasets: Mapping[str, xr.Dataset]) -> AnalysisResult:
        """Fuse already downloaded/processed datasets on one target grid.

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
            if sensor in auxiliary_names:
                if dataset.attrs.get("analysis_shape") in {"station_table", "point_table"}:
                    # Station tables (OpenAQ) are repeated timeseries at
                    # fixed locations that can be interpolated onto a grid;
                    # point tables (Carbon Mapper) are one-off event catalogs
                    # with no such structure - both are kept as references
                    # rather than rasterized.
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
        return AnalysisResult(
            cube=merged,
            plan=self._plan,
            provenance={"sensors": list(prepared), "auxiliary": list(auxiliary), "auxiliary_rasters": list(auxiliary_rasters), "target_sensor": target_sensor, "variables": list(merged.data_vars)},
            thermal_cube=prepared.get("sentinel3"),
            predictor_cube=predictor_cube,
            auxiliary=auxiliary or None,
        )

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

        if np.array_equal(dataset.x.values, grid.x) and np.array_equal(dataset.y.values, grid.y) and dataset.attrs.get("crs") == grid.crs:
            return dataset
        template = xr.Dataset(
            {"_grid_template": (("time", "y", "x"), np.zeros((dataset.sizes["time"], grid.height, grid.width), dtype=np.float32))},
            coords={"time": dataset.time, "y": grid.y, "x": grid.x},
            attrs={"crs": grid.crs, "grid_id": grid.grid_id},
        )
        result = harmonize_spatial(template, dataset)
        return result.drop_vars("_grid_template", errors="ignore")

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
        logger = logging.getLogger(__name__)
        # Lazy: a request for "landsat" alone needs no CDSE credentials at
        # all (it reads directly from Planetary Computer's signed URLs), so
        # ClientConfig.from_env() must not run unless a CDSE-backed sensor
        # is actually requested.
        downloader_holder: list[CDSEDownloader] = []

        def get_downloader() -> CDSEDownloader:
            if not downloader_holder:
                downloader_holder.append(CDSEDownloader(config or ClientConfig.from_env(), cache=AssetCache(output_dir / "cache")))
            return downloader_holder[0]

        products = self.discover()
        prepared: dict[str, xr.Dataset] = {}
        for sensor, references in products.items():
            if not references:
                continue
            unique_references = {reference.product_id: reference for reference in references}
            download_args = list(unique_references.values())

            if sensor == "landsat":
                landsat_grid = self.request.predictor_grid or self.request.grid
                if landsat_grid is None:
                    raise ValueError("A predictor_grid or grid is required to reproject Landsat scenes")
                datasets = [read_landsat_lst(reference, landsat_grid) for reference in download_args]
                if progress:
                    progress(sensor, len(datasets), len(download_args))
                prepared[sensor] = xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]
                continue

            if sensor == "ecostress":
                ecostress_grid = self.request.predictor_grid or self.request.grid
                if ecostress_grid is None:
                    raise ValueError("A predictor_grid or grid is required to reproject ECOSTRESS scenes")
                datasets = [read_ecostress_lst(reference, ecostress_grid) for reference in download_args]
                if progress:
                    progress(sensor, len(datasets), len(download_args))
                # A single overpass can split an AOI across more than one MGRS
                # tile (same acquisition time, different item) - mosaic those
                # before concatenating along time, exactly like Sentinel-2.
                datasets = _mosaic_temporal_tiles(datasets)
                prepared[sensor] = xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]
                continue

            if sensor == "sentinel1" and self.request.sentinel1_backend == "hyp3_rtc":
                # Never downloads the raw GRD from CDSE at all: ASF HyP3
                # processes the named granule entirely in the cloud, so the
                # only input is `reference.name` (the SAFE product name).
                datasets = [
                    sentinel1_indices(read_s1_rtc(process_s1_rtc(reference.name, output_dir / "processed" / sensor).output_path))
                    for reference in download_args
                ]
                if progress:
                    progress(sensor, len(datasets), len(download_args))
                prepared[sensor] = xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]
                continue

            def download(reference, sensor=sensor):
                logger.info("Downloading %s", reference.name)
                return get_downloader().download(reference, output_dir / "downloads" / sensor)
            if max_workers > 1 and len(download_args) > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    archives = list(executor.map(download, download_args))
            else:
                archives = [download(reference) for reference in download_args]
            if progress:
                progress(sensor, len(archives), len(download_args))
            if sensor == "sentinel3":
                thermal_grid = self.request.thermal_grid or self.request.grid
                if thermal_grid is None:
                    raise ValueError("A thermal_grid or grid is required to georeference Sentinel-3 LST")
                raw_datasets = [Sentinel3LST.read(archive) for archive in archives]
                prepared[sensor] = to_celsius(_combine_sentinel3(raw_datasets, thermal_grid))
            elif sensor == "sentinel2":
                datasets = [sentinel2_indices(read_s2_l2a(archive, aoi=self.request.aoi)) for archive in archives]
                if self.request.predictor_grid is not None:
                    datasets = [self._to_grid(dataset, self.request.predictor_grid) for dataset in datasets]
                datasets = _mosaic_temporal_tiles(datasets)
                combined = xr.concat(datasets, dim="time")
                prepared[sensor] = compose_s2(
                    combined,
                    method=self.request.s2_composite_method,
                    min_observations=self.request.s2_min_observations,
                ) if self.request.s2_composite_method else combined
            elif sensor == "sentinel1":
                datasets = []
                for archive in archives:
                    extracted = get_downloader().extract(archive, output_dir / "extracted" / sensor / Path(archive).stem)
                    result = process_s1(extracted, output_dir / "processed" / sensor, backend=self.request.sentinel1_backend, gpt=gpt)
                    reader = read_s1_ard if self.request.sentinel1_backend == "s1ard" else read_s1_grd
                    datasets.append(sentinel1_indices(reader(result.output_path)))
                prepared[sensor] = xr.concat(datasets, dim="time")
            elif sensor == "sentinel5p":
                datasets = []
                for archive in archives:
                    extracted = archive
                    if zipfile.is_zipfile(archive):
                        extracted = get_downloader().extract(archive, output_dir / "extracted" / sensor / Path(archive).stem)
                    files = sorted(extracted.rglob("*.nc")) if extracted.is_dir() else [extracted]
                    if not files:
                        raise FileNotFoundError(f"No NetCDF product found after extracting {archive}")
                    from ..sensors.sentinel5p import read_s5p_l2
                    try:
                        datasets.append(grid_s5p(read_s5p_l2(files[0]), resolution_deg=self.request.s5p_resolution_deg, aoi=self.request.aoi))
                    except ValueError as exc:
                        if "no valid observations in the requested AOI" not in str(exc):
                            raise
                if not datasets:
                    logger.warning("No Sentinel-5P product contains valid observations in the requested AOI; continuing without Sentinel-5P")
                    continue
                prepared[sensor] = xr.concat(datasets, dim="time")
        prepared.update(self.acquire_auxiliary(output_dir / "auxiliary"))
        return self.run(prepared)
