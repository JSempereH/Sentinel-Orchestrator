"""Per-sensor acquisition adapters used by ``AnalysisWorkflow``.

Each adapter owns everything sensor-specific: how to search for products and
how to turn the selected products into one prepared, time-sorted cube. The
workflow runner only loops over ``SENSOR_ADAPTERS``; adding a sensor means
writing one adapter here and listing its name in
``request.SUPPORTED_SENSORS``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import logging
from pathlib import Path
import shutil
from threading import Lock
from typing import TYPE_CHECKING, Callable, Iterable, Mapping, Protocol, TypeVar
import zipfile

import numpy as np
import requests
import xarray as xr

from ..cache import AssetCache
from ..catalog import ProductRef, filter_overpass
from ..cube import AnalysisGrid
from ..download import CDSEDownloader
from ..harmonization.spatial import harmonize_spatial
from ..sensors.ecostress import EcostressCatalog, read_ecostress_lst
from ..sensors.landsat import LandsatCatalog, read_landsat_lst
from ..sensors.sentinel1 import (
    Sentinel1Catalog,
    Sentinel1RTCSTACCatalog,
    process_s1,
    process_s1_rtc,
    read_s1_ard,
    read_s1_grd,
    read_s1_rtc,
    read_s1_rtc_cog,
    sentinel1_indices,
)
from ..sensors.sentinel2 import Sentinel2Catalog, Sentinel2STACCatalog, compose_s2, read_s2_l2a, read_s2_l2a_cog, sentinel2_indices
from ..sensors.sentinel3.catalog import CDSECatalog, ProductQuery
from ..sensors.sentinel3.georeference import grid_l2_lst
from ..sensors.sentinel3.processing import apply_quality_mask, to_celsius
from ..sensors.sentinel3.reader import SLSTR_LST_FILES, SLSTR_PROBE_FILES, aoi_clear_fraction, read_l2_lst, subset_clear_fraction
from ..storage import NETCDF_LOCK, netcdf_safe
from ..sensors.sentinel5p import Sentinel5PCatalog, Sentinel5PReadConfig, grid_s5p, read_s5p_l2

if TYPE_CHECKING:
    from .request import AnalysisRequest

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int], None]
# Bump when what a stored AOI subset contains changes, so older subsets are
# rebuilt instead of silently reused.
SUBSET_FORMAT_VERSION = "1"
T = TypeVar("T")
S5P_GASES = {"NO2", "SO2", "CO", "O3", "CH4", "HCHO", "AER_AI"}


def _locked_read(reader: Callable[..., xr.Dataset], *args: object, **kwargs: object) -> xr.Dataset:
    """Read and load a NetCDF product under NETCDF_LOCK (downloads stay parallel)."""

    with NETCDF_LOCK:
        return reader(*args, **kwargs).load()


class ProductAcquisitionError(RuntimeError):
    """Every product selected for a sensor failed to download or read."""


def concat_time(datasets: list[xr.Dataset]) -> xr.Dataset:
    """Concatenate acquisitions along time in chronological order.

    Products arrive in catalogue-ranking order (cloud cover first), not by
    date; nearest-time matching downstream requires a monotonic time index.
    """

    if not datasets:
        raise ValueError("At least one dataset is required")
    combined = xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]
    return combined.sortby("time")


def mosaic_temporal_tiles(datasets: list[xr.Dataset]) -> list[xr.Dataset]:
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


def combine_sentinel3(raw_datasets: Iterable[xr.Dataset], grid) -> xr.Dataset:
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

    Pass a generator to keep memory flat: each raw swath is then read,
    gridded and released before the next one is opened.
    """

    georeferenced = [grid_l2_lst(apply_quality_mask(raw), grid) for raw in raw_datasets]
    if not georeferenced:
        raise ValueError("At least one product path is required")
    if len(georeferenced) == 1:
        return georeferenced[0]
    combined = xr.concat(georeferenced, dim="time", data_vars="minimal", coords="minimal", compat="override")
    return combined.sortby("time")


def to_grid(dataset: xr.Dataset, grid: AnalysisGrid) -> xr.Dataset:
    """Regrid a prepared dataset onto ``grid`` (no-op if already on it)."""

    if np.array_equal(dataset.x.values, grid.x) and np.array_equal(dataset.y.values, grid.y) and dataset.attrs.get("crs") == grid.crs:
        return dataset
    template = xr.Dataset(
        {"_grid_template": (("time", "y", "x"), np.zeros((dataset.sizes["time"], grid.height, grid.width), dtype=np.float32))},
        coords={"time": dataset.time, "y": grid.y, "x": grid.x},
        attrs={"crs": grid.crs, "grid_id": grid.grid_id},
    )
    result = harmonize_spatial(template, dataset)
    return result.drop_vars("_grid_template", errors="ignore")


@dataclass
class AcquisitionContext:
    """Shared execution state handed to every adapter's ``acquire``."""

    request: "AnalysisRequest"
    output_dir: Path
    downloader_factory: Callable[[], CDSEDownloader]
    max_workers: int = 1
    gpt: str = "gpt"
    progress: ProgressCallback | None = None
    # One record per product left out under on_product_error="skip".
    failures: list[dict[str, str]] = field(default_factory=list)
    # Products a cloud probe found too cloudy over the AOI to download.
    probed_out: list[dict[str, object]] = field(default_factory=list)
    _failures_lock: Lock = field(default_factory=Lock, repr=False)

    @property
    def downloader(self) -> CDSEDownloader:
        return self.downloader_factory()

    def predictor_grid(self, sensor: str) -> AnalysisGrid:
        grid = self.request.predictor_grid or self.request.grid
        if grid is None:
            raise ValueError(f"A predictor_grid or grid is required to reproject {sensor} scenes")
        return grid

    def thermal_grid(self, sensor: str) -> AnalysisGrid:
        grid = self.request.thermal_grid or self.request.grid
        if grid is None:
            raise ValueError(f"A thermal_grid or grid is required to georeference {sensor}")
        return grid

    def report(self, sensor: str, done: int, total: int) -> None:
        if self.progress:
            self.progress(sensor, done, total)

    def map_products(self, sensor: str, references: list[ProductRef], function: Callable[[ProductRef], T], *, parallel: bool = True) -> list[T]:
        """Apply ``function`` per product, in parallel only when allowed.

        Under ``on_product_error="skip"`` a product whose ``function`` raises
        is logged, recorded in ``failures`` and left out of the result, so one
        corrupt download or unreadable scene no longer aborts the whole run.
        If every product fails, ``ProductAcquisitionError`` is raised instead:
        an empty sensor would silently change what the run produces (e.g.
        another sensor becoming the fusion target).
        """

        def guarded(reference: ProductRef) -> T | None:
            try:
                return function(reference)
            except Exception as exc:
                if self.request.on_product_error == "raise":
                    raise
                logger.warning("Skipping %s product %s: %s", sensor, reference.name, exc, exc_info=True)
                with self._failures_lock:
                    self.failures.append({"sensor": sensor, "product": reference.name, "error": f"{type(exc).__name__}: {exc}"})
                return None

        if parallel and self.max_workers > 1 and len(references) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(guarded, references))
        else:
            results = [guarded(reference) for reference in references]
        self.report(sensor, len(results), len(references))
        kept = [result for result in results if result is not None]
        if references and not kept:
            errors = "; ".join(f"{item['product']}: {item['error']}" for item in self.failures if item["sensor"] == sensor)
            raise ProductAcquisitionError(f"All {len(references)} {sensor} products failed: {errors}")
        return kept

    def extract(self, sensor: str, archive: Path) -> Path:
        return self.downloader.extract(archive, self.output_dir / "extracted" / sensor / Path(archive).stem)

    def subset_path(self, sensor: str, reference: ProductRef) -> Path:
        """Where ``subset`` stores this product's AOI subset."""

        aoi = self.request.aoi
        key = hashlib.sha1(f"{aoi.west:.6f},{aoi.south:.6f},{aoi.east:.6f},{aoi.north:.6f}:{SUBSET_FORMAT_VERSION}".encode(), usedforsecurity=False).hexdigest()[:12]
        return self.output_dir / "subsets" / sensor / f"{reference.name}__{key}.nc"

    def subset(self, sensor: str, reference: ProductRef, build: Callable[[], tuple[xr.Dataset, list[Path]]]) -> xr.Dataset:
        """Return a product's AOI subset, building and storing it once.

        ``build`` downloads/reads the product and returns the AOI-cropped
        dataset plus the raw paths it created. The subset is written as
        NetCDF under ``subsets/<sensor>/`` (keyed by product, AOI and subset
        format) and re-used by later runs over the same AOI without any
        download. With ``raw_retention="aoi_subset"`` the raw paths are then
        deleted; a larger AOI or another variable later means downloading
        the product again (the product id stays in the subset's attributes).
        """

        aoi = self.request.aoi
        path = self.subset_path(sensor, reference)
        directory = path.parent
        directory.mkdir(parents=True, exist_ok=True)
        cache = AssetCache(directory / ".cache")
        if path.exists() and cache.valid(path.name, path):
            with NETCDF_LOCK, xr.open_dataset(path) as stored:
                return stored.load()
        dataset, raw_paths = build()
        with NETCDF_LOCK:  # build() may return a lazily read NetCDF
            dataset = dataset.load()
            dataset.close()
        dataset.attrs.update({"source_product_id": reference.product_id, "source_product_name": reference.name})
        partial = path.with_name(path.name + ".part")
        with NETCDF_LOCK:
            netcdf_safe(dataset).to_netcdf(partial)
        partial.replace(path)
        cache.record(path.name, path, metadata={"product_id": reference.product_id, "aoi": [aoi.west, aoi.south, aoi.east, aoi.north]})
        if self.request.raw_retention == "aoi_subset":
            for raw in raw_paths:
                if raw.is_dir():
                    shutil.rmtree(raw, ignore_errors=True)
                else:
                    raw.unlink(missing_ok=True)
        return dataset


class SensorAdapter(Protocol):
    """Search and acquisition contract for one sensor."""

    name: str

    def search(self, request: "AnalysisRequest", *, limit: int) -> list[ProductRef]: ...

    def acquire(self, references: list[ProductRef], context: AcquisitionContext) -> xr.Dataset | None: ...


class Sentinel1Adapter:
    name = "sentinel1"

    def search(self, request: "AnalysisRequest", *, limit: int) -> list[ProductRef]:
        if request.sentinel1_backend == "pc_rtc":
            return Sentinel1RTCSTACCatalog().search(request.aoi, request.start, request.end, limit=limit)
        return Sentinel1Catalog().search(request.aoi, request.start, request.end, limit=limit)

    def acquire(self, references: list[ProductRef], context: AcquisitionContext) -> xr.Dataset | None:
        backend = context.request.sentinel1_backend
        if backend == "pc_rtc":
            grid = context.predictor_grid(self.name)
            return concat_time(context.map_products(self.name, references, lambda reference: sentinel1_indices(read_s1_rtc_cog(reference, grid))))
        processed = context.output_dir / "processed" / self.name
        if backend == "hyp3_rtc":
            # Never downloads the raw GRD from CDSE at all: ASF HyP3 processes
            # the named granule entirely in the cloud, so the only input is
            # `reference.name` (the SAFE product name).
            def hyp3(reference: ProductRef) -> xr.Dataset:
                return sentinel1_indices(read_s1_rtc(process_s1_rtc(reference.name, processed).output_path))

            return concat_time(context.map_products(self.name, references, hyp3, parallel=False))
        reader = read_s1_ard if backend == "s1ard" else read_s1_grd

        # Serial: each local SNAP/s1ard run needs ~11 GB of RAM.
        def local(reference: ProductRef) -> xr.Dataset:
            archive = context.downloader.download(reference, context.output_dir / "downloads" / self.name)
            extracted = context.extract(self.name, archive)
            result = process_s1(extracted, processed, backend=backend, gpt=context.gpt)
            return sentinel1_indices(reader(result.output_path))

        return concat_time(context.map_products(self.name, references, local, parallel=False))


class Sentinel2Adapter:
    name = "sentinel2"

    def search(self, request: "AnalysisRequest", *, limit: int) -> list[ProductRef]:
        catalog = Sentinel2STACCatalog() if request.sentinel2_source == "stac_cog" else Sentinel2Catalog()
        return catalog.search(request.aoi, request.start, request.end, cloud_cover_max=request.s2_cloud_cover_max, limit=limit)

    def acquire(self, references: list[ProductRef], context: AcquisitionContext) -> xr.Dataset | None:
        request = context.request
        if request.sentinel2_source == "stac_cog":
            grid = context.predictor_grid(self.name)
            datasets = context.map_products(self.name, references, lambda reference: sentinel2_indices(read_s2_l2a_cog(reference, grid)))
        else:
            def safe_subset(reference: ProductRef) -> xr.Dataset:
                def build() -> tuple[xr.Dataset, list[Path]]:
                    archive = context.downloader.download(reference, context.output_dir / "downloads" / self.name)
                    return read_s2_l2a(archive, aoi=request.aoi), [archive]
                return context.subset(self.name, reference, build)

            datasets = [sentinel2_indices(dataset) for dataset in context.map_products(self.name, references, safe_subset)]
            if request.predictor_grid is not None:
                datasets = [to_grid(dataset, request.predictor_grid) for dataset in datasets]
        combined = concat_time(mosaic_temporal_tiles(datasets))
        if request.s2_composite_method:
            return compose_s2(combined, method=request.s2_composite_method, min_observations=request.s2_min_observations)
        return combined


class Sentinel3Adapter:
    name = "sentinel3"

    def search(self, request: "AnalysisRequest", *, limit: int) -> list[ProductRef]:
        return CDSECatalog().search(ProductQuery(request.aoi, request.start, request.end), limit=limit)

    def acquire(self, references: list[ProductRef], context: AcquisitionContext) -> xr.Dataset | None:
        grid = context.thermal_grid("Sentinel-3 LST")
        downloads = context.output_dir / "downloads" / self.name
        if context.request.min_clear_fraction > 0:
            references = self._select_clear(references, context, downloads)

        def product_subset(reference: ProductRef) -> xr.Dataset:
            def build() -> tuple[xr.Dataset, list[Path]]:
                archive = downloads / f"{reference.name}.zip"
                if archive.exists():  # a full archive from an earlier run
                    return _locked_read(read_l2_lst, archive, aoi=context.request.aoi), [archive]
                try:
                    # Only the ~16 MB of files the analysis reads, not the ~70 MB archive.
                    root = context.downloader.download_files(reference, SLSTR_LST_FILES, downloads)
                except requests.HTTPError:
                    logger.warning("Partial download of %s failed; downloading the full archive", reference.name)
                    archive = context.downloader.download(reference, downloads)
                    return _locked_read(read_l2_lst, archive, aoi=context.request.aoi), [archive]
                return _locked_read(read_l2_lst, root, aoi=context.request.aoi), [root]
            return context.subset(self.name, reference, build)

        subsets = context.map_products(self.name, references, product_subset)
        return to_celsius(combine_sentinel3(subsets, grid))

    def _select_clear(self, references: list[ProductRef], context: AcquisitionContext, downloads: Path) -> list[ProductRef]:
        """Probe candidates in ranking order until enough are clear over the AOI.

        Each probe downloads only ``SLSTR_PROBE_FILES`` (~14 MB) and measures
        the share of the AOI that is clear and within the view-angle cut; the rest of
        a product is fetched only if it passes. Catalogue cloud cover is per
        granule and says little about the AOI: without this, many of the
        products downloaded for a downscaling run were later skipped as
        clouded. Products already stored as AOI subsets are judged from the
        subset, with no download.
        """

        request = context.request
        accepted: list[ProductRef] = []
        for reference in references:
            if len(accepted) >= request.max_products_per_sensor:
                break
            root = None
            subset = context.subset_path(self.name, reference)
            try:
                if subset.exists():
                    # Judged from the stored subset: being cached says nothing about its clouds.
                    with NETCDF_LOCK:
                        clear = subset_clear_fraction(subset, request.aoi)
                else:
                    root = context.downloader.download_files(reference, SLSTR_PROBE_FILES, downloads)
                    with NETCDF_LOCK:
                        clear = aoi_clear_fraction(root, request.aoi)
            except (requests.RequestException, OSError, KeyError, ValueError) as exc:
                logger.warning("Could not probe %s (%s); acquiring it unprobed", reference.name, exc)
                accepted.append(reference)
                continue
            if clear is not None and clear >= request.min_clear_fraction:
                accepted.append(reference)
                continue
            context.probed_out.append({"product": reference.name, "aoi_clear_fraction": None if clear is None else round(clear, 3)})
            if root is not None and request.raw_retention == "aoi_subset":
                shutil.rmtree(root, ignore_errors=True)
        logger.info("sentinel3: probed %d candidates for clear sky over the AOI, kept %d", len(accepted) + len(context.probed_out), len(accepted))
        return accepted


class Sentinel5PAdapter:
    name = "sentinel5p"

    @staticmethod
    def gas(request: "AnalysisRequest") -> str:
        return next((value.upper() for value in request.variables if value.upper() in S5P_GASES), "NO2")

    def search(self, request: "AnalysisRequest", *, limit: int) -> list[ProductRef]:
        # TROPOMI's trace-gas retrievals need sunlight: night-side orbits
        # come back empty after a ~600 MB download each.
        products = Sentinel5PCatalog().search(request.aoi, request.start, request.end, gas=self.gas(request), limit=limit)
        aoi = request.aoi
        return filter_overpass(products, longitude=(aoi.west + aoi.east) / 2, overpass="day")

    def acquire(self, references: list[ProductRef], context: AcquisitionContext) -> xr.Dataset | None:
        request = context.request
        config = Sentinel5PReadConfig(gas=self.gas(request))  # previously always NO2, whatever was searched

        def product_subset(reference: ProductRef) -> xr.Dataset:
            def build() -> tuple[xr.Dataset, list[Path]]:
                archive = context.downloader.download(reference, context.output_dir / "downloads" / self.name)
                raw = [archive]
                extracted = archive
                if zipfile.is_zipfile(archive):
                    extracted = context.extract(self.name, archive)
                    raw.append(extracted)
                files = sorted(extracted.rglob("*.nc")) if extracted.is_dir() else [extracted]
                if not files:
                    raise FileNotFoundError(f"No NetCDF product found after extracting {archive}")
                return _locked_read(read_s5p_l2, files[0], config=config, aoi=request.aoi), raw
            return context.subset(self.name, reference, build)

        datasets = []
        for subset in context.map_products(self.name, references, product_subset):
            try:
                datasets.append(grid_s5p(subset, resolution_deg=request.s5p_resolution_deg, aoi=request.aoi))
            except ValueError as exc:
                if "no valid observations in the requested AOI" not in str(exc):
                    raise
        if not datasets:
            logger.warning("No Sentinel-5P product contains valid observations in the requested AOI; continuing without Sentinel-5P")
            return None
        return concat_time(datasets)


class LandsatAdapter:
    name = "landsat"

    def search(self, request: "AnalysisRequest", *, limit: int) -> list[ProductRef]:
        return LandsatCatalog().search(request.aoi, request.start, request.end, limit=limit)

    def acquire(self, references: list[ProductRef], context: AcquisitionContext) -> xr.Dataset | None:
        grid = context.predictor_grid(self.name)
        return concat_time(context.map_products(self.name, references, lambda reference: read_landsat_lst(reference, grid)))


class EcostressAdapter:
    name = "ecostress"

    def search(self, request: "AnalysisRequest", *, limit: int) -> list[ProductRef]:
        return EcostressCatalog().search(request.aoi, request.start, request.end, limit=limit)

    def acquire(self, references: list[ProductRef], context: AcquisitionContext) -> xr.Dataset | None:
        grid = context.predictor_grid(self.name)
        datasets = context.map_products(self.name, references, lambda reference: read_ecostress_lst(reference, grid))
        # A single overpass can split an AOI across more than one MGRS tile
        # (same acquisition time, different item) - mosaic those before
        # concatenating along time, exactly like Sentinel-2.
        return concat_time(mosaic_temporal_tiles(datasets))


SENSOR_ADAPTERS: Mapping[str, SensorAdapter] = {
    adapter.name: adapter
    for adapter in (Sentinel1Adapter(), Sentinel2Adapter(), Sentinel3Adapter(), Sentinel5PAdapter(), LandsatAdapter(), EcostressAdapter())
}
