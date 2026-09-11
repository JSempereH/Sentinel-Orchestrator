"""Sentinel-1 GRD processing and analysis-ready backscatter utilities.

The raw GRD measurement is not treated as a georeferenced predictor. The
preferred path produces Sentinel-1 NRB gamma0 with s1ard, while the explicit
SNAP backend produces terrain-corrected sigma0 or gamma0. Orbit correction,
thermal/border-noise removal, radiometric calibration and terrain geometry are
never approximated with a plain TIFF reader.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import configparser
import os
from pathlib import Path
import re
import shutil
import subprocess
import warnings
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import requests
import xarray as xr

from ..catalog import ProductRef, _product_ref
from ..config import AOI, ClientConfig
from ..download import CDSEDownloader
from ..http import http_session
from ..metadata import apply_variable_contract


SENTINEL1_COLLECTION_NAME = "SENTINEL-1"
SENTINEL1_PRODUCT_TYPE = "S1_GRD"
SENTINEL1_POLARIZATIONS = ("VV", "VH", "HH", "HV")


@dataclass(frozen=True)
class Sentinel1ProcessingConfig:
    """Parameters for a reproducible Sentinel-1 GRD to sigma0 workflow."""

    polarizations: tuple[str, ...] = ("VV", "VH")
    orbit_type: str = "Sentinel Precise (Auto Download)"
    remove_thermal_noise: bool = True
    remove_border_noise: bool = True
    terrain_correction: bool = True
    dem_name: str = "SRTM 1Sec HGT"
    target_resolution_m: int = 20
    map_projection: str = ""
    speckle_filter: str | None = None
    output_units: str = "linear"

    def __post_init__(self) -> None:
        if not self.polarizations or any(p not in SENTINEL1_POLARIZATIONS for p in self.polarizations):
            raise ValueError(f"Unsupported Sentinel-1 polarization: {self.polarizations}")
        if self.target_resolution_m <= 0:
            raise ValueError("target_resolution_m must be positive")
        if self.output_units not in {"linear", "dB"}:
            raise ValueError("output_units must be 'linear' or 'dB'")


@dataclass(frozen=True)
class S1ARDProcessingConfig:
    """Configuration for the s1ard Sentinel-1 NRB processor."""

    polarizations: tuple[str, ...] = ("VV", "VH")
    acquisition_mode: str = "IW"
    measurement: str = "gamma"
    dem_type: str = "Copernicus 30m Global DEM"
    spacing_m: int = 20
    annotations: tuple[str, ...] = ("dm", "ei", "em", "id", "lc", "li", "np", "ratio")
    clean_edges: bool = True
    clean_edges_pixels: int = 4
    gpt_args: str = ""

    def __post_init__(self) -> None:
        if any(p not in {"VV", "VH", "HH", "HV"} for p in self.polarizations):
            raise ValueError(f"Unsupported Sentinel-1 polarization: {self.polarizations}")
        if self.measurement not in {"gamma", "sigma"}:
            raise ValueError("measurement must be 'gamma' or 'sigma'")
        if self.acquisition_mode not in {"IW", "EW", "SM"}:
            raise ValueError("acquisition_mode must be IW, EW or SM")
        if self.spacing_m <= 0:
            raise ValueError("spacing_m must be positive")


@dataclass(frozen=True)
class Sentinel1RTCConfig:
    """Parameters for an ASF HyP3 on-demand RTC job.

    Mirrors `hyp3_sdk.HyP3.submit_rtc_job`'s own parameters (not every one -
    only those relevant to what `read_s1_rtc` can consume). Unlike
    `Sentinel1ProcessingConfig`/`S1ARDProcessingConfig`, this never touches
    SNAP, pyroSAR or a local DEM - ASF processes the granule entirely in
    the cloud from its name alone.
    """

    radiometry: str = "gamma0"
    resolution: int = 20
    scale: str = "power"
    speckle_filter: bool = False
    include_dem: bool = False
    include_inc_map: bool = False
    dem_matching: bool = False
    poll_interval_s: float = 60
    poll_timeout_s: float = 10800

    def __post_init__(self) -> None:
        if self.radiometry not in {"sigma0", "gamma0"}:
            raise ValueError("radiometry must be 'sigma0' or 'gamma0'")
        if self.resolution not in {10, 20, 30}:
            raise ValueError("resolution must be 10, 20 or 30 (meters)")
        if self.scale not in {"amplitude", "decibel", "power"}:
            raise ValueError("scale must be 'amplitude', 'decibel' or 'power'")
        if self.poll_interval_s <= 0 or self.poll_timeout_s <= 0:
            raise ValueError("poll_interval_s and poll_timeout_s must be positive")


@dataclass(frozen=True)
class Sentinel1ReadConfig:
    """Input/output units for a terrain-corrected sigma0 raster."""

    polarizations: tuple[str, ...] = ("VV", "VH")
    input_units: str = "linear"
    output_units: str = "linear"
    observation_time: str | np.datetime64 | None = None
    backscatter_quantity: str = "sigma0"

    def __post_init__(self) -> None:
        if any(p not in SENTINEL1_POLARIZATIONS for p in self.polarizations):
            raise ValueError(f"Unsupported Sentinel-1 polarization: {self.polarizations}")
        if self.input_units not in {"linear", "dB"} or self.output_units not in {"linear", "dB"}:
            raise ValueError("Sentinel-1 units must be 'linear' or 'dB'")
        if self.backscatter_quantity not in {"sigma0", "gamma0"}:
            raise ValueError("backscatter_quantity must be 'sigma0' or 'gamma0'")


@dataclass(frozen=True)
class Sentinel1ProcessingResult:
    """Files and command produced by the SNAP processing invocation."""

    output_path: Path
    graph_path: Path | None
    command: tuple[str, ...]
    completed: subprocess.CompletedProcess[str] | None = None
    backend: str = "snap"


def _xml_parameter(parent: ET.Element, name: str, value: Any) -> None:
    element = ET.SubElement(parent, name)
    element.text = str(value).lower() if isinstance(value, bool) else str(value)


def build_s1_graph(
    product: str | Path,
    output_path: str | Path,
    *,
    config: Sentinel1ProcessingConfig | None = None,
) -> str:
    """Build the SNAP GPT graph for an analysis-ready Sentinel-1 raster."""

    config = config or Sentinel1ProcessingConfig()
    root = ET.Element("graph", {"id": "sentinel1-grd-analysis-ready"})
    ET.SubElement(root, "version").text = "1.0"

    def node(node_id: str, operator: str, previous: str | None = None, **parameters: Any) -> str:
        element = ET.SubElement(root, "node", {"id": node_id})
        ET.SubElement(element, "operator").text = operator
        sources = ET.SubElement(element, "sources")
        if previous:
            ET.SubElement(sources, "sourceProduct").text = previous
        parameter_element = ET.SubElement(element, "parameters")
        for key, value in parameters.items():
            _xml_parameter(parameter_element, key, value)
        return node_id

    current = node("Read", "Read", file=Path(product).resolve())
    current = node("Apply-Orbit-File", "Apply-Orbit-File", current, orbitType=config.orbit_type, polyDegree=3)
    if config.remove_thermal_noise:
        current = node("Thermal-Noise-Removal", "ThermalNoiseRemoval", current, removeThermalNoise=True)
    if config.remove_border_noise:
        current = node("Remove-GRD-Border-Noise", "Remove-GRD-Border-Noise", current, borderLimit=50, trimThreshold=0.5)
    current = node(
        "Calibration",
        "Calibration",
        current,
        outputSigmaBand=True,
        outputBetaBand=False,
        outputGammaBand=False,
        selectedPolarisations=", ".join(config.polarizations),
    )
    if config.speckle_filter:
        current = node("Speckle-Filter", "Speckle-Filter", current, filter=config.speckle_filter)
    if config.terrain_correction:
        terrain_parameters: dict[str, Any] = {
            "demName": config.dem_name,
            "pixelSpacingInMeter": config.target_resolution_m,
            "imgResamplingMethod": "BILINEAR_INTERPOLATION",
            "demResamplingMethod": "BILINEAR_INTERPOLATION",
        }
        if config.map_projection:
            terrain_parameters["mapProjection"] = config.map_projection
        current = node("Terrain-Correction", "Terrain-Correction", current, **terrain_parameters)
    if config.output_units == "dB":
        current = node(
            "Linear-To-dB",
            "LinearToFromdB",
            current,
            sourceBands=", ".join(f"Sigma0_{polarization}" for polarization in config.polarizations),
        )
    node("Write", "Write", current, file=Path(output_path).resolve(), formatName="GeoTIFF")
    return ET.tostring(root, encoding="unicode")


def process_s1_grd(
    product: str | Path,
    output_dir: str | Path,
    *,
    config: Sentinel1ProcessingConfig | None = None,
    gpt: str = "gpt",
    execute: bool = True,
) -> Sentinel1ProcessingResult:
    """Run SNAP GPT or return a dry-run command for a Sentinel-1 GRD product."""

    config = config or Sentinel1ProcessingConfig()
    product = Path(product)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"\.SAFE$", "", product.stem, flags=re.IGNORECASE)
    output_path = output_dir / f"{stem}_sigma0_tc.tif"
    graph_path = output_dir / f"{stem}_sigma0_tc.xml"
    graph_path.write_text(build_s1_graph(product, output_path, config=config), encoding="utf-8")
    command = (gpt, str(graph_path), "-c", "4G")
    if execute and shutil.which(gpt) is None:
        raise RuntimeError("SNAP GPT was not found; install SNAP and expose its gpt executable in PATH")
    completed = subprocess.run(command, check=True, capture_output=True, text=True) if execute else None
    return Sentinel1ProcessingResult(output_path, graph_path, command, completed, backend="snap")


def _build_s1ard_config(
    product: Path,
    work_dir: Path,
    *,
    config: S1ARDProcessingConfig,
) -> Path:
    """Write a complete s1ard INI configuration for one GRD scene."""

    sensor_match = re.match(r"(S1[A-D])", product.name.upper())
    sensor = sensor_match.group(1) if sensor_match else "S1A"
    parser = configparser.ConfigParser()
    parser["PROCESSING"] = {
        # s1ard >=2.13 rejects 'nrb' together with a direct `scene` - it
        # raises "if argument 'scene' is set, the processing mode must be
        # 'sar'" (confirmed against a real run with s1ard 2.13.1). The NRB
        # (metadata/STAC-packaged ARD) assembly step is not reachable via
        # this single-scene path in that version; only the SAR/backscatter
        # step runs here.
        "mode": "sar",
        "scene": str(product.resolve()),
        "aoi_tiles": "",
        "aoi_geometry": "",
        "mindate": "",
        "maxdate": "",
        "date_strict": "True",
        "sensor": sensor,
        "acq_mode": config.acquisition_mode,
        "spacing_ew": str(config.spacing_m),
        "spacing_iw": str(config.spacing_m),
        "spacing_sm": str(config.spacing_m),
        "product": "GRD",
        "datatake": "",
        "processor": "snap",
        "work_dir": str(work_dir.resolve()),
        "sar_dir": "SAR",
        "tmp_dir": "TMP",
        "ard_dir": "ARD",
        "wbm_dir": "WBM",
        "logfile": "",
        # s1ard requires exactly one "scene search option" (db_file,
        # stac_catalog or parquet) to be set even when `scene` already
        # points at a single local product directly - confirmed against a
        # real run with s1ard 2.13.1, which raises "Please define a scene
        # search option" otherwise. db_file is s1ard's own SQLite scene
        # inventory - it does NOT get populated unless `scene_dir` is also
        # set (s1ard.processor.main only calls archive.insert(...) when
        # scene_dir is set), so a bare db_file path alone leaves the
        # archive empty and check_acquisition_completeness() crashes
        # trying to find even the scene itself. Pointing scene_dir at the
        # scene's own parent directory makes s1ard register it (and any
        # sibling scenes already downloaded alongside it) before `scene`
        # selects which one(s) to actually process.
        "db_file": "s1ard.db",
        "scene_dir": str(product.parent.resolve()),
        "stac_catalog": "",
        "stac_collections": "",
        "parquet": "",
        "dem_type": config.dem_type,
        "gdal_threads": "4",
        "measurement": config.measurement,
        "annotation": ", ".join(config.annotations),
    }
    parser["SNAP"] = {
        "allow_res_osv": "True",
        "clean_edges": str(config.clean_edges),
        "clean_edges_pixels": str(config.clean_edges_pixels),
        "cleanup": "True",
        "dem_resampling_method": "BILINEAR_INTERPOLATION",
        "img_resampling_method": "BILINEAR_INTERPOLATION",
        "gpt_args": config.gpt_args,
    }
    parser["METADATA"] = {
        "format": "OGC, STAC",
        "copy_original": "True",
        "access_url": "None",
        "licence": "None",
        "doi": "None",
        "processing_center": "None",
    }
    path = work_dir / "s1ard.ini"
    with path.open("w", encoding="utf-8") as handle:
        parser.write(handle)
    return path


def process_s1_ard(
    product: str | Path,
    output_dir: str | Path,
    *,
    config: S1ARDProcessingConfig | None = None,
    s1rb: str = "s1rb",
    execute: bool = True,
) -> Sentinel1ProcessingResult:
    """Run the s1ard CLI to create a Sentinel-1 NRB/gamma0 product.

    Known broken as of s1ard 2.13.1 + spatialist 0.20.1: `pyroSAR.Archive.insert()`
    (called internally by s1ard's own scene-registration step) passes an
    `ogr.DataSource` where `gdal.VectorTranslate()` expects a `gdal.Dataset` -
    two incompatible SWIG-wrapped types despite representing the same
    underlying object, a known GDAL Python-binding gotcha (see the
    gdal-dev mailing list thread on `wrapper_GDALVectorTranslateDestName`).
    Confirmed against a real run, not inferred: `TypeError: in method
    'wrapper_GDALVectorTranslateDestName', argument 2 of type
    'GDALDatasetShadow *'`. This is a bug in `spatialist`, not something
    fixable from here - use `process_s1_grd` (`backend="snap"`, the
    default) instead until upstream fixes it.
    """

    warnings.warn(
        "process_s1_ard (s1ard/pyroSAR backend) is known broken against "
        "s1ard>=2.13 + current GDAL builds - see this function's docstring. "
        "Use backend='snap' (process_s1_grd, now the default) instead.",
        stacklevel=2,
    )
    config = config or S1ARDProcessingConfig()
    product = Path(product)
    work_dir = Path(output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = _build_s1ard_config(product, work_dir, config=config)
    command = (s1rb, "process", "-c", str(config_path))
    if execute and shutil.which(s1rb) is None:
        raise RuntimeError("s1rb was not found; install sentinel-analysis[s1ard] to process Sentinel-1 NRB")
    completed = subprocess.run(command, check=True, capture_output=True, text=True) if execute else None
    ard_dir = work_dir / "ARD"
    return Sentinel1ProcessingResult(ard_dir, config_path, command, completed, backend="s1ard")


def _hyp3_client(*, hyp3_url: str | None = None):
    """Build an authenticated `hyp3_sdk.HyP3` client from the environment.

    Reuses ECOSTRESS's `EARTHDATA_BEARER_TOKEN` (both HyP3 and NASA's
    Earthdata Cloud sit behind the same `urs.earthdata.nasa.gov` login), and
    adds `EARTHDATA_USERNAME`/`EARTHDATA_PASSWORD` as a fallback for the
    token-less case `hyp3_sdk` itself supports.
    """

    try:
        import hyp3_sdk
    except ImportError as exc:
        raise RuntimeError("Install sentinel-analysis[hyp3] to submit Sentinel-1 RTC jobs to ASF HyP3") from exc

    token = os.getenv("EARTHDATA_BEARER_TOKEN")
    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")
    kwargs: dict[str, Any] = {"api_url": hyp3_url} if hyp3_url else {}
    if token:
        return hyp3_sdk.HyP3(token=token, **kwargs)
    if username and password:
        return hyp3_sdk.HyP3(username=username, password=password, **kwargs)
    raise RuntimeError(
        "hyp3_rtc needs Earthdata credentials: set EARTHDATA_BEARER_TOKEN (generate one at "
        "https://urs.earthdata.nasa.gov/profile) or EARTHDATA_USERNAME/EARTHDATA_PASSWORD."
    )


def process_s1_rtc(
    granule: str,
    output_dir: str | Path,
    *,
    config: Sentinel1RTCConfig | None = None,
    name: str | None = None,
    hyp3_url: str | None = None,
    execute: bool = True,
) -> Sentinel1ProcessingResult:
    """Submit a Sentinel-1 granule to ASF HyP3 for on-demand RTC processing,
    block until it completes, and download the result.

    Architecturally different from `process_s1_grd`/`process_s1_ard`: those
    process a GRD product already downloaded from CDSE; this one never
    downloads the raw GRD at all - HyP3 processes the named granule
    entirely in the cloud, so only its name (the CDSE/ASF SAFE product
    name, without the `.SAFE`/`.zip` suffix - `ProductRef.name` already in
    that form once stripped) is needed to submit the job.
    """

    config = config or Sentinel1RTCConfig()
    granule = re.sub(r"\.(SAFE|zip)$", "", str(granule), flags=re.IGNORECASE)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = ("hyp3_sdk.submit_rtc_job", granule)
    if not execute:
        return Sentinel1ProcessingResult(output_dir / granule, None, command, None, backend="hyp3_rtc")

    hyp3 = _hyp3_client(hyp3_url=hyp3_url)
    batch = hyp3.submit_rtc_job(
        granule,
        name=name or granule,
        radiometry=config.radiometry,
        resolution=config.resolution,
        scale=config.scale,
        speckle_filter=config.speckle_filter,
        include_dem=config.include_dem,
        include_inc_map=config.include_inc_map,
        dem_matching=config.dem_matching,
    )
    batch = hyp3.watch(batch, timeout=int(config.poll_timeout_s), interval=config.poll_interval_s)
    job = list(batch)[0]
    if not job.succeeded():
        raise RuntimeError(f"HyP3 RTC job for {granule!r} did not succeed: status={job.status_code}, logs={job.logs}")
    downloaded = job.download_files(output_dir)
    product_dir = output_dir / granule
    zips = [path for path in downloaded if path.suffix.lower() == ".zip"]
    if zips:
        product_dir = CDSEDownloader.extract(zips[0], product_dir)
    return Sentinel1ProcessingResult(product_dir, None, command, None, backend="hyp3_rtc")


def process_s1(
    product: str | Path,
    output_dir: str | Path,
    *,
    backend: str = "snap",
    config: Sentinel1ProcessingConfig | None = None,
    ard_config: S1ARDProcessingConfig | None = None,
    gpt: str = "gpt",
    s1rb: str = "s1rb",
    execute: bool = True,
) -> Sentinel1ProcessingResult:
    """Process Sentinel-1 with SNAP GPT by default, or the s1ard NRB
    processor explicitly (currently broken - see `process_s1_ard`).

    `backend="hyp3_rtc"` is not handled here - it takes a granule *name*,
    not an already-downloaded `product` path, since ASF processes it
    entirely in the cloud without a local GRD download. Call
    `process_s1_rtc()` directly (`workflow/runner.py`'s sentinel1 handling
    branches to it before any CDSE download happens).
    """

    if backend == "s1ard":
        return process_s1_ard(product, output_dir, config=ard_config, s1rb=s1rb, execute=execute)
    if backend == "snap":
        return process_s1_grd(product, output_dir, config=config, gpt=gpt, execute=execute)
    raise ValueError("backend must be 's1ard' or 'snap'")


def _require_rasterio():
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Install sentinel-analysis[sar] to read Sentinel-1 GeoTIFF data") from exc
    return rasterio


def _band_polarization(description: str | None, fallback: str) -> str:
    match = re.search(r"(?:sigma.?0|gamma.?0|beta.?0)[_ -]?(VV|VH|HH|HV)", description or "", re.IGNORECASE)
    return match.group(1).upper() if match else fallback


def _observation_time(path: Path, configured: str | np.datetime64 | None) -> np.datetime64:
    if configured is not None:
        return np.datetime64(configured, "ns")
    timestamp_pattern = r"20\d{6}T\d{6}"
    match = re.search(timestamp_pattern, path.name, re.IGNORECASE)
    if match is None and path.is_dir():
        match = next((candidate for child in path.rglob("*") if (candidate := re.search(timestamp_pattern, child.name, re.IGNORECASE))), None)
    if not match:
        raise ValueError("Provide observation_time when the Sentinel-1 filename has no acquisition timestamp")
    value = match.group(0)
    formatted = f"{value[:4]}-{value[4:6]}-{value[6:8]}T{value[9:11]}:{value[11:13]}:{value[13:15]}"
    return np.datetime64(formatted, "ns")


def read_s1_grd(
    path: str | Path,
    *,
    config: Sentinel1ReadConfig | None = None,
) -> xr.Dataset:
    """Read a SNAP terrain-corrected sigma0 GeoTIFF into the common cube form.

    This function intentionally rejects SAFE/ZIP inputs. Reading raw GRD DN
    values as geospatial backscatter would skip orbit, calibration and terrain
    correction, so callers must run :func:`process_s1_grd` first.
    """

    path = Path(path)
    if path.is_dir() or path.suffix.lower() in {".zip", ".safe"}:
        raise ValueError("Sentinel-1 SAFE products must be processed with process_s1_grd before reading")
    config = config or Sentinel1ReadConfig()
    rasterio = _require_rasterio()
    with rasterio.open(path) as source:
        transform = source.transform
        crs = source.crs
        descriptions = source.descriptions
        variables: dict[str, tuple[tuple[str, str], np.ndarray]] = {}
        for index, fallback in enumerate(config.polarizations, start=1):
            if index > source.count:
                break
            polarization = _band_polarization(descriptions[index - 1], fallback)
            if polarization not in config.polarizations:
                continue
            array = source.read(index, masked=True).astype(np.float32).filled(np.nan)
            if config.input_units == "dB" and config.output_units == "linear":
                array = np.power(10.0, array / 10.0)
            elif config.input_units == "linear" and config.output_units == "dB":
                array = 10.0 * np.log10(np.maximum(array, np.finfo(np.float32).tiny))
            name = f"{config.backscatter_quantity}_{polarization}"
            variables[name] = (("y", "x"), array)
        if not variables:
            raise ValueError(f"No requested Sentinel-1 polarizations found in {path}")
        height, width = source.height, source.width
        x = transform.c + (np.arange(width) + 0.5) * transform.a
        y = transform.f + (np.arange(height) + 0.5) * transform.e
    dataset = xr.Dataset(variables, coords={"x": x, "y": y}).expand_dims(
        time=[_observation_time(path, config.observation_time)]
    )
    dataset.attrs.update({
        "source": str(path),
        "sensor": "Sentinel-1 SAR",
        "product_type": SENTINEL1_PRODUCT_TYPE,
        "processing_level": "GRD terrain corrected",
        "backscatter_quantity": config.backscatter_quantity,
        "units": config.output_units,
        "crs": crs.to_string() if crs else "unknown",
    })
    valid = xr.ones_like(next(iter(dataset.data_vars.values())), dtype=bool)
    for name, value in dataset.data_vars.items():
        valid &= np.isfinite(value)
        if name.startswith(f"{config.backscatter_quantity}_"):
            value.attrs.update({"quantity": config.backscatter_quantity, "units": config.output_units})
    dataset["valid_mask"] = valid.rename("valid_mask")
    return apply_variable_contract(dataset, sensor="Sentinel-1", product=SENTINEL1_PRODUCT_TYPE, source=str(path))


def read_s1_ard(
    path: str | Path,
    *,
    config: Sentinel1ReadConfig | None = None,
) -> xr.Dataset:
    """Read s1ard NRB measurement layers into the common cube contract."""

    root = Path(path)
    if not root.is_dir():
        raise ValueError("read_s1_ard expects an extracted s1ard ARD directory")
    config = config or Sentinel1ReadConfig(backscatter_quantity="gamma0")
    quantity_code = "g" if config.backscatter_quantity == "gamma0" else "s"
    rasterio = _require_rasterio()
    files: dict[str, Path] = {}
    for polarization in config.polarizations:
        candidates = sorted(root.rglob(f"*-{polarization.lower()}-{quantity_code}-lin.tif"))
        candidates += sorted(root.rglob(f"*-{polarization.lower()}-{quantity_code}-lin.vrt"))
        if not candidates:
            candidates = sorted(root.rglob(f"*-{polarization.lower()}-{quantity_code}-log.vrt"))
        if not candidates:
            raise FileNotFoundError(f"Could not find s1ard {config.backscatter_quantity} {polarization} measurement under {root}")
        files[polarization] = candidates[0]
    with rasterio.open(next(iter(files.values()))) as first:
        transform = first.transform
        crs = first.crs
        height, width = first.height, first.width
    variables: dict[str, tuple[tuple[str, str], np.ndarray]] = {}
    for polarization, file in files.items():
        with rasterio.open(file) as source:
            array = source.read(1, masked=True).astype(np.float32).filled(np.nan)
            description = (source.descriptions[0] or "").lower()
            is_log = "-log" in file.name.lower() or "log" in description
            if is_log and config.output_units == "linear":
                array = np.power(10.0, array / 10.0)
            elif not is_log and config.output_units == "dB":
                array = 10.0 * np.log10(np.maximum(array, np.finfo(np.float32).tiny))
            variables[f"{config.backscatter_quantity}_{polarization}"] = (("y", "x"), array)
    data_masks = sorted(root.rglob("*-dm.tif"))
    if data_masks:
        with rasterio.open(data_masks[0]) as source:
            variables["s1ard_data_mask"] = (("y", "x"), source.read(1, masked=True).filled(0).astype(np.uint8))
    x = transform.c + (np.arange(width) + 0.5) * transform.a
    y = transform.f + (np.arange(height) + 0.5) * transform.e
    dataset = xr.Dataset(variables, coords={"x": x, "y": y}).expand_dims(
        time=[_observation_time(root, config.observation_time)]
    )
    dataset.attrs.update({
        "source": str(root),
        "sensor": "Sentinel-1 SAR",
        "product_type": "S1_NRB",
        "processing_level": "ARD NRB",
        "backscatter_quantity": config.backscatter_quantity,
        "units": config.output_units,
        "crs": crs.to_string() if crs else "unknown",
        "layover_shadow_annotation": bool(list(root.rglob("*-dm.tif"))),
    })
    if "s1ard_data_mask" in dataset:
        dataset["s1ard_data_mask"].attrs.update({
            "long_name": "s1ard data mask bit field",
            "source": "s1ard annotation dm",
            "interpretation": "consult the s1ard product metadata before filtering bits",
        })
    valid = xr.ones_like(next(iter(dataset.data_vars.values())), dtype=bool)
    for name, value in dataset.data_vars.items():
        valid &= np.isfinite(value)
        if name.startswith(f"{config.backscatter_quantity}_"):
            value.attrs.update({"quantity": config.backscatter_quantity, "units": config.output_units})
    dataset["valid_mask"] = valid.rename("valid_mask")
    return apply_variable_contract(dataset, sensor="Sentinel-1", product="S1_NRB", source=str(root))


def read_s1_rtc(
    path: str | Path,
    *,
    config: Sentinel1ReadConfig | None = None,
    rtc_scale: str = "power",
) -> xr.Dataset:
    """Read an ASF HyP3 RTC product's polarization GeoTIFFs into the common
    cube form.

    `rtc_scale` must match whatever `scale` the RTC job was submitted with
    (`Sentinel1RTCConfig.scale`) - unlike SNAP/s1ard output, a HyP3 product's
    GeoTIFFs carry no unit metadata this reader can use to detect it.
    """

    root = Path(path)
    if not root.is_dir():
        raise ValueError("read_s1_rtc expects an extracted HyP3 RTC product directory")
    if rtc_scale not in {"power", "decibel", "amplitude"}:
        raise ValueError("rtc_scale must be 'power', 'decibel' or 'amplitude'")
    config = config or Sentinel1ReadConfig(backscatter_quantity="gamma0")
    rasterio = _require_rasterio()
    files: dict[str, Path] = {}
    for polarization in config.polarizations:
        candidates = sorted(root.rglob(f"*_{polarization}.tif"))
        if candidates:
            files[polarization] = candidates[0]
    if not files:
        raise FileNotFoundError(f"No HyP3 RTC polarization GeoTIFFs (e.g. *_VV.tif) found under {root}")
    with rasterio.open(next(iter(files.values()))) as first:
        transform = first.transform
        crs = first.crs
        height, width = first.height, first.width
    variables: dict[str, tuple[tuple[str, str], np.ndarray]] = {}
    for polarization, file in files.items():
        with rasterio.open(file) as source:
            array = source.read(1, masked=True).astype(np.float32).filled(np.nan)
        if rtc_scale == "amplitude":
            array = np.power(array, 2.0)
            rtc_scale_linear = "power"
        else:
            rtc_scale_linear = rtc_scale
        if rtc_scale_linear == "decibel" and config.output_units == "linear":
            array = np.power(10.0, array / 10.0)
        elif rtc_scale_linear == "power" and config.output_units == "dB":
            array = 10.0 * np.log10(np.maximum(array, np.finfo(np.float32).tiny))
        variables[f"{config.backscatter_quantity}_{polarization}"] = (("y", "x"), array)
    layover_shadow = sorted(root.rglob("*_ls_map.tif"))
    if layover_shadow:
        with rasterio.open(layover_shadow[0]) as source:
            variables["hyp3_layover_shadow_mask"] = (("y", "x"), source.read(1, masked=True).filled(0).astype(np.uint8))
    x = transform.c + (np.arange(width) + 0.5) * transform.a
    y = transform.f + (np.arange(height) + 0.5) * transform.e
    dataset = xr.Dataset(variables, coords={"x": x, "y": y}).expand_dims(
        time=[_observation_time(root, config.observation_time)]
    )
    dataset.attrs.update({
        "source": str(root),
        "sensor": "Sentinel-1 SAR",
        "product_type": "S1_RTC",
        "processing_level": "ASF HyP3 RTC",
        "backscatter_quantity": config.backscatter_quantity,
        "units": config.output_units,
        "crs": crs.to_string() if crs else "unknown",
    })
    valid = xr.ones_like(next(iter(dataset.data_vars.values())), dtype=bool)
    for name, value in dataset.data_vars.items():
        valid &= np.isfinite(value)
        if name.startswith(f"{config.backscatter_quantity}_"):
            value.attrs.update({"quantity": config.backscatter_quantity, "units": config.output_units})
    dataset["valid_mask"] = valid.rename("valid_mask")
    return apply_variable_contract(dataset, sensor="Sentinel-1", product="S1_RTC", source=str(root))


def _linear_backscatter(value: xr.DataArray) -> xr.DataArray:
    units = value.attrs.get("units", "linear")
    return xr.apply_ufunc(np.power, 10.0, value / 10.0) if units == "dB" else value


def sentinel1_indices(dataset: xr.Dataset) -> xr.Dataset:
    """Compute dual-polarization SAR predictors from sigma0 VV/VH."""

    prefix = "gamma0" if {"gamma0_VV", "gamma0_VH"}.issubset(dataset.data_vars) else "sigma0"
    required = {f"{prefix}_VV", f"{prefix}_VH"}
    missing = sorted(required.difference(dataset.data_vars))
    if missing:
        raise KeyError(f"Missing Sentinel-1 bands for indices: {missing}")
    vv = _linear_backscatter(dataset[f"{prefix}_VV"])
    vh = _linear_backscatter(dataset[f"{prefix}_VH"])
    result = dataset.copy()
    result["VH_VV_ratio_dB"] = 10.0 * xr.apply_ufunc(np.log10, vh / vv.where(vv > 0))
    result["RVI"] = 4.0 * vh / (vv + vh).where((vv + vh) > 0)
    result["radar_span"] = vv + vh
    result["VH_VV_ratio_dB"].attrs.update({"long_name": "VH to VV backscatter ratio", "units": "dB"})
    result["RVI"].attrs.update({"long_name": "dual-polarization radar vegetation index", "units": "1"})
    result["radar_span"].attrs.update({"long_name": f"VV plus VH {prefix}", "units": "linear"})
    return result


class Sentinel1Catalog:
    """Search Sentinel-1 IW GRD acquisitions through the CDSE OData catalogue.

    Not STAC-based (unlike Landsat/ECOSTRESS here): CDSE's STAC endpoint
    returns Sentinel-1 GRD in a "COG" packaging by default that neither
    `pyroSAR.identify()` nor SNAP GPT's `Read` operator accept, and its
    `STACItem` has no `ProductRef`-shaped fields (`online`, `cloud_cover`,
    `start_datetime`) that `select_product_refs()` requires - passing STAC
    items through the normal discovery/download path used to crash with
    `AttributeError: 'STACItem' object has no attribute 'online'` before
    this rewrite. CDSE's OData catalogue (the same one `Sentinel2Catalog`
    already uses) still serves the classic non-COG SAFE for every
    acquisition; `not contains(Name,'COG')` excludes the COG variant,
    confirmed against a real query returning real `S1[C|D]_IW_GRDH_1SDV_...
    .SAFE` products.
    """

    def __init__(self, config: ClientConfig | None = None, *, session: requests.Session | None = None):
        self.config = config
        self.session = session or http_session()
        self.catalog_url = config.catalog_url if config else "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

    def search(
        self,
        aoi: AOI,
        start: str | date | datetime,
        end: str | date | datetime,
        *,
        acquisition_mode: str = "IW",
        limit: int = 100,
    ) -> list[ProductRef]:
        """Return online Sentinel-1 IW GRD products intersecting an AOI."""

        def iso(value: str | date | datetime, *, end_of_day: bool = False) -> str:
            date_only = False
            if isinstance(value, datetime):
                parsed = value
            elif isinstance(value, date):
                parsed = datetime.combine(value, datetime.min.time())
                date_only = True
            else:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                date_only = "T" not in value and " " not in value
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            if end_of_day and date_only:
                parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
            return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

        start_iso, end_iso = iso(start), iso(end, end_of_day=True)
        product_type = f"{acquisition_mode}_GRDH_1S"
        filters = [
            f"Collection/Name eq '{SENTINEL1_COLLECTION_NAME}'",
            "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq "
            f"'productType' and att/OData.CSC.StringAttribute/Value eq '{product_type}')",
            f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi.as_wkt()}')",
            f"ContentDate/Start lt {end_iso}",
            f"ContentDate/End gt {start_iso}",
            "Online eq true",
            "not contains(Name,'COG')",
        ]
        products: list[ProductRef] = []
        url: str | None = self.catalog_url
        params: dict[str, Any] | None = {
            "$filter": " and ".join(filters),
            "$expand": "Attributes",
            "$orderby": "ContentDate/Start asc",
            "$top": min(limit, 1000),
        }
        download_url = self.config.download_url if self.config else "https://download.dataspace.copernicus.eu/odata/v1/Products"
        while url and len(products) < limit:
            response = self.session.get(url, params=params, timeout=120)
            response.raise_for_status()
            payload = response.json()
            for item in payload.get("value", []):
                products.append(_product_ref(item, download_url, default_product_type=SENTINEL1_PRODUCT_TYPE))
                if len(products) >= limit:
                    break
            url = payload.get("@odata.nextLink")
            params = None
        return products
