"""Reader for Sentinel-3 SLSTR L2 LST SAFE products.

The product contains several NetCDF files whose exact names evolve with
processing baselines. This reader discovers variables from names and CF
metadata instead of depending on one fixed filename layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import tempfile
from typing import Iterable, Mapping
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import xarray as xr

from .quality import cf_flag_mask_array


class ProductFormatError(ValueError):
    """Raised when a local product does not contain a recognizable LST field."""


@dataclass(frozen=True)
class ProductMetadata:
    """Small, stable subset of SAFE metadata exposed by the reader."""

    product_name: str
    baseline: str | None
    start_datetime: str | None
    end_datetime: str | None
    manifest: Path | None


def _normal(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _data_variables(dataset: xr.Dataset) -> Iterable[tuple[str, xr.DataArray]]:
    for name, variable in dataset.data_vars.items():
        yield name, variable


def _score(name: str, variable: xr.DataArray, terms: tuple[str, ...]) -> int:
    text = " ".join(
        [name, str(variable.attrs.get("standard_name", "")), str(variable.attrs.get("long_name", ""))]
    )
    normalized = _normal(text)
    score = 0
    for index, term in enumerate(terms):
        if _normal(term) in normalized:
            score += 100 - index
    return score


def _best(dataset: xr.Dataset, terms: tuple[str, ...]) -> tuple[str, xr.DataArray] | None:
    candidates = [(score, name, variable) for name, variable in _data_variables(dataset)
                  if (score := _score(name, variable, terms)) > 0]
    if not candidates:
        return None
    best = max(candidates, key=lambda item: item[0])
    return best[1], best[2]


def _find_manifest(root: Path) -> Path | None:
    manifests = list(root.rglob("manifest.safe"))
    if not manifests:
        manifests = list(root.rglob("xfdumanifest.xml"))
    return manifests[0] if manifests else None


def _metadata(root: Path) -> ProductMetadata:
    manifest = _find_manifest(root)
    name = root.name
    baseline_match = re.search(r"_(?:NT|NR)_([0-9]{3})\.SEN3$", name)
    baseline = baseline_match.group(1) if baseline_match else None
    start_datetime = None
    end_datetime = None
    if manifest:
        try:
            tree = ET.parse(manifest)
            text = " ".join(tree.getroot().itertext())
            datetimes = re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", text)
            if datetimes:
                start_datetime = datetimes[0]
            if len(datetimes) > 1:
                end_datetime = datetimes[1]
        except ET.ParseError:
            pass
    if start_datetime is None:
        match = re.search(r"_(\d{8}T\d{6})_(\d{8}T\d{6})_", name)
        if match:
            start_datetime = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(
                tzinfo=timezone.utc
            ).isoformat().replace("+00:00", "Z")
            end_datetime = datetime.strptime(match.group(2), "%Y%m%dT%H%M%S").replace(
                tzinfo=timezone.utc
            ).isoformat().replace("+00:00", "Z")
    return ProductMetadata(name, baseline, start_datetime, end_datetime, manifest)


def _product_root(path: str | Path, extract_dir: str | Path | None = None) -> Path:
    path = Path(path)
    if path.is_dir():
        return path
    if path.suffix.lower() != ".zip":
        raise FileNotFoundError(f"Expected a SAFE directory or ZIP archive: {path}")
    destination = Path(extract_dir) if extract_dir else Path(tempfile.mkdtemp(prefix="sentinel-analysis-sentinel3-"))
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(path) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ProductFormatError(f"Unsafe archive member: {member.filename}")
        zipped.extractall(destination)
    products = list(destination.glob("*.SEN3"))
    return products[0] if len(products) == 1 else destination


def _open_files(root: Path, chunks: Mapping[str, int] | None) -> list[tuple[Path, xr.Dataset]]:
    opened: list[tuple[Path, xr.Dataset]] = []
    for path in sorted(root.rglob("*.nc")):
        try:
            dataset = xr.open_dataset(path, decode_cf=True, mask_and_scale=True, chunks=chunks)
        except Exception:
            continue
        if dataset.data_vars:
            opened.append((path, dataset))
        else:
            dataset.close()
    return opened


def _standard_dataset(root: Path, opened: list[tuple[Path, xr.Dataset]], metadata: ProductMetadata) -> xr.Dataset:
    selected: dict[str, xr.DataArray] = {}
    selected_paths: set[Path] = set()
    searches = {
        "lst": ("LST", "land surface temperature", "surface temperature"),
        "lst_uncertainty": ("uncertainty", "total uncertainty", "lst uncertainty"),
        "exception_flags": ("exception", "lst exception", "quality flags", "flags", "qf"),
        "cloud_flags": ("cloud", "bayes"),
        "latitude": ("latitude", "lat"),
        "longitude": ("longitude", "lon"),
        "ndvi": ("ndvi", "normalized difference vegetation"),
        "biome": ("biome", "surface classification", "globcover"),
        "fractional_vegetation_cover": ("fractional vegetation cover", "fvc"),
        "total_column_water_vapour": ("total column water vapour", "tcwv"),
    }
    for standard_name, terms in searches.items():
        candidates = []
        for path, dataset in opened:
            found = _best(dataset, terms)
            if found:
                name, variable = found
                candidates.append((_score(name, variable, terms), path, variable))
        if candidates:
            _, path, variable = max(candidates, key=lambda item: item[0])
            selected[standard_name] = variable.rename(standard_name)
            selected[standard_name].attrs["source_file"] = path.name
            selected_paths.add(path)

    if "lst" not in selected:
        available = [f"{path.name}: {list(dataset.data_vars)}" for path, dataset in opened]
        raise ProductFormatError("Could not identify an LST variable. Available variables: " + "; ".join(available))

    if "cloud_flags" in selected:
        if selected["cloud_flags"].dtype.kind == "b":
            selected["cloud_mask"] = selected["cloud_flags"].astype(bool).rename("cloud_mask")
        else:
            selected["cloud_mask"] = cf_flag_mask_array(
                selected["cloud_flags"],
                meanings=("gross_cloud", "thin_cirrus", "medium_high", "fog", "stratus"),
            ).rename("cloud_mask")

    dataset = xr.Dataset(selected)
    if metadata.start_datetime and "time" not in dataset.dims:
        dataset = dataset.expand_dims(
            time=[np.datetime64(metadata.start_datetime.replace("Z", ""), "ns")]
        )
    elif "t" in dataset.dims and "time" not in dataset.dims:
        dataset = dataset.rename_dims({"t": "time"})
    for old, new in (("rows", "y"), ("columns", "x")):
        if old in dataset.dims and new not in dataset.dims:
            dataset = dataset.rename_dims({old: new})
    dataset.attrs.update({
        "product_name": metadata.product_name,
        "baseline": metadata.baseline or "unknown",
        "product_level": "L2",
        "product_type": "SL_2_LST",
        "source": "Copernicus Sentinel-3 SLSTR",
    })
    if "units" not in dataset["lst"].attrs:
        dataset["lst"].attrs.update({"units": "K", "unit_assumption": "missing product units assumed Kelvin"})
    retained = tuple(dataset_ for path, dataset_ in opened if path in selected_paths)
    for path, opened_dataset in opened:
        if path not in selected_paths:
            opened_dataset.close()
    def close_opened() -> None:
        for opened_dataset in retained:
            opened_dataset.close()

    dataset.set_close(close_opened)
    return dataset


def read_l2_lst(
    path: str | Path,
    *,
    chunks: Mapping[str, int] | None = None,
    extract_dir: str | Path | None = None,
) -> xr.Dataset:
    """Read a SAFE directory or ZIP into a standardized lazy xarray Dataset."""

    root = _product_root(path, extract_dir)
    metadata = _metadata(root)
    opened = _open_files(root, chunks)
    if not opened:
        raise ProductFormatError(f"No readable NetCDF files found under {root}")
    return _standard_dataset(root, opened, metadata)


def inspect_l2_lst(path: str | Path) -> dict[str, object]:
    """Inspect a product without requiring a known baseline-specific layout."""

    root = _product_root(path)
    opened = _open_files(root, None)
    metadata = _metadata(root)
    try:
        return {
            "metadata": metadata,
            "files": {str(file): list(dataset.data_vars) for file, dataset in opened},
        }
    finally:
        for _, dataset in opened:
            dataset.close()
