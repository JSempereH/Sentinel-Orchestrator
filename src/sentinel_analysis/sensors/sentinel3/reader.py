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
import warnings
from typing import Iterable, Mapping
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import xarray as xr

from .quality import cf_flag_mask_array


# The files an LST analysis needs: LST/uncertainty/exceptions, pixel
# geolocation, quality and cloud flags, viewing geometry, and the manifest
# (acquisition times). Everything else in an SL_2_LST product - notably the
# ~38 MB met_tx.nc of a ~70 MB archive - is never read.
SLSTR_LST_FILES = ("LST_in.nc", "geodetic_in.nc", "flags_in.nc", "geometry_tn.nc", "xfdumanifest.xml")
# The two files that tell how much of an AOI a product sees clear of cloud
# (~10 MB of the ~18 MB above): fetched first when probing products.
# Geolocation, cloud mask and view geometry (~14 MB): enough to tell how much
# of the AOI a product would leave usable, before its LST is downloaded.
SLSTR_PROBE_FILES = ("geodetic_in.nc", "flags_in.nc", "geometry_tn.nc")
# Same cut as the default QualityPolicy.max_view_zenith.
PROBE_MAX_VIEW_ZENITH = 45.0


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


def _extract_members(archive: Path, destination: Path, wanted: Iterable[str] = SLSTR_LST_FILES) -> Path:
    """Safely extract only the known LST files (or every NetCDF if the
    archive uses an unknown layout) and return the product root."""

    wanted = set(wanted)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        members = [member for member in zipped.infolist() if not member.is_dir()]
        selected = [member for member in members if Path(member.filename).name in wanted]
        if not any(Path(member.filename).name.endswith(".nc") for member in selected):
            selected = [member for member in members if member.filename.endswith((".nc", ".xml", ".safe"))]
        for member in selected:
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ProductFormatError(f"Unsafe archive member: {member.filename}")
            zipped.extract(member, destination)
    products = list(destination.glob("*.SEN3"))
    return products[0] if len(products) == 1 else destination


def _open_files(root: Path, chunks: Mapping[str, int] | None) -> list[tuple[Path, xr.Dataset]]:
    opened: list[tuple[Path, xr.Dataset]] = []
    known = [path for name in SLSTR_LST_FILES if name.endswith(".nc") for path in root.rglob(name)]
    for path in sorted(known) if known else sorted(root.rglob("*.nc")):
        try:
            dataset = xr.open_dataset(path, decode_cf=True, mask_and_scale=True, chunks=chunks)
        except Exception:
            continue
        if dataset.data_vars:
            opened.append((path, dataset))
        else:
            dataset.close()
    return opened


def _across_track_resolution(value) -> float:
    """First element of a ``resolution`` attribute. Real SLSTR files store it
    as the string ``'[ 16000 1000 ]'``, not as a numeric array."""

    if isinstance(value, str):
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", value)
        if not numbers:
            raise ValueError(f"unparseable resolution attribute {value!r}")
        return float(numbers[0])
    return float(np.atleast_1d(value)[0])


def _sat_zenith_on_image_grid(opened: list[tuple[Path, xr.Dataset]], selected: Mapping[str, xr.DataArray]) -> xr.DataArray | None:
    """Interpolate the tie-point ``sat_zenith_tn`` onto the 1 km image grid.

    Tie points share the image rows but sample columns every 16 km. Each
    file's ``track_offset`` (in its own column units) and ``resolution``
    attributes place both grids on one across-track axis, so image column
    ``i`` sits at tie-point column ``tie_offset + (i - image_offset) *
    image_res / tie_res``.
    """

    geometry = next((dataset for path, dataset in opened if "sat_zenith_tn" in dataset.data_vars), None)
    image = next((dataset for path, dataset in opened if "latitude_in" in dataset.data_vars), None)
    if geometry is None or image is None or "lst" not in selected:
        return None
    try:
        tie_offset = float(geometry.attrs["track_offset"])
        image_offset = float(image.attrs["track_offset"])
        tie_resolution = _across_track_resolution(geometry.attrs["resolution"])
        image_resolution = _across_track_resolution(image.attrs["resolution"])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        warnings.warn(f"Could not place SLSTR tie points on the image grid ({exc}); sat_zenith is unavailable", stacklevel=2)
        return None
    tie = geometry["sat_zenith_tn"].values
    n_columns = selected["lst"].sizes.get("columns", selected["lst"].shape[-1])
    if tie.shape[0] != selected["lst"].shape[0]:
        return None
    position = tie_offset + (np.arange(n_columns) - image_offset) * image_resolution / tie_resolution
    left = np.clip(np.floor(position).astype(int), 0, tie.shape[1] - 1)
    right = np.clip(left + 1, 0, tie.shape[1] - 1)
    fraction = np.clip(position - left, 0.0, 1.0)
    values = tie[:, left] * (1 - fraction) + tie[:, right] * fraction
    values[:, (position < 0) | (position > tie.shape[1] - 1)] = np.nan
    return xr.DataArray(values.astype(np.float32), dims=selected["lst"].dims[-2:], name="sat_zenith", attrs={"units": "degree", "long_name": "satellite view zenith angle"})


def _crop_to_aoi(dataset: xr.Dataset, aoi, margin_deg: float) -> xr.Dataset:
    """Keep the smallest row/column window covering ``aoi`` plus a margin."""

    if aoi is None or "latitude" not in dataset or "longitude" not in dataset:
        return dataset
    latitude = dataset["latitude"].squeeze(drop=True).values
    longitude = dataset["longitude"].squeeze(drop=True).values
    inside = (
        (latitude >= aoi.south - margin_deg) & (latitude <= aoi.north + margin_deg)
        & (longitude >= aoi.west - margin_deg) & (longitude <= aoi.east + margin_deg)
    )
    if not inside.any():
        return dataset.isel(y=slice(0, 0), x=slice(0, 0))
    rows, cols = np.nonzero(inside)
    return dataset.isel(y=slice(rows.min(), rows.max() + 1), x=slice(cols.min(), cols.max() + 1))


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

    # Prefer the exact SLSTR flag variables over fuzzy matching: flags_in.nc
    # carries both ``cloud_in`` (individual threshold tests) and ``bayes_in``
    # (the Bayesian cloud mask), whose names/long names overlap.
    for path, dataset in opened:
        for exact, standard_name in (("cloud_in", "cloud_flags"), ("bayes_in", "bayes_flags"), ("confidence_in", "confidence_flags")):
            if exact in dataset.data_vars:
                selected[standard_name] = dataset[exact].rename(standard_name)
                selected[standard_name].attrs["source_file"] = path.name
                selected_paths.add(path)

    sat_zenith = _sat_zenith_on_image_grid(opened, selected)
    if sat_zenith is not None:
        selected["sat_zenith"] = sat_zenith

    if "lst" not in selected:
        available = [f"{path.name}: {list(dataset.data_vars)}" for path, dataset in opened]
        raise ProductFormatError("Could not identify an LST variable. Available variables: " + "; ".join(available))

    if "bayes_flags" in selected:
        # ESA's recommended cloud screening for SLSTR LST. On real Berlin
        # scenes the ``cloud_in`` subset below let most clouds through
        # (cloud-flagged pixels ~10-15 degC colder than the "clear" ones were
        # kept), while ``single_moderate`` separated them cleanly.
        selected["cloud_mask"] = cf_flag_mask_array(selected["bayes_flags"], meanings=("single_moderate",)).rename("cloud_mask")
        selected["cloud_mask"].attrs["cloud_mask_source"] = "bayes_in:single_moderate"
    elif "cloud_flags" in selected:
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
    aoi=None,
    aoi_margin_deg: float = 0.1,
) -> xr.Dataset:
    """Read a SAFE directory or ZIP into a standardized xarray Dataset.

    A ZIP is no longer extracted whole into a temporary directory that is
    never removed (an earlier version leaked one ~100 MB directory per read
    into /tmp): only the files in ``SLSTR_LST_FILES`` are extracted into a
    temporary directory, the data are loaded into memory and the directory
    is deleted before returning. Pass ``extract_dir`` to keep an extraction,
    or a directory to read lazily in place.

    With ``aoi``, the swath is cropped to the rows/columns covering the AOI
    plus ``aoi_margin_deg`` - a city needs a few thousand of a swath's
    ~1.8 million pixels.
    """

    path = Path(path)
    if path.is_dir() or extract_dir is not None:
        full = _read_root(_product_root(path, extract_dir), chunks)
        if aoi is None:
            return full  # lazy; closing it closes the underlying files
        return _load_and_close(full, aoi, aoi_margin_deg)
    if path.suffix.lower() != ".zip":
        raise FileNotFoundError(f"Expected a SAFE directory or ZIP archive: {path}")
    with tempfile.TemporaryDirectory(prefix="sentinel-analysis-sentinel3-") as temporary:
        return _load_and_close(_read_root(_extract_members(path, Path(temporary)), None), aoi, aoi_margin_deg)


def _load_and_close(full: xr.Dataset, aoi, aoi_margin_deg: float) -> xr.Dataset:
    """Load the (cropped) data into memory and close the *full* dataset's
    files. Closing only the cropped view leaves every opened NetCDF - and
    the arrays already read from it - alive, ~80 MB per product."""

    try:
        # deep copy: ``isel`` of in-memory arrays returns NumPy *views*, and
        # a 50 x 48 view keeps the whole 1200 x 1500 swath array alive.
        return _crop_to_aoi(full, aoi, aoi_margin_deg).load().copy(deep=True)
    finally:
        full.close()


def _read_root(root: Path, chunks: Mapping[str, int] | None) -> xr.Dataset:
    metadata = _metadata(root)
    opened = _open_files(root, chunks)
    if not opened:
        raise ProductFormatError(f"No readable NetCDF files found under {root}")
    return _standard_dataset(root, opened, metadata)


def inspect_l2_lst(path: str | Path) -> dict[str, object]:
    """Inspect a product without requiring a known baseline-specific layout."""

    path = Path(path)
    with tempfile.TemporaryDirectory(prefix="sentinel-analysis-sentinel3-") as temporary:
        root = path if path.is_dir() else _extract_members(path, Path(temporary))
        opened = _open_files(root, None)
        metadata = _metadata(root)
        try:
            return {
                "metadata": metadata,
                "files": {str(file.relative_to(root)): list(dataset.data_vars) for file, dataset in opened},
            }
        finally:
            for _, dataset in opened:
                dataset.close()


def aoi_clear_fraction(root: str | Path, aoi, *, max_view_zenith: float | None = PROBE_MAX_VIEW_ZENITH) -> float | None:
    """Share of a product's pixels inside ``aoi`` that would survive the
    default quality screening: clear for ESA's Bayesian cloud mask
    (``bayes_in`` ``single_moderate``, as the reader uses) and seen at no
    more than ``max_view_zenith`` degrees.

    Needs only ``SLSTR_PROBE_FILES``, so a product can be judged before its
    LST is downloaded. Without ``geometry_tn.nc`` the view angle is not
    checked. Returns ``None`` when no pixel falls inside the AOI. Falls back
    to the ``cloud_in`` test subset when a product has no Bayesian mask.
    """

    root = Path(root)
    with xr.open_dataset(root / "geodetic_in.nc") as geodetic, xr.open_dataset(root / "flags_in.nc") as flags:
        if "bayes_in" in flags:
            cloud = np.asarray(cf_flag_mask_array(flags["bayes_in"], meanings=("single_moderate",)).values)
        else:
            cloud = np.asarray(cf_flag_mask_array(flags["cloud_in"], meanings=("gross_cloud", "thin_cirrus", "medium_high", "fog", "stratus")).values)
        unusable = cloud.astype(bool)
        geometry_path = root / "geometry_tn.nc"
        if max_view_zenith is not None and geometry_path.exists():
            with xr.open_dataset(geometry_path) as geometry:
                zenith = _sat_zenith_on_image_grid([(geometry_path, geometry), (root / "geodetic_in.nc", geodetic)], {"lst": geodetic["latitude_in"]})
            if zenith is not None:
                unusable |= ~(np.asarray(zenith.values) <= max_view_zenith)
        return _clear_fraction(geodetic["latitude_in"].values, geodetic["longitude_in"].values, unusable, aoi)


def subset_clear_fraction(path: str | Path, aoi, *, max_view_zenith: float | None = PROBE_MAX_VIEW_ZENITH) -> float | None:
    """``aoi_clear_fraction`` of a stored AOI subset (same cloud test, same view-angle cut)."""

    with xr.open_dataset(path) as subset:
        if not {"latitude", "longitude", "cloud_mask"} <= set(subset.variables):
            return None
        unusable = np.asarray(subset["cloud_mask"].values).astype(bool)
        if max_view_zenith is not None and "sat_zenith" in subset:
            unusable |= ~(np.asarray(subset["sat_zenith"].values) <= max_view_zenith)
        return _clear_fraction(subset["latitude"].values, subset["longitude"].values, unusable, aoi)


def _clear_fraction(latitude, longitude, cloud, aoi) -> float | None:
    """Clear share of the pixels inside ``aoi`` (its polygon when it has one); ``None`` if none is inside."""

    latitude, longitude = np.asarray(latitude, dtype=float), np.asarray(longitude, dtype=float)
    cloud = np.asarray(cloud).astype(bool)
    inside = (
        np.isfinite(latitude) & np.isfinite(longitude)
        & (latitude >= aoi.south) & (latitude <= aoi.north) & (longitude >= aoi.west) & (longitude <= aoi.east)
    )
    if inside.any() and aoi.has_geometry:
        import shapely

        inside[inside] = shapely.contains_xy(aoi.shape(), longitude[inside], latitude[inside])
    if not inside.any():
        return None
    return float(1.0 - cloud[inside].mean())
