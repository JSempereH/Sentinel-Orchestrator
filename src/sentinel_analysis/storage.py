"""Optional cloud-native persistence helpers for analysis cubes."""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any, Hashable, Literal

import numpy as np
import xarray as xr


# The HDF5 library underneath netCDF4 keeps global state that is not thread
# safe, even across different files: concurrent reads and writes from the
# product threads of AnalysisWorkflow.execute(max_workers>1) segfaulted the
# process (reproduced 3 of 3 runs with 4 threads; 0 of 3 with this lock).
# Every NetCDF read or write the library does from worker threads holds it.
NETCDF_LOCK = threading.RLock()


def netcdf_safe(dataset: xr.Dataset) -> xr.Dataset:
    """A copy whose attributes NetCDF can store.

    NetCDF attributes cannot be booleans, ``None`` or nested values, which
    the analysis cubes use for provenance (``pixel_footprints_reconstructed``,
    skipped-scene lists, ...): booleans become 0/1, numeric lists arrays,
    ``None`` is dropped and anything else is stored as its string form.
    """

    def clean(attrs: dict) -> dict:
        result: dict[str, Any] = {}
        for key, value in attrs.items():
            if value is None:
                continue
            if isinstance(value, (bool, np.bool_)):
                result[key] = int(value)
            elif isinstance(value, (str, int, float, np.integer, np.floating)):
                result[key] = value
            elif isinstance(value, np.ndarray) and value.dtype != bool:
                result[key] = value
            elif isinstance(value, (list, tuple, np.ndarray)) and all(isinstance(item, (int, float, np.integer, np.floating, np.bool_)) for item in value):
                result[key] = np.asarray(value, dtype=float if any(isinstance(item, (float, np.floating)) for item in value) else int)
            else:
                result[key] = str(value)
        return result

    safe = dataset.copy()
    safe.attrs = clean(dict(safe.attrs))
    for name in list(safe.variables):
        safe[name].attrs = clean(dict(safe[name].attrs))
        safe[name].encoding = {}
    return safe


def write_netcdf(dataset: xr.Dataset, path: str | Path) -> Path:
    """Write a cube to NetCDF, coercing attributes NetCDF cannot hold (see ``netcdf_safe``)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = netcdf_safe(dataset)
    with NETCDF_LOCK:
        safe.to_netcdf(path, encoding=_datetime_encoding(safe))
    return path


def _datetime_encoding(dataset: xr.Dataset) -> dict[Hashable, dict[str, Any]]:
    """Exact, readable encodings for every datetime variable.

    Acquisition times carry microseconds, which xarray's default
    "days since ..." cannot hold. Integer microseconds work for complete
    variables, but integer-encoded NaT cannot be decoded again from Zarr v2
    or NetCDF (``*_matched_time`` is NaT wherever a source had no match), so variables
    with gaps are stored as float microseconds since their own first day:
    small offsets that float64 represents exactly at microsecond precision.
    """

    encoding: dict[Hashable, dict[str, Any]] = {}
    for name, variable in dataset.variables.items():
        if not np.issubdtype(variable.dtype, np.datetime64):
            continue
        values = np.asarray(variable.values).astype("datetime64[ns]")
        if not np.isnat(values).any():
            encoding[name] = {"units": "microseconds since 1970-01-01", "dtype": "int64"}
            continue
        finite = values[~np.isnat(values)]
        epoch = str(finite.min().astype("datetime64[D]")) if finite.size else "1970-01-01"
        encoding[name] = {"units": f"microseconds since {epoch}", "dtype": "float64"}
    return encoding


def write_zarr(
    dataset: xr.Dataset,
    store: str | Path,
    *,
    mode: Literal["w", "w-", "a", "a-", "r+", "r"] = "w",
    consolidated: bool = False,
) -> None:
    """Write a cube to a local or fsspec-backed Zarr store."""

    try:
        import zarr  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install sentinel-analysis[cloud] to write Zarr stores") from exc
    # Zarr v2 is currently the interoperable format for xarray and common
    # cloud readers; zarr v3 metadata support is still uneven across versions.
    encoding = _datetime_encoding(dataset) if mode in ("w", "w-") else None
    if encoding:
        # A time read back from NetCDF keeps that file's encoding; replace it.
        dataset = dataset.copy()
        for name in encoding:
            dataset[name].encoding = {}
    dataset.to_zarr(store, mode=mode, consolidated=consolidated, zarr_format=2, encoding=encoding)


def open_zarr(store: str | Path, *, chunks: str = "auto") -> xr.Dataset:
    """Open a Zarr cube lazily when Dask is available."""

    try:
        import zarr  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install sentinel-analysis[cloud] to read Zarr stores") from exc
    return xr.open_zarr(store, chunks=chunks, consolidated=False, zarr_format=2)


def write_cog(data: xr.DataArray, path: str | Path, *, compress: str = "DEFLATE") -> Path:
    """Write one regular-grid 2-D DataArray as a tiled Cloud Optimized GeoTIFF."""

    if data.ndim != 2 or set(data.dims) != {"y", "x"}:
        raise ValueError("COG output requires exactly 2-D data with y and x dimensions")
    if "crs" not in data.attrs:
        raise ValueError("COG output requires data.attrs['crs']")
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError as exc:
        raise RuntimeError("Install sentinel-analysis[optical] or sentinel-analysis[sar] to write COGs") from exc

    data = data.transpose("y", "x")
    x = np.asarray(data.coords["x"])
    y = np.asarray(data.coords["y"])
    if x.size < 2 or y.size < 2:
        raise ValueError("COG output requires at least two x and y coordinates")
    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    if not np.isclose(np.diff(x), dx).all() or not np.isclose(np.diff(y), dy).all():
        raise ValueError("COG output requires regularly spaced coordinates")
    values = np.asarray(data.values, dtype=np.float32)
    if dx < 0:
        values = np.flip(values, axis=1)
        left = float(x[-1] - abs(dx) / 2)
    else:
        left = float(x[0] - abs(dx) / 2)
    if dy > 0:
        values = np.flip(values, axis=0)
        top = float(y[-1] + abs(dy) / 2)
    else:
        top = float(y[0] + abs(dy) / 2)
    transform = from_origin(left, top, abs(dx), abs(dy))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output,
        "w",
        driver="COG",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype="float32",
        crs=data.attrs["crs"],
        transform=transform,
        nodata=np.nan,
        compress=compress,
        blocksize=512,
    ) as destination:
        destination.write(values, 1)
        destination.set_band_description(1, data.name or "value")
    return output


def read_cog(
    href: str | Path,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    band: int = 1,
) -> xr.DataArray:
    """Read a COG or HTTP range-addressable GeoTIFF window into xarray."""

    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError as exc:
        raise RuntimeError("Install sentinel-analysis[optical] or sentinel-analysis[sar] to read COGs") from exc
    with rasterio.open(href) as source:
        window = from_bounds(*bbox, transform=source.transform) if bbox else None
        values = source.read(band, window=window, masked=True).astype(np.float32).filled(np.nan)
        transform = source.window_transform(window) if window else source.transform
        height, width = values.shape
        x = transform.c + (np.arange(width) + 0.5) * transform.a
        y = transform.f + (np.arange(height) + 0.5) * transform.e
        return xr.DataArray(
            values,
            dims=("y", "x"),
            coords={"y": y, "x": x},
            name=source.descriptions[band - 1] or "band",
            attrs={"crs": source.crs.to_string() if source.crs else "unknown", "transform": tuple(transform), "source": str(href)},
        )


def validate_cog(path: str | Path) -> dict[str, object]:
    """Validate the structural properties required by a cloud raster asset."""

    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Install sentinel-analysis[optical] or sentinel-analysis[sar] to validate COGs") from exc
    with rasterio.open(path) as source:
        if source.crs is None:
            raise ValueError("COG has no CRS")
        tiled = bool(source.profile.get("tiled", False))  # rasterio's is_tiled is deprecated
        if not tiled:
            raise ValueError("COG is not tiled")
        return {
            "width": source.width,
            "height": source.height,
            "count": source.count,
            "crs": source.crs.to_string(),
            "overviews": [source.overviews(index) for index in range(1, source.count + 1)],
            "tiled": tiled,
        }
