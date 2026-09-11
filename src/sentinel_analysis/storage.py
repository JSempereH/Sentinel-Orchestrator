"""Optional cloud-native persistence helpers for analysis cubes."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr


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
    dataset.to_zarr(store, mode=mode, consolidated=consolidated, zarr_format=2)


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
        if not source.is_tiled:
            raise ValueError("COG is not tiled")
        return {
            "width": source.width,
            "height": source.height,
            "count": source.count,
            "crs": source.crs.to_string(),
            "overviews": [source.overviews(index) for index in range(1, source.count + 1)],
            "tiled": source.is_tiled,
        }
