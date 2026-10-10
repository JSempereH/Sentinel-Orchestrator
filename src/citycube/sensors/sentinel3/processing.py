"""Local xarray processing helpers for standardized LST datasets."""

from __future__ import annotations


import numpy as np
import xarray as xr

from .quality import QualityPolicy, apply_quality_mask


def to_celsius(dataset: xr.Dataset, *, lst_name: str = "lst") -> xr.Dataset:
    """Convert the LST variable to Celsius using its declared units."""

    if lst_name not in dataset:
        raise KeyError(f"Dataset does not contain {lst_name!r}")
    result = dataset.copy()
    units = str(result[lst_name].attrs.get("units", "K")).strip().lower()
    original_attrs = dict(result[lst_name].attrs)
    if units in {"k", "kelvin", "degrees_kelvin"}:
        result[lst_name] = result[lst_name] - 273.15
        result[lst_name].attrs.update(original_attrs)
    elif units in {"c", "degc", "degree_celsius", "degrees_celsius", "celsius", "°c"}:
        pass
    else:
        raise ValueError(f"Unsupported or missing LST units: {units!r}")
    result[lst_name].attrs.update({"units": "degC", "long_name": "land surface temperature"})
    result.attrs["temperature_conversion"] = "Kelvin to Celsius" if units.startswith("k") else "none"
    return result


def daily_mean(
    dataset: xr.Dataset,
    *,
    quality: QualityPolicy | None = None,
    assume_quality_masked: bool = False,
) -> xr.Dataset:
    """Aggregate LST to daily means without averaging bit fields."""

    if "time" not in dataset.dims and "time" not in dataset.coords:
        raise ValueError("Dataset needs a time dimension for daily aggregation")
    prepared = dataset if assume_quality_masked else apply_quality_mask(dataset, quality)
    grouped = prepared["lst"].resample(time="1D").mean(skipna=True)
    result = xr.Dataset({"lst": grouped})
    if "lst_uncertainty" in prepared:
        result["lst_uncertainty"] = prepared["lst_uncertainty"].resample(time="1D").mean(skipna=True)
    result["valid_count"] = prepared["lst"].notnull().resample(time="1D").sum(dim="time")
    result.attrs.update(prepared.attrs)
    result.attrs["temporal_aggregation"] = "daily mean of quality-masked LST"
    return result


def clip_bbox(dataset: xr.Dataset, west: float, south: float, east: float, north: float) -> xr.Dataset:
    """Clip using 1-D or 2-D longitude/latitude variables without reprojection."""

    if "longitude" not in dataset or "latitude" not in dataset:
        raise ValueError("Dataset must contain longitude and latitude for geographic clipping")
    longitude = dataset["longitude"]
    latitude = dataset["latitude"]
    mask = (longitude >= west) & (longitude <= east) & (latitude >= south) & (latitude <= north)
    return dataset.where(mask, drop=True)


def statistics(dataset: xr.Dataset, *, lst_name: str = "lst") -> dict[str, float | int]:
    """Compute descriptive statistics after respecting ``valid_mask`` when present."""

    if lst_name not in dataset:
        raise KeyError(f"Dataset does not contain {lst_name!r}")
    values = dataset[lst_name]
    if "valid_mask" in dataset:
        values = values.where(dataset["valid_mask"])
    values = values.where(xr.apply_ufunc(np.isfinite, values))
    count = int(values.count().item())
    if count == 0:
        raise ValueError("Dataset contains no valid LST pixels")
    return {
        "pixels_total": int(values.size),
        "pixels_valid": count,
        "mean": float(values.mean(skipna=True).item()),
        "median": float(values.median(skipna=True).item()),
        "min": float(values.min(skipna=True).item()),
        "max": float(values.max(skipna=True).item()),
    }
