"""Temporal alignment primitives for Sentinel-1/2/3 feature fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .cube import CubeValidationError, validate_cube


@dataclass(frozen=True)
class TemporalMatch:
    """Policy for matching predictors to thermal observations."""

    tolerance: np.timedelta64 = np.timedelta64(3, "D")


@dataclass(frozen=True)
class TemporalPolicy:
    """Per-sensor temporal tolerances for feature matching."""

    default: np.timedelta64 = np.timedelta64(3, "D")
    by_sensor: dict[str, np.timedelta64] | None = None

    def for_sensor(self, sensor: str | None) -> TemporalMatch:
        tolerance = (self.by_sensor or {}).get(sensor or "", self.default)
        return TemporalMatch(tolerance)


def align_features(
    target: xr.Dataset,
    features: xr.Dataset,
    *,
    match: TemporalMatch | None = None,
    feature_name: str | None = None,
) -> xr.Dataset:
    """Align feature observations to target times without spatial interpolation."""

    validate_cube(target, require_time=True)
    validate_cube(features, require_time=True)
    if target.attrs.get("grid_id") and features.attrs.get("grid_id"):
        if target.attrs["grid_id"] != features.attrs["grid_id"]:
            raise CubeValidationError("Target and feature cubes use different grids")
    policy = match or TemporalMatch()
    # Nearest-time reindexing and the searchsorted below both require a
    # monotonic source index; acquisitions often arrive in catalogue
    # (cloud-cover) order rather than chronological order.
    features = features.sortby("time")
    boolean = [name for name, variable in features.data_vars.items() if variable.dtype == bool]
    aligned = features.reindex(
        time=target.time,
        method="nearest",
        tolerance=str(policy.tolerance),
    )
    # Unmatched times are NaN, which turns masks into float64 (8x the memory
    # of bool, and NaN reads as True); no match means "not valid".
    for name in boolean:
        aligned[name] = aligned[name].fillna(False).astype(bool)
    source_times = features.time.values.astype("datetime64[ns]")
    target_times = target.time.values.astype("datetime64[ns]")
    positions = np.searchsorted(source_times, target_times)
    positions = np.clip(positions, 0, max(0, len(source_times) - 1))
    left = np.maximum(positions - 1, 0)
    right = positions
    choose_right = np.abs(source_times[right] - target_times) < np.abs(source_times[left] - target_times)
    nearest = np.where(choose_right, source_times[right], source_times[left])
    deltas = np.abs(nearest - target_times).astype("timedelta64[ns]")
    valid = deltas <= policy.tolerance.astype("timedelta64[ns]")
    nearest = nearest.astype("datetime64[ns]")
    nearest[~valid] = np.datetime64("NaT", "ns")
    deltas[~valid] = np.timedelta64("NaT", "ns")
    prefix = f"{feature_name}_" if feature_name else ""
    aligned[f"{prefix}matched_time"] = xr.DataArray(nearest, dims="time", coords={"time": target.time})
    aligned[f"{prefix}time_delta"] = xr.DataArray(deltas, dims="time", coords={"time": target.time})
    aligned.attrs.update({
        "temporal_alignment": "nearest",
        "temporal_tolerance": str(policy.tolerance),
    })
    return xr.merge([target, aligned], compat="no_conflicts", join="exact")


def fusion_quality(dataset: xr.Dataset) -> xr.Dataset:
    """Add a per-time feature availability fraction for fusion diagnostics."""

    validate_cube(dataset, require_time=True)
    diagnostics = {
        "valid_mask", "feature_availability", "clear_observation_count", "lst_observation_count",
        "lst_invalid_observation_count", "source_footprint_count", "coverage_fraction", "no_observation_mask",
    }
    numeric = [
        name for name, value in dataset.data_vars.items()
        if value.dtype.kind in "fiu"
        and name not in diagnostics
        and name not in {"lst", "surface_temperature"}
        and not name.endswith("_time_delta")
        and not name.endswith("_matched_time")
        and not name.endswith("_station_count")
        and not name.endswith("_interpolation_uncertainty")
        and not name.endswith("_interpolation_valid")
        and value.attrs.get("variable_role") != "target"
    ]
    if numeric:
        availability = xr.concat(
            [xr.apply_ufunc(np.isfinite, dataset[name]).mean(dim=["y", "x"]) for name in numeric],
            dim="variable",
        ).mean(dim="variable")
    else:
        availability = xr.DataArray(np.full(dataset.sizes["time"], np.nan), dims="time", coords={"time": dataset.time})
    result = dataset.copy()
    result["feature_availability"] = availability.rename("feature_availability")
    result["feature_availability"].attrs["long_name"] = "fraction of finite feature pixels"
    result["feature_availability"].attrs["variables"] = numeric
    return result
