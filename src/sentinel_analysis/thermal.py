"""Historical heat-hazard metrics for analysis-ready city cubes."""

from __future__ import annotations

import xarray as xr

from .cube import validate_cube


def heat_hazard_metrics(
    dataset: xr.Dataset,
    *,
    lst_name: str = "lst",
    threshold_celsius: float = 35.0,
) -> xr.Dataset:
    """Compute transparent heat indicators without filling missing observations."""

    validate_cube(dataset, required_variables=(lst_name,), require_time=True)
    lst = dataset[lst_name]
    exceedance = (lst >= threshold_celsius).sum(dim="time", skipna=True)
    result = xr.Dataset({
        "lst_mean": lst.mean(dim="time", skipna=True),
        "lst_p95": lst.quantile(0.95, dim="time", skipna=True),
        "lst_max": lst.max(dim="time", skipna=True),
        "hot_observation_count": exceedance,
        "observation_count": lst.notnull().sum(dim="time"),
    })
    result.attrs.update(dataset.attrs)
    result.attrs.update({
        "hazard_definition": "surface-temperature observation statistics",
        "hot_threshold_celsius": threshold_celsius,
        "missing_values_filled": False,
    })
    return result
