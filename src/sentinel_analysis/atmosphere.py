"""Atmospheric-column semantics and deliberately explicit surface estimates."""

from __future__ import annotations

from dataclasses import dataclass

import xarray as xr


MOLECULAR_WEIGHT_G_PER_MOL = {
    "NO2": 46.0055,
    "SO2": 64.066,
    "CO": 28.0101,
    "O3": 47.9982,
    "HCHO": 30.026,
}


@dataclass(frozen=True)
class ColumnToSurfaceConfig:
    """Assumptions for a first-order column-to-surface conversion."""

    gas: str
    boundary_layer_height_variable: str = "boundary_layer_height"
    uniform_mixing_in_boundary_layer: bool = True


def column_to_surface_estimate(dataset: xr.Dataset, *, config: ColumnToSurfaceConfig) -> xr.Dataset:
    """Estimate surface mass concentration from a column and boundary layer.

    This is an explicitly modelled approximation, not a retrieval of surface
    concentration. It assumes the column is uniformly mixed through the
    supplied boundary-layer height and does not apply an averaging kernel.
    """

    gas = config.gas.upper()
    if gas not in MOLECULAR_WEIGHT_G_PER_MOL:
        raise ValueError(f"No molecular weight is registered for {gas}")
    if gas not in dataset:
        raise KeyError(f"Dataset does not contain atmospheric column {gas}")
    if config.boundary_layer_height_variable not in dataset:
        raise KeyError(f"Dataset does not contain {config.boundary_layer_height_variable}")
    column = dataset[gas]
    height = dataset[config.boundary_layer_height_variable]
    output = (column * MOLECULAR_WEIGHT_G_PER_MOL[gas] * 1_000_000 / height.where(height > 0)).rename(f"{gas}_surface_estimated")
    output.attrs.update({
        "units": "ug m-3",
        "standard_name": "mass_concentration_of_air_pollutant",
        "sensor": "Sentinel-5P plus meteorology",
        "product": "modelled_column_to_surface",
        "variable_role": "estimated_surface_concentration",
        "modelled": True,
        "assumption": "uniform mixing through boundary layer",
        "averaging_kernel_applied": False,
    })
    result = dataset.copy()
    result[output.name] = output
    result.attrs.update({"atmospheric_surface_estimate": True, "atmospheric_surface_estimate_gas": gas})
    return result
