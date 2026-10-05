"""Machine-checkable variable contracts and provenance helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import xarray as xr


@dataclass(frozen=True)
class VariableContract:
    """Scientific metadata required for a published analysis variable."""

    units: str
    standard_name: str
    sensor: str
    product: str
    aggregation_method: str = "native"
    uncertainty_variable: str | None = None
    quality_variable: str | None = "valid_mask"
    footprint: str = "pixel"
    role: str = "feature"


VARIABLE_CONTRACTS: Mapping[str, VariableContract] = {
    "lst": VariableContract("K", "surface_temperature", "Sentinel-3", "SL_2_LST", role="target"),
    "landsat_lst": VariableContract("K", "surface_temperature", "Landsat-8/9", "L2SP", role="target"),
    "ecostress_lst": VariableContract("K", "surface_temperature", "ECOSTRESS", "L2T_LSTE", role="target"),
    "sigma0_VV": VariableContract("1", "radar_backscatter", "Sentinel-1", "S1_GRD", role="feature"),
    "sigma0_VH": VariableContract("1", "radar_backscatter", "Sentinel-1", "S1_GRD", role="feature"),
    "gamma0_VV": VariableContract("1", "radar_backscatter", "Sentinel-1", "S1_NRB", role="feature"),
    "gamma0_VH": VariableContract("1", "radar_backscatter", "Sentinel-1", "S1_NRB", role="feature"),
    "NDVI": VariableContract("1", "normalized_difference_vegetation_index", "Sentinel-2", "S2MSI2A"),
    "NO2": VariableContract("mol m-2", "atmosphere_mole_content_of_nitrogen_dioxide", "Sentinel-5P", "L2__NO2___", footprint="swath"),
    "SO2": VariableContract("mol m-2", "atmosphere_mole_content_of_sulfur_dioxide", "Sentinel-5P", "L2__SO2___", footprint="swath"),
    "CO": VariableContract("mol m-2", "atmosphere_mass_content_of_carbon_monoxide", "Sentinel-5P", "L2__CO____", footprint="swath"),
}

_KNOWN_DIAGNOSTIC_UNITS: Mapping[str, str] = {
    "lst_uncertainty": "K",
    "ndvi": "1",
    "biome": "1",
    "fractional_vegetation_cover": "1",
    "total_column_water_vapour": "kg m-2",
    "cloud_mask": "1",
    "cloud_flags": "1",
    "bayes_flags": "1",
    "confidence_flags": "1",
    "sat_zenith": "degree",
    "elevation": "m",
    "slope": "degree",
    "aspect": "degree",
    "cos_incidence": "1",
    "exception_flags": "1",
    "coverage_fraction": "1",
    "lst_invalid_observation_count": "1",
    "source_footprint_count": "1",
    "lst_observation_count": "1",
    "clear_observation_count": "1",
    "no_observation_mask": "1",
    "valid_mask": "1",
}


def apply_variable_contract(
    dataset: xr.Dataset,
    *,
    sensor: str,
    product: str,
    version: str = "unknown",
    aggregation_method: str | None = None,
    source: str | None = None,
) -> xr.Dataset:
    """Attach a complete contract to known variables without changing data."""

    result = dataset.copy()
    for name, value in result.data_vars.items():
        contract = VARIABLE_CONTRACTS.get(name)
        if contract:
            value.attrs.update({
                "units": value.attrs.get("units", contract.units),
                "standard_name": contract.standard_name,
                "sensor": contract.sensor,
                "product": contract.product,
                "aggregation_method": aggregation_method or contract.aggregation_method,
                "uncertainty_variable": contract.uncertainty_variable or "",
                "quality_variable": contract.quality_variable or "",
                "footprint": contract.footprint,
                "variable_role": contract.role,
            })
        else:
            if not value.attrs.get("units") and name in _KNOWN_DIAGNOSTIC_UNITS:
                value.attrs["units"] = _KNOWN_DIAGNOSTIC_UNITS[name]
            value.attrs.setdefault("sensor", sensor)
            value.attrs.setdefault("product", product)
            value.attrs.setdefault("aggregation_method", aggregation_method or "native")
    result.attrs.update({
        "sensor": sensor,
        "product": product,
        "processing_version": version,
        "source": source or result.attrs.get("source", ""),
        "metadata_contract": "sentinel-analysis-v1",
    })
    return result


def validate_variable_contract(dataset: xr.Dataset, *, required: Iterable[str] = ()) -> xr.Dataset:
    """Reject variables missing units, sensor, product and standard semantics."""

    missing = sorted(set(required).difference(dataset.data_vars))
    if missing:
        raise ValueError(f"Dataset is missing required variables: {missing}")
    errors: list[str] = []
    for name, value in dataset.data_vars.items():
        if name in {"valid_mask", "clear_observation_count", "lst_observation_count"}:
            continue
        for attribute in ("units", "sensor", "product", "aggregation_method"):
            if not value.attrs.get(attribute):
                errors.append(f"{name}.{attribute}")
        if name in VARIABLE_CONTRACTS and not value.attrs.get("standard_name"):
            errors.append(f"{name}.standard_name")
    if errors:
        raise ValueError(f"Variables do not satisfy the metadata contract: {errors}")
    return dataset


def ensure_compatible_units(left: xr.Dataset, right: xr.Dataset) -> None:
    """Reject obvious thermal/atmospheric unit and role mismatches before merge."""

    for name in set(left.data_vars).intersection(right.data_vars):
        left_units = left[name].attrs.get("units")
        right_units = right[name].attrs.get("units")
        if left_units and right_units and left_units != right_units:
            raise ValueError(f"Cannot merge {name}: units differ ({left_units!r} vs {right_units!r})")
