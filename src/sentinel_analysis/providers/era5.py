"""ERA5 and ERA5-Land access through the official CDS API."""

from __future__ import annotations

from dataclasses import dataclass
import os
import importlib
from pathlib import Path
from typing import Any, Mapping

import xarray as xr
from dotenv import load_dotenv

from ..cache import AssetCache
from ..config import AOI
from .base import (
    AuxiliaryArtifact,
    AuxiliaryProviderError,
    AuxiliarySpec,
    artifact_from_path,
    date_parts,
    open_gridded_dataset,
    request_key,
    time_values,
)


ERA5_VARIABLES: Mapping[str, Mapping[str, str]] = {
    "air_temperature_2m": {"api": "2m_temperature", "source": "t2m", "units": "K", "standard_name": "air_temperature"},
    "dewpoint_temperature_2m": {"api": "2m_dewpoint_temperature", "source": "d2m", "units": "K", "standard_name": "dew_point_temperature"},
    "wind_u_10m": {"api": "10m_u_component_of_wind", "source": "u10", "units": "m s-1", "standard_name": "eastward_wind"},
    "wind_v_10m": {"api": "10m_v_component_of_wind", "source": "v10", "units": "m s-1", "standard_name": "northward_wind"},
    "surface_pressure": {"api": "surface_pressure", "source": "sp", "units": "Pa", "standard_name": "surface_air_pressure"},
    "skin_temperature": {"api": "skin_temperature", "source": "skt", "units": "K", "standard_name": "surface_temperature"},
    "total_precipitation": {"api": "total_precipitation", "source": "tp", "units": "m", "standard_name": "precipitation_flux"},
    "surface_solar_radiation_downwards": {"api": "surface_solar_radiation_downwards", "source": "ssrd", "units": "J m-2", "standard_name": "surface_downwelling_shortwave_flux_in_air"},
    "soil_moisture_layer_1": {"api": "volumetric_soil_water_layer_1", "source": "swvl1", "units": "m3 m-3", "standard_name": "volume_fraction_of_condensed_water_in_soil"},
    "boundary_layer_height": {"api": "boundary_layer_height", "source": "blh", "units": "m", "standard_name": "atmosphere_boundary_layer_thickness"},
    "mean_sea_level_pressure": {"api": "mean_sea_level_pressure", "source": "msl", "units": "Pa", "standard_name": "air_pressure_at_mean_sea_level"},
}
ERA5_SOURCE_TO_NAME = {value["source"]: name for name, value in ERA5_VARIABLES.items()}


@dataclass(frozen=True)
class ERA5Config:
    """CDS settings. Credentials may also come from ``~/.cdsapirc``."""

    dataset: str = "reanalysis-era5-land"
    api_url: str | None = None
    api_key: str | None = None
    data_format: str = "netcdf"

    @classmethod
    def from_env(cls, dataset: str | None = None) -> "ERA5Config":
        load_dotenv()
        resolved_dataset: str = dataset or os.getenv("ERA5_DATASET") or "reanalysis-era5-land"
        return cls(
            dataset=resolved_dataset,
            api_url=os.getenv("CDS_API_URL") or os.getenv("CDSAPI_URL"),
            api_key=os.getenv("CDS_API_KEY") or os.getenv("CDSAPI_KEY"),
            data_format=os.getenv("CDS_DATA_FORMAT", "netcdf"),
        )


class ERA5Provider:
    """Build, download and open an ERA5/ERA5-Land request."""

    name = "era5"

    def __init__(self, config: ERA5Config | None = None):
        self.config = config or ERA5Config.from_env()

    @staticmethod
    def variables_for(spec: AuxiliarySpec) -> tuple[str, ...]:
        return spec.variables or (
            "air_temperature_2m",
            "dewpoint_temperature_2m",
            "wind_u_10m",
            "wind_v_10m",
            "surface_pressure",
            "skin_temperature",
            "total_precipitation",
            "surface_solar_radiation_downwards",
        )

    def build_request(
        self,
        aoi: AOI,
        start: str,
        end: str,
        *,
        variables: tuple[str, ...],
    ) -> dict[str, Any]:
        unknown = sorted(set(variables).difference(ERA5_VARIABLES))
        if unknown:
            raise ValueError(f"Unsupported ERA5 variables: {unknown}")
        years, months, days = date_parts(start, end)
        return {
            "product_type": ["reanalysis"],
            "variable": [ERA5_VARIABLES[name]["api"] for name in variables],
            "year": years,
            "month": months,
            "day": days,
            "time": time_values(),
            "area": [aoi.north, aoi.west, aoi.south, aoi.east],
            "data_format": self.config.data_format,
            "download_format": "unarchived",
        }

    def download(
        self,
        aoi: AOI,
        start: str,
        end: str,
        spec: AuxiliarySpec,
        output_dir: str | Path,
    ) -> AuxiliaryArtifact:
        dataset = spec.dataset or self.config.dataset
        variables = self.variables_for(spec)
        request = self.build_request(aoi, start, end, variables=variables)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        key = request_key(self.name, dataset, request)
        path = output_dir / f"{self.name}-{key.rsplit(':', 1)[-1]}.nc"
        cache = AssetCache(output_dir / ".cache")
        if cache.valid(key, path):
            return artifact_from_path(path, provider=self.name, dataset=dataset, request=request, cache=cache, manifest_path=output_dir / "manifest.json")
        try:
            cdsapi = importlib.import_module("cdsapi")
        except ImportError as exc:
            raise AuxiliaryProviderError("Install the 'auxiliary' extra to download ERA5 data") from exc
        client_kwargs: dict[str, Any] = {}
        if self.config.api_url:
            client_kwargs["url"] = self.config.api_url
        if self.config.api_key:
            client_kwargs["key"] = self.config.api_key
        client = cdsapi.Client(**client_kwargs)
        temporary = path.with_suffix(path.suffix + ".part")
        client.retrieve(dataset, request, str(temporary))
        if not temporary.exists():
            raise AuxiliaryProviderError(f"CDS did not create the expected file: {temporary}")
        temporary.replace(path)
        return artifact_from_path(path, provider=self.name, dataset=dataset, request=request, cache=cache, manifest_path=output_dir / "manifest.json")

    def open(self, path: str | Path, *, dataset: str | None = None) -> xr.Dataset:
        metadata = {name: {"units": value["units"], "standard_name": value["standard_name"]} for name, value in ERA5_VARIABLES.items()}
        result = open_gridded_dataset(path, provider=self.name, dataset=dataset or self.config.dataset, variable_metadata=metadata)
        rename = {source: name for source, name in ERA5_SOURCE_TO_NAME.items() if source in result.data_vars and name not in result.data_vars}
        if rename:
            result = result.rename(rename)
        for name, value in result.data_vars.items():
            if name in ERA5_VARIABLES:
                value.attrs.update({
                    "units": ERA5_VARIABLES[name]["units"],
                    "standard_name": ERA5_VARIABLES[name]["standard_name"],
                })
        return result
