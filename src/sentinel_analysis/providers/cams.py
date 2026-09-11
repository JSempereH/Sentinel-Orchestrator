"""CAMS atmospheric composition access through the official ADS API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from math import ceil, floor
import os
import importlib
from pathlib import Path
from typing import Any, Mapping
import zipfile

import xarray as xr
from dotenv import load_dotenv

from ..cache import AssetCache
from ..config import AOI
from .base import AuxiliaryArtifact, AuxiliaryProviderError, AuxiliarySpec, artifact_from_path, as_date, open_gridded_dataset, request_key, time_values


CAMS_VARIABLES: Mapping[str, Mapping[str, str]] = {
    "NO2": {"api": "total_column_nitrogen_dioxide", "source": "tcno2", "units": "kg m-2", "standard_name": "atmosphere_mass_content_of_nitrogen_dioxide"},
    "CO": {"api": "total_column_carbon_monoxide", "source": "tcco", "units": "kg m-2", "standard_name": "atmosphere_mass_content_of_carbon_monoxide"},
    "O3": {"api": "total_column_ozone", "source": "tco3", "units": "kg m-2", "standard_name": "atmosphere_mass_content_of_ozone"},
    "PM2.5": {"api": "particulate_matter_2.5um", "source": "pm2p5", "units": "kg m-3", "standard_name": "mass_concentration_of_pm2p5_ambient_aerosol_in_air"},
    "PM10": {"api": "particulate_matter_10um", "source": "pm10", "units": "kg m-3", "standard_name": "mass_concentration_of_pm10_ambient_aerosol_in_air"},
}
CAMS_SOURCE_TO_NAME = {value["source"]: name for name, value in CAMS_VARIABLES.items()}


@dataclass(frozen=True)
class CAMSConfig:
    dataset: str = "cams-global-reanalysis-eac4"
    api_url: str | None = None
    api_key: str | None = None
    data_format: str = "netcdf_zip"

    @classmethod
    def from_env(cls, dataset: str | None = None) -> "CAMSConfig":
        load_dotenv()
        resolved_dataset: str = dataset or os.getenv("CAMS_DATASET") or "cams-global-reanalysis-eac4"
        return cls(
            dataset=resolved_dataset,
            api_url=os.getenv("CAMS_API_URL") or os.getenv("CDS_API_URL") or os.getenv("CDSAPI_URL"),
            api_key=os.getenv("CAMS_API_KEY") or os.getenv("CDS_API_KEY") or os.getenv("CDSAPI_KEY"),
            data_format=os.getenv("CAMS_DATA_FORMAT", "netcdf_zip"),
        )


class CAMSProvider:
    """Download CAMS reanalysis or forecast data with explicit dataset choice."""

    name = "cams"

    def __init__(self, config: CAMSConfig | None = None):
        self.config = config or CAMSConfig.from_env()

    @staticmethod
    def variables_for(spec: AuxiliarySpec) -> tuple[str, ...]:
        return spec.variables or ("NO2", "CO", "O3", "PM2.5", "PM10")

    def build_request(self, aoi: AOI, start: str, end: str, *, variables: tuple[str, ...]) -> dict[str, Any]:
        unknown = sorted(set(variables).difference(CAMS_VARIABLES))
        if unknown:
            raise ValueError(f"Unsupported CAMS variables: {unknown}")
        first, last = as_date(start), as_date(end)
        dates = []
        current = first
        while current <= last:
            dates.append(current.isoformat())
            current += timedelta(days=1)
        resolution = 0.4 if "forecast" in self.config.dataset else 0.75
        west = floor(aoi.west / resolution) * resolution
        east = ceil(aoi.east / resolution) * resolution
        south = floor(aoi.south / resolution) * resolution
        north = ceil(aoi.north / resolution) * resolution
        if east <= west:
            east = west + resolution
        if north <= south:
            north = south + resolution
        request: dict[str, Any] = {
            "variable": [CAMS_VARIABLES[name]["api"] for name in variables],
            "date": dates,
            "time": time_values(),
            "area": [north, west, south, east],
            "data_format": self.config.data_format,
        }
        if "forecast" in self.config.dataset:
            request["type"] = ["forecast"]
            request["leadtime_hour"] = [str(hour) for hour in range(0, 121, 3)]
        return request

    def download(self, aoi: AOI, start: str, end: str, spec: AuxiliarySpec, output_dir: str | Path) -> AuxiliaryArtifact:
        dataset = spec.dataset or self.config.dataset
        config = self.config if dataset == self.config.dataset else CAMSConfig(dataset=dataset, api_url=self.config.api_url, api_key=self.config.api_key, data_format=self.config.data_format)
        provider = CAMSProvider(config)
        variables = provider.variables_for(spec)
        request = provider.build_request(aoi, start, end, variables=variables)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        key = request_key(self.name, dataset, request)
        suffix = ".zip" if config.data_format == "netcdf_zip" else ".grib"
        path = output_dir / f"{self.name}-{key.rsplit(':', 1)[-1]}{suffix}"
        cache = AssetCache(output_dir / ".cache")
        if cache.valid(key, path):
            return artifact_from_path(path, provider=self.name, dataset=dataset, request=request, cache=cache, manifest_path=output_dir / "manifest.json")
        try:
            cdsapi = importlib.import_module("cdsapi")
        except ImportError as exc:
            raise AuxiliaryProviderError("Install the 'auxiliary' extra to download CAMS data") from exc
        client_kwargs: dict[str, Any] = {}
        if config.api_url:
            client_kwargs["url"] = config.api_url
        if config.api_key:
            client_kwargs["key"] = config.api_key
        client = cdsapi.Client(**client_kwargs)
        temporary = path.with_suffix(path.suffix + ".part")
        client.retrieve(dataset, request, str(temporary))
        if not temporary.exists():
            raise AuxiliaryProviderError(f"ADS did not create the expected file: {temporary}")
        temporary.replace(path)
        return artifact_from_path(path, provider=self.name, dataset=dataset, request=request, cache=cache, manifest_path=output_dir / "manifest.json")

    def open(self, path: str | Path, *, dataset: str | None = None) -> xr.Dataset:
        path = Path(path)
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                members = [member for member in archive.namelist() if member.lower().endswith((".nc", ".netcdf"))]
                if len(members) != 1:
                    raise AuxiliaryProviderError(f"Expected one NetCDF file in CAMS archive, found {len(members)}")
                extracted = path.with_suffix(".nc")
                extracted.write_bytes(archive.read(members[0]))
                path = extracted
        metadata = {name: {"units": value["units"], "standard_name": value["standard_name"]} for name, value in CAMS_VARIABLES.items()}
        result = open_gridded_dataset(path, provider=self.name, dataset=dataset or self.config.dataset, variable_metadata=metadata)
        rename = {source: name for source, name in CAMS_SOURCE_TO_NAME.items() if source in result.data_vars and name not in result.data_vars}
        if rename:
            result = result.rename(rename)
        for name, value in result.data_vars.items():
            if name in CAMS_VARIABLES:
                value.attrs.update({
                    "units": CAMS_VARIABLES[name]["units"],
                    "standard_name": CAMS_VARIABLES[name]["standard_name"],
                })
        return result
