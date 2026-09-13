"""Carbon Mapper airborne methane plume catalog (static Zenodo dataset).

Unlike ERA5/CAMS/OpenAQ, this is one static, versioned file - the 2020-2021
airborne (AVIRIS-NG/GAO) plume list backing Varon et al. 2024 (Nat. Commun.
s41467-024-47754-y), published on Zenodo under
`10.5281/zenodo.7072824 <https://doi.org/10.5281/zenodo.7072824>`
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import requests
import xarray as xr

from ..cache import AssetCache
from ..config import AOI
from ..http import http_session
from .base import AuxiliaryArtifact, AuxiliaryProviderError, AuxiliarySpec, artifact_from_path, as_date, request_key


CARBON_MAPPER_CATALOG_URL = (
    "https://zenodo.org/api/records/7072824/files/carbonmapper_ch4_plumelist_2020_2021.xls/content"
)
CARBON_MAPPER_SHEET_NAME = "carbonmapper_ch4_plumelist_2020"
# xlrd/Excel's date epoch, per the ECMA-376 1900 date system Excel uses
# (1899-12-30, not 1900-01-01, to preserve Lotus 1-2-3's leap-year bug).
_EXCEL_EPOCH = date(1899, 12, 30)


@dataclass(frozen=True)
class CarbonMapperConfig:
    catalog_url: str = CARBON_MAPPER_CATALOG_URL
    sheet_name: str = CARBON_MAPPER_SHEET_NAME

    @classmethod
    def from_env(cls) -> "CarbonMapperConfig":
        return cls(
            catalog_url=os.getenv("CARBON_MAPPER_CATALOG_URL", CARBON_MAPPER_CATALOG_URL),
            sheet_name=os.getenv("CARBON_MAPPER_SHEET_NAME", CARBON_MAPPER_SHEET_NAME),
        )


def _require_xlrd():
    try:
        import xlrd
    except ImportError as exc:
        raise RuntimeError("Install sentinel-analysis[carbon_mapper] to parse the Carbon Mapper .xls catalog") from exc
    return xlrd


def _excel_serial_to_date(value: float) -> date:
    return _EXCEL_EPOCH + timedelta(days=int(value))


class CarbonMapperProvider:
    """Fetch and AOI/date-filter the Carbon Mapper airborne methane plume catalog."""

    name = "carbon_mapper"

    def __init__(self, config: CarbonMapperConfig | None = None, *, session: requests.Session | None = None):
        self.config = config or CarbonMapperConfig.from_env()
        self.session = session or http_session()

    def build_request(self, aoi: AOI, start: str | date | datetime, end: str | date | datetime) -> dict[str, Any]:
        return {
            "bbox": [aoi.west, aoi.south, aoi.east, aoi.north],
            "date_from": as_date(start).isoformat(),
            "date_to": as_date(end).isoformat(),
            "catalog_url": self.config.catalog_url,
        }

    def _fetch_raw_catalog(self, cache_dir: Path) -> Path:
        cache_dir.mkdir(parents=True, exist_ok=True)
        raw_path = cache_dir / "carbon_mapper_raw.xls"
        cache = AssetCache(cache_dir / ".cache")
        key = f"{self.name}:raw:{self.config.catalog_url}"
        if cache.valid(key, raw_path):
            return raw_path
        response = self.session.get(self.config.catalog_url, timeout=120)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise AuxiliaryProviderError(f"Carbon Mapper catalog request failed ({response.status_code})") from exc
        temporary = raw_path.with_suffix(raw_path.suffix + ".part")
        temporary.write_bytes(response.content)
        temporary.replace(raw_path)
        cache.record(key, raw_path, metadata={"provider": self.name, "url": self.config.catalog_url})
        return raw_path

    def _read_plumes(self, raw_path: Path) -> list[dict[str, Any]]:
        xlrd = _require_xlrd()
        book = xlrd.open_workbook(str(raw_path))
        sheet = book.sheet_by_name(self.config.sheet_name)
        header = [str(sheet.cell_value(0, column)) for column in range(sheet.ncols)]
        return [{header[column]: sheet.cell_value(row, column) for column in range(sheet.ncols)} for row in range(1, sheet.nrows)]

    def download(self, aoi: AOI, start: str | date | datetime, end: str | date | datetime, spec: AuxiliarySpec, output_dir: str | Path) -> AuxiliaryArtifact:
        if spec.options:
            config = CarbonMapperConfig(
                catalog_url=str(spec.options.get("catalog_url", self.config.catalog_url)),
                sheet_name=str(spec.options.get("sheet_name", self.config.sheet_name)),
            )
            provider = CarbonMapperProvider(config, session=self.session)
            return provider.download(aoi, start, end, AuxiliarySpec(spec.provider, spec.variables, spec.dataset), output_dir)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset = spec.dataset or "carbon-mapper-2020-2021"
        request = self.build_request(aoi, start, end)
        key = request_key(self.name, dataset, request)
        path = output_dir / f"{self.name}-{key.rsplit(':', 1)[-1]}.json"
        cache = AssetCache(output_dir / ".cache")
        if cache.valid(key, path):
            return artifact_from_path(path, provider=self.name, dataset=dataset, request=request, cache=cache, manifest_path=output_dir / "manifest.json")

        raw_path = self._fetch_raw_catalog(output_dir / "raw")
        west, south, east, north = request["bbox"]
        start_date, end_date = as_date(start), as_date(end)
        plumes: list[dict[str, Any]] = []
        for row in self._read_plumes(raw_path):
            latitude, longitude = float(row["plume_lat"]), float(row["plume_lon"])
            if not (west <= longitude <= east and south <= latitude <= north):
                continue
            plume_date = _excel_serial_to_date(float(row["date"]))
            if not (start_date <= plume_date <= end_date):
                continue
            plumes.append({
                "source_id": str(row["source_id"]),
                "candidate_id": str(row["candidate_id"]),
                "latitude": latitude,
                "longitude": longitude,
                "date": plume_date.isoformat(),
                "emission_rate_kg_h": float(row["qplume"]),
                "emission_rate_uncertainty_kg_h": float(row["sigma_qplume"]),
            })

        payload = {"provider": self.name, "source_url": self.config.catalog_url, "request": request, "plumes": plumes}
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
        return artifact_from_path(path, provider=self.name, dataset=dataset, request=request, cache=cache, manifest_path=output_dir / "manifest.json")

    def open(self, path: str | Path, *, dataset: str = "carbon-mapper-2020-2021") -> xr.Dataset:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        plumes = payload.get("plumes", [])
        common_attrs = {
            "source": self.name,
            "product": dataset,
            "crs": "EPSG:4326",
            "analysis_shape": "point_table",
            "processing_version": "sentinel-analysis-auxiliary-v1",
            "metadata_contract": "sentinel-analysis-v1",
        }
        if not plumes:
            return xr.Dataset(
                coords={"plume": [], "latitude": ("plume", []), "longitude": ("plume", []), "time": ("plume", np.array([], dtype="datetime64[ns]"))},
                attrs=common_attrs,
            )
        return xr.Dataset(
            {
                "emission_rate": ("plume", [item["emission_rate_kg_h"] for item in plumes], {"units": "kg h-1", "sensor": self.name, "product": dataset, "variable_role": "reference"}),
                "emission_rate_uncertainty": ("plume", [item["emission_rate_uncertainty_kg_h"] for item in plumes], {"units": "kg h-1", "sensor": self.name, "product": dataset, "variable_role": "diagnostic"}),
                "source_id": ("plume", [item["source_id"] for item in plumes]),
            },
            coords={
                "plume": [item["candidate_id"] for item in plumes],
                "latitude": ("plume", [item["latitude"] for item in plumes]),
                "longitude": ("plume", [item["longitude"] for item in plumes]),
                "time": ("plume", np.asarray([item["date"] for item in plumes], dtype="datetime64[ns]")),
            },
            attrs=common_attrs,
        )
