"""Shared contracts and cache helpers for auxiliary providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import xarray as xr

from ..cache import AssetCache


AUXILIARY_PROVIDERS = ("era5", "cams", "openaq", "carbon_mapper")


class AuxiliaryProviderError(RuntimeError):
    """Raised when an auxiliary provider cannot fulfil a request."""


@dataclass(frozen=True)
class AuxiliarySpec:
    """Serializable selection for one auxiliary provider.

    ``options`` contains provider-specific values such as a CDS dataset or
    the OpenAQ maximum number of stations. It must remain JSON serializable.
    """

    provider: str
    variables: tuple[str, ...] = ()
    dataset: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        provider = self.provider.lower()
        if provider not in AUXILIARY_PROVIDERS:
            raise ValueError(f"Unsupported auxiliary provider: {self.provider!r}")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "variables", tuple(self.variables))
        options = dict(self.options)
        sensitive = {"api_key", "token", "password", "secret", "client_secret"}
        if any(str(key).lower() in sensitive for key in options):
            raise ValueError("Auxiliary options must not contain credentials")
        object.__setattr__(self, "options", options)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AuxiliarySpec":
        return cls(
            provider=str(value["provider"]),
            variables=tuple(value.get("variables", ())),
            dataset=value.get("dataset"),
            options=dict(value.get("options", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "variables": list(self.variables),
            "dataset": self.dataset,
            "options": dict(self.options),
        }


@dataclass(frozen=True)
class AuxiliaryArtifact:
    """A cached provider response plus its reproducibility metadata."""

    provider: str
    dataset: str
    path: Path
    sha256: str
    manifest_path: Path
    request: dict[str, Any]


def as_date(value: str | date | datetime) -> date:
    """Normalize an ISO date/datetime to a calendar date."""

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def date_parts(start: str | date | datetime, end: str | date | datetime) -> tuple[list[str], list[str], list[str]]:
    """Return unique CDS year, month and day selections for an inclusive range."""

    first, last = as_date(start), as_date(end)
    if first > last:
        raise ValueError("Auxiliary request start must not be after end")
    values: list[date] = []
    current = first
    while current <= last:
        values.append(current)
        current += timedelta(days=1)
    return (
        sorted({value.strftime("%Y") for value in values}),
        sorted({value.strftime("%m") for value in values}),
        sorted({value.strftime("%d") for value in values}),
    )


def time_values() -> list[str]:
    """Return the hourly selection accepted by CDS datasets."""

    return [f"{hour:02d}:00" for hour in range(24)]


def canonical_request(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def request_key(provider: str, dataset: str, request: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_request(request).encode("utf-8")).hexdigest()[:20]
    return f"{provider}:{dataset}:{digest}"


def _write_manifest(path: Path, artifact: AuxiliaryArtifact) -> None:
    manifest_path = artifact.manifest_path
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[request_key(artifact.provider, artifact.dataset, artifact.request)] = {
        "provider": artifact.provider,
        "dataset": artifact.dataset,
        "path": str(artifact.path),
        "sha256": artifact.sha256,
        "request": artifact.request,
    }
    temporary = manifest_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(manifest_path)


def artifact_from_path(
    path: Path,
    *,
    provider: str,
    dataset: str,
    request: Mapping[str, Any],
    cache: AssetCache,
    manifest_path: Path,
) -> AuxiliaryArtifact:
    """Record a completed file in both the checksum cache and public manifest."""

    path = Path(path)
    key = request_key(provider, dataset, request)
    checksum = cache.checksum(path)
    cache.record(key, path, metadata={"provider": provider, "dataset": dataset, "request": dict(request)})
    artifact = AuxiliaryArtifact(provider, dataset, path, checksum, manifest_path, dict(request))
    _write_manifest(manifest_path, artifact)
    return artifact


def open_gridded_dataset(
    path: str | Path,
    *,
    provider: str,
    dataset: str,
    variable_metadata: Mapping[str, Mapping[str, str]] | None = None,
) -> xr.Dataset:
    """Open a NetCDF/GRIB response and normalize it to y/x EPSG:4326."""

    path = Path(path)
    try:
        result = xr.open_dataset(path)
    except (ValueError, ImportError):
        try:
            result = xr.open_dataset(path, engine="cfgrib")
        except Exception as exc:  # pragma: no cover - depends on optional cfgrib
            raise AuxiliaryProviderError(
                f"Cannot open {path}; install netCDF4 or cfgrib/eccodes"
            ) from exc

    rename: dict[str, str] = {}
    for old, new in (("latitude", "y"), ("lat", "y"), ("longitude", "x"), ("lon", "x"), ("valid_time", "time")):
        if old in result.dims or old in result.coords:
            if new not in result.dims and new not in result.coords:
                rename[old] = new
    if rename:
        result = result.rename(rename)
    if "time" not in result.dims and "time" in result.coords:
        result = result.expand_dims("time")
    result.attrs.update({
        "crs": "EPSG:4326",
        "source": provider,
        "product": dataset,
        "analysis_shape": "regular_grid",
        "processing_version": "sentinel-analysis-auxiliary-v1",
        "metadata_contract": "sentinel-analysis-v1",
    })
    metadata = variable_metadata or {}
    for name, variable in result.data_vars.items():
        info = metadata.get(name, {})
        variable.attrs.setdefault("units", info.get("units", "unknown"))
        variable.attrs.setdefault("standard_name", info.get("standard_name", name))
        variable.attrs.update({
            "sensor": provider,
            "product": dataset,
            "aggregation_method": "native",
            "variable_role": "feature",
        })
    return result
