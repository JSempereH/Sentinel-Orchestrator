"""Reusable city definitions for global urban analysis."""

from __future__ import annotations

from dataclasses import dataclass

from .config import AOI


@dataclass(frozen=True)
class CitySpec:
    """City analysis extent and its preferred local projected CRS."""

    city_id: str
    name: str
    country: str
    aoi: AOI
    timezone: str
    crs: str
    target_resolution_m: int = 100

    @property
    def grid_id(self) -> str:
        return f"{self.city_id}:{self.crs}:{self.target_resolution_m}m"


_CITIES = {
    "berlin": CitySpec(
        "berlin", "Berlin", "Germany", AOI(13.20, 52.40, 13.55, 52.58),
        "Europe/Berlin", "EPSG:32633",
    ),
    "guadalajara": CitySpec(
        "guadalajara", "Guadalajara", "Mexico", AOI(-103.60, 20.50, -103.20, 20.85),
        "America/Mexico_City", "EPSG:32613",
    ),
    "mexico-city": CitySpec(
        "mexico-city", "Mexico City", "Mexico", AOI(-99.35, 19.15, -98.95, 19.60),
        "America/Mexico_City", "EPSG:32614",
    ),
    "lagos": CitySpec(
        "lagos", "Lagos", "Nigeria", AOI(2.70, 6.35, 3.70, 6.80),
        "Africa/Lagos", "EPSG:32631",
    ),
    "nairobi": CitySpec(
        "nairobi", "Nairobi", "Kenya", AOI(36.55, -1.45, 37.15, -1.05),
        "Africa/Nairobi", "EPSG:32737",
    ),
}


def get_city(city_id: str) -> CitySpec:
    """Return a registered city by stable identifier."""

    key = city_id.strip().lower().replace("_", "-")
    try:
        return _CITIES[key]
    except KeyError as exc:
        available = ", ".join(sorted(_CITIES))
        raise KeyError(f"Unknown city {city_id!r}. Available cities: {available}") from exc


def list_cities() -> tuple[CitySpec, ...]:
    """Return registered cities in stable identifier order."""

    return tuple(_CITIES[key] for key in sorted(_CITIES))
