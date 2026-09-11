"""Configuration and input validation for the Sentinel-3 client."""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv


@dataclass(frozen=True)
class AOI:
    """A WGS84 bounding box."""

    west: float
    south: float
    east: float
    north: float

    def as_extent(self) -> dict[str, float | str]:
        values = (self.west, self.south, self.east, self.north)
        if not all(isfinite(value) for value in values):
            raise ValueError("AOI bounds must be finite numbers")
        if not (-180 <= self.west < self.east <= 180):
            raise ValueError("AOI longitudes must satisfy -180 <= west < east <= 180")
        if not (-90 <= self.south < self.north <= 90):
            raise ValueError("AOI bounds must satisfy west < east and south < north")
        return {
            "west": self.west,
            "south": self.south,
            "east": self.east,
            "north": self.north,
            "crs": "EPSG:4326",
        }

    def as_wkt(self) -> str:
        """Return the bounding box as an OGC WKT polygon."""

        self.as_extent()
        return (
            "POLYGON (("
            f"{self.west} {self.south}, "
            f"{self.east} {self.south}, "
            f"{self.east} {self.north}, "
            f"{self.west} {self.north}, "
            f"{self.west} {self.south}))"
        )

    def as_geojson(self) -> dict[str, object]:
        """Return a GeoJSON polygon suitable for provider adapters."""

        self.as_extent()
        return {
            "type": "Polygon",
            "coordinates": [[
                [self.west, self.south],
                [self.east, self.south],
                [self.east, self.north],
                [self.west, self.north],
                [self.west, self.south],
            ]],
        }


@dataclass(frozen=True)
class ClientConfig:
    """Connection settings loaded from environment variables."""

    client_id: str
    client_secret: str
    backend_url: str = "https://openeo.dataspace.copernicus.eu"
    catalog_url: str = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    download_url: str = "https://download.dataspace.copernicus.eu/odata/v1/Products"
    token_url: str = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
        "protocol/openid-connect/token"
    )
    username: str | None = None
    password: str | None = None
    cdse_connect_timeout_s: float = 30.0
    cdse_read_timeout_s: float = 600.0
    cdse_download_retries: int = 6

    def __post_init__(self) -> None:
        if self.cdse_connect_timeout_s <= 0 or self.cdse_read_timeout_s <= 0:
            raise ValueError("CDSE timeouts must be positive")
        if self.cdse_download_retries < 1:
            raise ValueError("cdse_download_retries must be positive")

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> "ClientConfig":
        if env_file is not None:
            load_dotenv(Path(env_file))
        client_id = os.getenv("CDSE_CLIENT_ID") or os.getenv("SH_CLIENT_ID")
        client_secret = os.getenv("CDSE_CLIENT_SECRET") or os.getenv("SH_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise RuntimeError(
                "Missing CDSE credentials. Set CDSE_CLIENT_ID and "
                "CDSE_CLIENT_SECRET in .env or the environment."
            )
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            username=os.getenv("CDSE_USERNAME"),
            password=os.getenv("CDSE_PASSWORD"),
            cdse_connect_timeout_s=float(os.getenv("CDSE_CONNECT_TIMEOUT_S", "30")),
            cdse_read_timeout_s=float(os.getenv("CDSE_READ_TIMEOUT_S", "600")),
            cdse_download_retries=int(os.getenv("CDSE_DOWNLOAD_RETRIES", "6")),
            backend_url=os.getenv(
                "CDSE_BACKEND_URL", "https://openeo.dataspace.copernicus.eu"
            ),
            catalog_url=os.getenv(
                "CDSE_CATALOG_URL",
                "https://catalogue.dataspace.copernicus.eu/odata/v1/Products",
            ),
            download_url=os.getenv(
                "CDSE_DOWNLOAD_URL",
                "https://download.dataspace.copernicus.eu/odata/v1/Products",
            ),
            token_url=os.getenv(
                "CDSE_TOKEN_URL",
                "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
                "protocol/openid-connect/token",
            ),
        )


def extent_from_mapping(value: Mapping[str, float]) -> dict[str, float | str]:
    """Validate and normalize a mapping with west, south, east and north."""

    required = ("west", "south", "east", "north")
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"Missing AOI keys: {', '.join(missing)}")
    return AOI(*(float(value[key]) for key in required)).as_extent()
