"""Configuration and input validation for the Sentinel-3 client."""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv


# A city boundary rarely needs more; it bounds the cost of masks and queries.
MAX_AOI_VERTICES = 10_000


@dataclass(frozen=True)
class AOI:
    """A WGS84 area of interest: a bounding box, optionally with its exact polygon.

    Catalogue searches always use the bounding box (a detailed boundary would
    not fit in a query URL). ``geometry_wkt``, when given, is the real shape
    (a city or district boundary): it masks results, measures how much of it
    a product covers and sizes requests. Build one from GeoJSON with
    ``AOI.from_geojson`` or from WKT with ``AOI.from_geometry``.
    """

    west: float
    south: float
    east: float
    north: float
    geometry_wkt: str | None = None

    def __post_init__(self) -> None:
        # Validated on creation: an inverted or out-of-range box used to pass
        # through requests and only fail deep inside grid construction.
        values = (self.west, self.south, self.east, self.north)
        if not all(isinstance(value, Real) and not isinstance(value, bool) for value in values):
            raise ValueError("AOI bounds must be numbers")
        if not all(isfinite(value) for value in values):
            raise ValueError("AOI bounds must be finite numbers")
        if not (-180 <= self.west < self.east <= 180):
            raise ValueError("AOI longitudes must satisfy -180 <= west < east <= 180")
        if not (-90 <= self.south < self.north <= 90):
            raise ValueError("AOI latitudes must satisfy -90 <= south < north <= 90")
        if self.geometry_wkt is not None:
            geometry = self.shape()
            if geometry.geom_type not in ("Polygon", "MultiPolygon") or geometry.is_empty:
                raise ValueError("An AOI geometry must be a non-empty Polygon or MultiPolygon")
            west, south, east, north = geometry.bounds
            tolerance = 1e-6
            if west < self.west - tolerance or east > self.east + tolerance or south < self.south - tolerance or north > self.north + tolerance:
                raise ValueError("The AOI geometry extends beyond its bounding box")

    @classmethod
    def from_geometry(cls, geometry: "Mapping[str, object] | str") -> "AOI":
        """An AOI from a GeoJSON geometry, Feature or FeatureCollection, or a WKT string (WGS84)."""

        import shapely
        from shapely import wkt
        from shapely.geometry import shape

        try:
            if isinstance(geometry, str):
                parsed = wkt.loads(geometry)
            elif geometry.get("type") == "FeatureCollection":
                features: list = geometry.get("features") or []  # type: ignore[assignment]
                parts = [shape(feature["geometry"]) for feature in features]
                if int(shapely.get_num_coordinates(parts).sum()) > MAX_AOI_VERTICES:
                    raise ValueError(f"The AOI polygons have more than {MAX_AOI_VERTICES} vertices; simplify them first")
                parsed = shapely.unary_union(parts)
            elif geometry.get("type") == "Feature":
                parsed = shape(geometry["geometry"])  # type: ignore[arg-type]
            else:
                parsed = shape(geometry)
            if shapely.get_num_coordinates(parsed) > MAX_AOI_VERTICES:
                raise ValueError(f"The AOI polygon has more than {MAX_AOI_VERTICES} vertices; simplify it first")
            parsed = shapely.make_valid(parsed)
            if parsed.geom_type == "GeometryCollection":
                parsed = shapely.unary_union([part for part in parsed.geoms if part.geom_type in ("Polygon", "MultiPolygon")])
        except (AttributeError, IndexError, KeyError, TypeError, shapely.errors.GEOSException) as exc:
            raise ValueError(f"Not a valid GeoJSON or WKT geometry: {exc}") from exc
        if parsed.is_empty or parsed.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(f"An AOI geometry must be a non-empty Polygon or MultiPolygon, not {parsed.geom_type}")
        west, south, east, north = parsed.bounds
        if east - west > 180:
            raise ValueError("AOIs crossing the antimeridian are not supported; split the area at 180 degrees")
        return cls(west=west, south=south, east=east, north=north, geometry_wkt=parsed.wkt)

    @classmethod
    def from_geojson(cls, source: "str | Path | Mapping[str, object]") -> "AOI":
        """An AOI from a GeoJSON file path, JSON text or parsed object (all features are merged)."""

        import json

        if isinstance(source, Mapping):
            return cls.from_geometry(source)
        text = str(source)
        if not text.lstrip().startswith("{"):
            text = Path(source).read_text(encoding="utf-8")
        return cls.from_geometry(json.loads(text))

    @classmethod
    def from_dict(cls, value: "Mapping[str, object]") -> "AOI":
        """Inverse of ``to_dict``: bounds, or a ``geometry`` GeoJSON object."""

        if value.get("geometry"):
            return cls.from_geometry(value["geometry"])  # type: ignore[arg-type]
        missing = [key for key in ("west", "south", "east", "north") if key not in value]
        if missing:
            raise ValueError(f"Missing AOI keys: {', '.join(missing)}")
        return cls(**{key: value[key] for key in ("west", "south", "east", "north")})  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"west": self.west, "south": self.south, "east": self.east, "north": self.north}
        if self.geometry_wkt is not None:
            from shapely.geometry import mapping

            result["geometry"] = mapping(self.shape())
        return result

    @property
    def has_geometry(self) -> bool:
        return self.geometry_wkt is not None

    def shape(self):
        """The AOI as a shapely geometry: its polygon, or its bounding box."""

        from shapely import wkt
        from shapely.geometry import box

        return wkt.loads(self.geometry_wkt) if self.geometry_wkt is not None else box(self.west, self.south, self.east, self.north)

    def as_extent(self) -> dict[str, float | str]:
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
    west, south, east, north = (float(value[key]) for key in required)
    return AOI(west, south, east, north).as_extent()
