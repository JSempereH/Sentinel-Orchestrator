"""Catalog and download access for Sentinel-3 products in CDSE."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Mapping

import requests

from ...catalog import ProductRef, _product_ref
from ...config import AOI, ClientConfig
from ...http import http_session


PRODUCT_TYPE = "SL_2_LST___"
COLLECTION_NAME = "SENTINEL-3"


def _iso_datetime(value: str | date | datetime, *, end: bool = False) -> str:
    date_only = isinstance(value, date) and not isinstance(value, datetime)
    if isinstance(value, datetime):
        result = value
        if result.tzinfo is None:
            result = result.replace(tzinfo=timezone.utc)
    elif isinstance(value, date):
        result = datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)
    else:
        date_only = "T" not in value and " " not in value
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid ISO date or datetime: {value!r}") from exc
        if result.tzinfo is None:
            result = result.replace(tzinfo=timezone.utc)
    if end and date_only:
        result = result.replace(hour=23, minute=59, second=59, microsecond=999999)
    return result.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ProductQuery:
    """Validated query for Sentinel-3 SLSTR L2 LST products."""

    aoi: AOI | Mapping[str, float]
    start: str | date | datetime
    end: str | date | datetime
    timeliness: str | None = "NTC"
    platform: str | None = None
    online_only: bool = True

    def normalized_aoi(self) -> AOI:
        return self.aoi if isinstance(self.aoi, AOI) else AOI.from_dict(self.aoi)

    def normalized_dates(self) -> tuple[str, str]:
        start = _iso_datetime(self.start)
        end = _iso_datetime(self.end, end=True)
        if start >= end:
            raise ValueError("Query start must be before query end")
        return start, end


class CDSECatalog:
    """Search the official CDSE OData catalogue without downloading products."""

    def __init__(self, config: ClientConfig | None = None, *, session: requests.Session | None = None):
        self.config = config
        self.session = session or http_session()
        self.catalog_url = (
            config.catalog_url
            if config is not None
            else "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        )
        self.download_url = (
            config.download_url
            if config is not None
            else "https://download.dataspace.copernicus.eu/odata/v1/Products"
        )

    def search(self, query: ProductQuery, *, limit: int = 1000) -> list[ProductRef]:
        """Return matching L2 LST products, filtering timeliness locally."""

        aoi = query.normalized_aoi()
        aoi.as_extent()
        start, end = query.normalized_dates()
        filters = [
            f"Collection/Name eq '{COLLECTION_NAME}'",
            (
                "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq "
                f"'productType' and att/OData.CSC.StringAttribute/Value eq '{PRODUCT_TYPE}')"
            ),
            f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi.as_wkt()}')",
            f"ContentDate/Start lt {end}",
            f"ContentDate/End gt {start}",
        ]
        if query.online_only:
            filters.append("Online eq true")
        if query.platform:
            platform = query.platform.upper().replace("SENTINEL-3", "S3")
            serial = platform[2:] if platform.startswith("S3") else platform
            filters.append(
                "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq "
                f"'platformSerialIdentifier' and att/OData.CSC.StringAttribute/Value eq '{serial}')"
            )

        results: list[ProductRef] = []
        url: str | None = self.catalog_url
        params: dict[str, Any] | None = {
            "$filter": " and ".join(filters),
            # CDSE OData accepts one expand value per request; Attributes is
            # enough for product type, timeliness and platform filtering.
            "$expand": "Attributes",
            "$orderby": "ContentDate/Start asc",
            "$top": min(limit, 1000),
        }
        while url and len(results) < limit:
            response = self.session.get(url, params=params, timeout=120)
            response.raise_for_status()
            payload = response.json()
            for item in payload.get("value", []):
                product = _product_ref(item, self.download_url)
                if self._matches_timeliness(product, query.timeliness):
                    results.append(product)
                    if len(results) >= limit:
                        break
            url = payload.get("@odata.nextLink")
            params = None
        return results

    @staticmethod
    def _matches_timeliness(product: ProductRef, timeliness: str | None) -> bool:
        if timeliness is None or product.timeliness is None:
            if timeliness is None:
                return True
            marker = "_NT_" if timeliness.upper() == "NTC" else "_NR_"
            return marker in product.name
        expected = {"NTC": {"NT", "NTC"}, "NRT": {"NR", "NRT"}}
        return product.timeliness.upper() in expected.get(timeliness.upper(), {timeliness.upper()})

