"""Small dependency-free STAC client for cloud-native catalogue search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


from .http import http_session


@dataclass(frozen=True)
class STACItem:
    """Stable subset of a STAC Item plus its complete asset mapping."""

    item_id: str
    collections: tuple[str, ...]
    datetime: str | None
    bbox: tuple[float, ...] | None
    assets: Mapping[str, Mapping[str, Any]]
    properties: Mapping[str, Any]
    raw: Mapping[str, Any]


class _STACSession(Protocol):
    def post(self, url: str, *, json: Mapping[str, Any], timeout: int) -> Any: ...


class STACCatalog:
    """Search a STAC API using the standard POST /search endpoint."""

    def __init__(self, endpoint: str = "https://stac.dataspace.copernicus.eu/v1", *, session: _STACSession | None = None):
        self.endpoint = endpoint.rstrip("/")
        self.session = session or http_session()

    def search(
        self,
        *,
        collections: Sequence[str] | None = None,
        bbox: Sequence[float] | None = None,
        datetime_range: str | None = None,
        query: Mapping[str, Any] | None = None,
        limit: int = 100,
    ) -> list[STACItem]:
        """Search items with spatial, temporal and property constraints."""

        payload: dict[str, Any] = {"limit": min(limit, 1000)}
        if collections:
            payload["collections"] = list(collections)
        if bbox:
            if len(bbox) not in {4, 6}:
                raise ValueError("STAC bbox must contain 4 or 6 coordinates")
            payload["bbox"] = list(bbox)
        if datetime_range:
            payload["datetime"] = self._normalize_datetime_range(datetime_range)
        if query:
            payload["query"] = dict(query)

        items: list[STACItem] = []
        url = f"{self.endpoint}/search"
        while url and len(items) < limit:
            response = self.session.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            for raw in result.get("features", []):
                raw_collections = raw.get("collections")
                if raw_collections is None and raw.get("collection"):
                    raw_collections = (raw["collection"],)
                items.append(STACItem(
                    item_id=str(raw["id"]),
                    collections=tuple(raw_collections or ()),
                    datetime=raw.get("properties", {}).get("datetime"),
                    bbox=tuple(raw["bbox"]) if raw.get("bbox") else None,
                    assets=raw.get("assets", {}),
                    properties=raw.get("properties", {}),
                    raw=raw,
                ))
                if len(items) >= limit:
                    break
            next_link = next((link for link in result.get("links", []) if link.get("rel") == "next"), None)
            url = next_link.get("href") if next_link else ""
            payload = dict(next_link.get("body", {})) if next_link else {}
        return items

    @staticmethod
    def _normalize_datetime_range(value: str) -> str:
        parts = value.split("/")
        if len(parts) != 2:
            return value
        start, end = parts
        if len(start) == 10:
            start += "T00:00:00Z"
        if len(end) == 10:
            end += "T23:59:59Z"
        return f"{start}/{end}"
