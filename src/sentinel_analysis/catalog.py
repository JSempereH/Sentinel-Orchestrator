"""Provider-neutral catalogue records shared by sensor adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ProductRef:
    """Provider-neutral description of one CDSE product."""

    product_id: str
    name: str
    product_type: str
    start_datetime: str | None
    end_datetime: str | None
    timeliness: str | None
    online: bool | None
    download_url: str
    metadata: Mapping[str, Any]

    @property
    def is_ntc(self) -> bool:
        return (self.timeliness or "").upper() in {"NT", "NTC"} or "_NT_" in self.name

    @property
    def is_nrt(self) -> bool:
        return (self.timeliness or "").upper() in {"NR", "NRT"} or "_NR_" in self.name

    @property
    def cloud_cover(self) -> float | None:
        value = self.metadata.get("attributes", {}).get("cloudCover")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def coverage(self) -> float | None:
        for key in ("coverage", "footprintCoverage", "areaCoverage"):
            value = self.metadata.get("attributes", {}).get(key, self.metadata.get(key))
            try:
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                continue
        return None


def select_product_refs(
    products: Iterable[ProductRef],
    *,
    limit: int | None = None,
    max_cloud_cover: float | None = None,
) -> list[ProductRef]:
    """Rank products by usable quality with deterministic tie breakers."""

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    selected = list(products)
    if max_cloud_cover is not None:
        selected = [product for product in selected if product.cloud_cover is None or product.cloud_cover <= max_cloud_cover]
    selected.sort(
        key=lambda product: (
            not bool(product.online),
            product.cloud_cover is None,
            product.cloud_cover if product.cloud_cover is not None else float("inf"),
            -(product.coverage if product.coverage is not None else -1.0),
            product.start_datetime or "",
            product.product_id,
        )
    )
    return selected[:limit] if limit is not None else selected


def _attributes(item: Mapping[str, Any]) -> dict[str, Any]:
    attributes = item.get("Attributes", {})
    if isinstance(attributes, Mapping):
        return dict(attributes)
    result: dict[str, Any] = {}
    for attribute in attributes or []:
        if isinstance(attribute, Mapping) and "Name" in attribute:
            result[str(attribute["Name"])] = attribute.get("Value")
    return result


def _product_ref(
    item: Mapping[str, Any],
    download_url: str,
    *,
    default_product_type: str = "unknown",
) -> ProductRef:
    attributes = _attributes(item)
    product_id = str(item.get("Id") or item.get("id"))
    name = str(item.get("Name") or item.get("name") or product_id)
    return ProductRef(
        product_id=product_id,
        name=name,
        product_type=str(attributes.get("productType", default_product_type)),
        start_datetime=(item.get("ContentDate") or {}).get("Start"),
        end_datetime=(item.get("ContentDate") or {}).get("End"),
        timeliness=attributes.get("timeliness"),
        online=item.get("Online"),
        download_url=f"{download_url.rstrip('/')}({product_id})/$value",
        metadata={**dict(item), "attributes": attributes},
    )
