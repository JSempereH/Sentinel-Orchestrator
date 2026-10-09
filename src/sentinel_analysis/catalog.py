"""Provider-neutral catalogue records shared by sensor adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Iterable, Mapping

import numpy as np

from .config import AOI

OVERPASSES = ("any", "day", "night")
# Local mean solar hours counted as a daytime overpass. Sentinel-3 SLSTR
# crosses at ~10:00 (day) and ~22:00 (night) local time; ECOSTRESS (ISS
# orbit) samples every hour of the day, and this window still separates
# sun-heated from night-time surfaces.
DAYTIME_SOLAR_HOURS = (6.0, 18.0)


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
        if self.metadata.get("aoi_coverage") is not None:
            return float(self.metadata["aoi_coverage"])
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
    if limit is not None and len(selected) > limit and selected and all(product.cloud_cover is None for product in selected):
        # Nothing ranks these products (radar, Sentinel-5P): sample the whole
        # period evenly instead of keeping its first days.
        return spread_in_time(selected, limit)
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


def local_solar_hour(timestamp: str, longitude: float) -> float:
    """Local mean solar hour (0-24) of a UTC timestamp at ``longitude``.

    Ignores the equation of time (at most ~16 minutes), which is irrelevant
    for telling a day pass from a night pass.
    """

    utc = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return (utc.hour + utc.minute / 60 + utc.second / 3600 + longitude / 15) % 24


def _overpass_time(product: ProductRef) -> str | None:
    """Middle of the acquisition: for a whole-orbit product its start is far from the overpass."""

    if not product.start_datetime:
        return None
    if not product.end_datetime:
        return product.start_datetime
    start = datetime.fromisoformat(product.start_datetime.replace("Z", "+00:00"))
    end = datetime.fromisoformat(product.end_datetime.replace("Z", "+00:00"))
    return (start + (end - start) / 2).isoformat()


def filter_overpass(products: Iterable[ProductRef], *, longitude: float, overpass: str) -> list[ProductRef]:
    """Keep only daytime or night-time acquisitions at ``longitude``.

    Downscaling learns a relation between surface temperature and optical
    predictors that only holds while the sun heats the surface; mixing night
    passes into it degrades every model. Products without a start time are
    dropped when filtering, since their overpass cannot be verified.
    """

    if overpass not in OVERPASSES:
        raise ValueError(f"overpass must be one of {OVERPASSES}")
    products = list(products)
    if overpass == "any":
        return products
    start, end = DAYTIME_SOLAR_HOURS
    kept = []
    for product in products:
        when = _overpass_time(product)
        if when is None:
            continue
        is_day = start <= local_solar_hour(when, longitude) < end
        if is_day == (overpass == "day"):
            kept.append(product)
    return kept


def footprint(product: ProductRef) -> Mapping[str, Any] | None:
    """The product's footprint as GeoJSON: OData ``GeoFootprint`` or the STAC item geometry."""

    geometry = product.metadata.get("GeoFootprint") or product.metadata.get("geometry")
    return geometry if isinstance(geometry, Mapping) and geometry.get("type") else None


def annotate_aoi_coverage(products: Iterable[ProductRef], aoi: AOI) -> list[ProductRef]:
    """Add ``metadata["aoi_coverage"]``: the fraction of the AOI the product covers.

    Computed from catalogue footprints, so it costs no download. Products of
    one pass (same start time: Sentinel-2 or ECOSTRESS tiles that are later
    mosaicked) share the coverage of their union. Footprints are clipped to
    the AOI in degrees before projecting, so whole-orbit footprints
    (Sentinel-5P) never have to be projected. Products without a footprint
    keep ``None``.
    """

    import shapely
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform

    from .cube import utm_crs

    products = list(products)
    area_box = aoi.shape()
    to_metres = Transformer.from_crs("EPSG:4326", utm_crs((aoi.west + aoi.east) / 2, (aoi.south + aoi.north) / 2), always_xy=True).transform
    aoi_area = transform(to_metres, area_box).area
    passes: dict[str, list] = {}
    for product in products:
        geometry = footprint(product)
        if geometry is None:
            continue
        try:
            clipped = shapely.make_valid(shape(geometry)).intersection(area_box)
        except (ValueError, TypeError, shapely.errors.GEOSException):
            continue
        passes.setdefault(product.start_datetime or product.product_id, []).append(clipped)
    coverage = {
        key: min(1.0, transform(to_metres, shapely.unary_union(parts)).area / aoi_area)
        for key, parts in passes.items()
    }
    return [
        replace(product, metadata={**product.metadata, "aoi_coverage": round(coverage[key], 4)})
        if (key := product.start_datetime or product.product_id) in coverage else product
        for product in products
    ]


def spread_in_time(products: Iterable[ProductRef], limit: int) -> list[ProductRef]:
    """``limit`` products evenly spaced through their time span, in time order."""

    ordered = sorted(products, key=lambda product: (product.start_datetime or "", product.product_id))
    if len(ordered) <= limit:
        return ordered
    picks = np.unique(np.round(np.linspace(0, len(ordered) - 1, limit)).astype(int))
    return [ordered[index] for index in picks]


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
