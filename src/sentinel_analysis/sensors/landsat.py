"""Landsat 8/9 Collection 2 Level-2 surface temperature.

Unlike Sentinel-3 (a raw, unprojected scan requiring footprint
reconstruction - see ``sensors/sentinel3/georeference.py``), Landsat
Collection 2 Level-2 is already an ortho-rectified, projected
Cloud-Optimized GeoTIFF - the same shape as Sentinel-2, so reading it is a
single ``rasterio`` reproject per band (mirroring
``sensors/sentinel2.py``'s ``_read_band``), not a binning/aggregation step.

Scenes are discovered via Microsoft Planetary Computer's public STAC API
(no account needed to search) and read directly off Planetary Computer's
signed remote URLs (no local download step) - Collection 2 assets are
hosted on Azure Blob Storage behind a short-lived SAS token that the
``planetary-computer`` package's ``sign()`` produces.

Asset names, scale/offset and QA_PIXEL bit layout below were confirmed
against a real Planetary Computer STAC search response for the
``landsat-c2-l2`` collection (2026-09), not assumed from documentation.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import xarray as xr

from ..catalog import ProductRef
from ..config import AOI
from ..cube import AnalysisGrid
from ..metadata import apply_variable_contract
from ..stac import STACCatalog, STACItem
from .sentinel3.processing import to_celsius

PC_STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"
LANDSAT_COLLECTION = "landsat-c2-l2"
DEFAULT_PLATFORMS = ("landsat-8", "landsat-9")

# QA_PIXEL bit layout (USGS Collection 2, confirmed via a real STAC item's
# "classification:bitfields"): bit0 fill, bit1 dilated_cloud, bit2 cirrus,
# bit3 cloud, bit4 cloud_shadow. Reject a pixel if any of those are set.
_QA_PIXEL_REJECT_MASK = 0b0000000000011111


def _stac_item_to_product_ref(item: STACItem) -> ProductRef:
    """Map a Planetary Computer Landsat STAC item onto the provider-neutral
    ``ProductRef`` used by every other sensor's catalogue."""

    lwir = item.assets.get("lwir11")
    if lwir is None:
        raise ValueError(f"STAC item {item.item_id!r} has no 'lwir11' (thermal) asset")
    return ProductRef(
        product_id=item.item_id,
        name=item.item_id,
        product_type="L2SP",
        start_datetime=item.datetime,
        end_datetime=item.datetime,
        timeliness=None,
        online=True,
        download_url=lwir["href"],
        # `metadata["attributes"]["cloudCover"]` matches ProductRef.cloud_cover's
        # expected nested shape as-is, so select_product_refs() sorts these
        # exactly like CDSE-derived references without any change there.
        metadata={
            "attributes": {"cloudCover": item.properties.get("eo:cloud_cover")},
            "platform": item.properties.get("platform"),
            "assets": item.assets,
        },
    )


class LandsatCatalog:
    """Search Landsat Collection 2 Level-2 via Microsoft Planetary Computer."""

    def __init__(self, *, endpoint: str = PC_STAC_ENDPOINT, catalog: STACCatalog | None = None):
        self.catalog = catalog or STACCatalog(endpoint=endpoint)

    def search(
        self,
        aoi: AOI,
        start: str | date | datetime,
        end: str | date | datetime,
        *,
        platforms: tuple[str, ...] = DEFAULT_PLATFORMS,
        cloud_cover_max: float | None = None,
        limit: int = 100,
    ) -> list[ProductRef]:
        """Search Landsat Collection 2 Level-2 scenes intersecting an AOI/date range."""

        start = start if isinstance(start, str) else start.isoformat()
        end = end if isinstance(end, str) else end.isoformat()
        query: dict[str, Any] = {"platform": {"in": list(platforms)}}
        if cloud_cover_max is not None:
            query["eo:cloud_cover"] = {"lt": cloud_cover_max}
        aoi.as_extent()  # validates the bounds; raises ValueError if malformed
        items = self.catalog.search(
            collections=[LANDSAT_COLLECTION],
            bbox=[aoi.west, aoi.south, aoi.east, aoi.north],
            datetime_range=f"{start}/{end}",
            query=query,
            limit=limit,
        )
        return [_stac_item_to_product_ref(item) for item in items if "lwir11" in item.assets]


def _read_and_reproject(href: str, grid: AnalysisGrid, *, categorical: bool, source_nodata: float | None):
    import rasterio
    from rasterio.warp import Resampling, reproject

    with rasterio.open(href) as source:
        data = source.read(1).astype(np.float64)
        if source_nodata is not None:
            data[data == source_nodata] = np.nan
        destination = np.full((grid.height, grid.width), np.nan, dtype=np.float64)
        reproject(
            source=data,
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.nearest if categorical else Resampling.bilinear,
        )
        return destination


def read_landsat_lst(
    product: ProductRef,
    grid: AnalysisGrid,
    *,
    observation_time: str | np.datetime64 | None = None,
) -> xr.Dataset:
    """Read one Landsat scene's surface temperature, quality-mask it with
    QA_PIXEL, and reproject it directly onto ``grid`` - a single pass, no
    intermediate native-resolution read, since a straight
    ``rasterio.warp.reproject`` already does source-to-target resampling.

    Reads straight off Planetary Computer's signed remote URL (GDAL's
    ``/vsicurl/` HTTP range-read support) - there is no local download step.
    """

    try:
        import planetary_computer as pc
    except ImportError as exc:
        raise RuntimeError("Install sentinel-analysis[landsat] to read Landsat scenes") from exc

    assets = product.metadata.get("assets", {})
    lwir_asset = assets.get("lwir11")
    if lwir_asset is None:
        raise ValueError(f"Product {product.product_id!r} has no 'lwir11' asset")
    qa_asset = assets.get("qa_pixel")

    lwir_bands = (lwir_asset.get("raster:bands") or [{}])[0]
    scale = float(lwir_bands.get("scale", 1.0))
    offset = float(lwir_bands.get("offset", 0.0))
    lwir_nodata = lwir_bands.get("nodata")

    lwir_href = pc.sign(lwir_asset["href"])
    raw = _read_and_reproject(lwir_href, grid, categorical=False, source_nodata=lwir_nodata)
    lst_kelvin = raw * scale + offset

    valid = np.isfinite(lst_kelvin)
    if qa_asset is not None:
        qa_bands = (qa_asset.get("raster:bands") or [{}])[0]
        qa_href = pc.sign(qa_asset["href"])
        qa_pixel = _read_and_reproject(qa_href, grid, categorical=True, source_nodata=qa_bands.get("nodata"))
        # Nearest-resampled NaNs (edge of the reprojected frame) carry no
        # flag information - treat as "no data" rather than "flagged clear".
        qa_int = np.where(np.isfinite(qa_pixel), qa_pixel, 1).astype(np.int64)
        valid &= (qa_int & _QA_PIXEL_REJECT_MASK) == 0

    lst_kelvin = np.where(valid, lst_kelvin, np.nan).astype(np.float32)
    valid = np.isfinite(lst_kelvin)

    time_value = observation_time or product.start_datetime
    time = np.datetime64(str(time_value).replace("Z", ""), "ns")

    dataset = xr.Dataset(
        {"landsat_lst": (("time", "y", "x"), lst_kelvin[np.newaxis, :, :])},
        coords={"time": [time], "y": grid.y, "x": grid.x},
        attrs={"crs": grid.crs, "grid_id": grid.grid_id, "analysis_shape": "regular_grid"},
    )
    dataset["landsat_lst"].attrs["units"] = "K"
    dataset["valid_mask"] = (("time", "y", "x"), valid[np.newaxis, :, :])
    dataset = apply_variable_contract(dataset, sensor="Landsat-8/9", product="L2SP", source=product.product_id)
    return to_celsius(dataset, lst_name="landsat_lst")
