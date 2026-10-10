"""Windowed reads of cloud-optimized rasters straight onto an analysis grid.

Reading a whole scene and then reprojecting it (``source.read(1)`` followed by
``rasterio.warp.reproject``) materializes the full native-resolution array in
memory - ~1.7 GB for one Sentinel-1 RTC band - even when the analysis grid is
a small city at 100 m. A ``WarpedVRT`` instead lets GDAL fetch only the blocks
(and the overview level) that intersect the target grid.
"""

from __future__ import annotations

import time
from typing import Any, Mapping
from urllib.parse import urlparse

import numpy as np

from ..catalog import ProductRef
from ..cube import AnalysisGrid
from ..stac import STACItem

_RESAMPLING = ("nearest", "bilinear", "average")


def sign_href(href: str) -> str:
    """Sign Planetary Computer blob URLs; return any other href unchanged."""

    if not urlparse(href).netloc.endswith("blob.core.windows.net"):
        return href
    try:
        import planetary_computer as pc
    except ImportError as exc:
        raise RuntimeError("Install citycube[landsat] to read Planetary Computer assets") from exc
    return pc.sign(href)


def read_cog_to_grid(
    href: str,
    grid: AnalysisGrid,
    *,
    resampling: str = "bilinear",
    nodata: float | None = None,
    band: int = 1,
    attempts: int = 3,
    retry_delay_s: float = 2.0,
    use_overviews: bool = False,
) -> np.ndarray:
    """Read one raster band resampled onto ``grid`` as float32, NaN outside data.

    ``nodata`` overrides the file's own nodata value (STAC ``raster:bands``
    metadata is sometimes more reliable than the GeoTIFF tag).

    With ``use_overviews`` a projected source is read from its coarsest
    overview that is still at least as fine as ``grid`` (80 m for 10 m bands
    on a 100 m grid), which transfers about a third of the bytes. Off by
    default: on a real Sentinel-2 L2A scene over Berlin at 100 m the overview
    read correlated only 0.95 with the full-resolution average (NDVI 95th
    percentile difference 0.086), too large a change for predictors.
    """

    if resampling not in _RESAMPLING:
        raise ValueError(f"resampling must be one of {_RESAMPLING}")
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.vrt import WarpedVRT
    except ImportError as exc:
        raise RuntimeError("Install citycube[optical] to read cloud-optimized rasters") from exc

    from rasterio.errors import RasterioIOError

    remote = urlparse(href).scheme in {"http", "https"}
    for attempt in range(attempts if remote else 1):
        try:
            # GDAL retries transient HTTP errors itself; the outer loop also
            # re-opens the dataset, which survives a dropped connection.
            level = _overview_level(href, grid) if use_overviews else None
            open_kwargs = {"overview_level": level} if level is not None else {}
            with rasterio.Env(GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2"), rasterio.open(href, **open_kwargs) as source:
                source_nodata = nodata if nodata is not None else source.nodata
                with WarpedVRT(
                    source,
                    crs=grid.crs,
                    transform=grid.transform,
                    width=grid.width,
                    height=grid.height,
                    resampling=getattr(Resampling, resampling),
                    src_nodata=source_nodata,
                    nodata=np.nan,
                    dtype="float32",
                ) as warped:
                    return warped.read(band)
        except RasterioIOError:
            if attempt == attempts - 1 or not remote:
                raise
            time.sleep(retry_delay_s * 2**attempt)
    raise AssertionError("unreachable")


def _overview_level(href: str, grid: AnalysisGrid) -> int | None:
    """Index of the coarsest overview no coarser than ``grid``, or None to read full resolution."""

    import rasterio

    with rasterio.open(href) as source:
        if source.crs is None or not source.crs.is_projected:
            return None
        resolution = abs(source.res[0])
        factors = source.overviews(1)
    usable = [index for index, factor in enumerate(factors) if resolution * factor <= grid.resolution_m]
    return usable[-1] if usable else None


def asset_scale_offset(asset: Mapping[str, Any]) -> tuple[float, float, float | None]:
    """Return ``(scale, offset, nodata)`` from a STAC asset's ``raster:bands``."""

    bands = asset.get("raster:bands") or [{}]
    first = bands[0] if bands else {}
    return float(first.get("scale", 1.0)), float(first.get("offset", 0.0)), first.get("nodata")


def stac_product_ref(item: STACItem, *, product_type: str) -> ProductRef:
    """Map a STAC item onto the provider-neutral ``ProductRef``.

    ``metadata["attributes"]["cloudCover"]`` matches the nested shape
    ``ProductRef.cloud_cover`` reads, so ``select_product_refs`` ranks these
    exactly like CDSE OData results.
    """

    return ProductRef(
        product_id=item.item_id,
        name=item.item_id,
        product_type=product_type,
        start_datetime=item.datetime,
        end_datetime=item.datetime,
        timeliness=None,
        online=True,
        download_url="",
        metadata={
            "attributes": {"cloudCover": item.properties.get("eo:cloud_cover")},
            "platform": item.properties.get("platform"),
            "properties": dict(item.properties),
            "assets": item.assets,
            "geometry": item.raw.get("geometry"),
        },
    )


def observation_time(product: ProductRef) -> np.datetime64:
    """Acquisition time of a STAC-derived product as naive UTC ``datetime64[ns]``."""

    if not product.start_datetime:
        raise ValueError(f"Product {product.product_id!r} has no acquisition datetime")
    return np.datetime64(str(product.start_datetime).replace("Z", "").split("+")[0], "ns")
