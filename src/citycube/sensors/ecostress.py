"""ECOSTRESS L2T LSTE (tiled land surface temperature), ~70m.

Like Landsat (``sensors/landsat.py``), the L2T product is already tiled and
ortho-rectified (on the same MGRS tiling scheme Sentinel-2 uses) - reading
it is a ``rasterio`` reproject per band, not swath footprint reconstruction.

Two real differences from Landsat, both confirmed against a real search
against NASA's CMR-STAC bridge for LP DAAC (2026-09), not assumed:

1. **Asset keys are unusable.** This particular STAC bridge emits asset
   dict keys as the full granule-id-plus-band-suffix string (e.g.
   ``"003/ECOv003_L2T_LSTE_..._LST"``) instead of a clean ``"LST"`` key, and
   provides no ``raster:bands`` scale/offset metadata at all (unlike
   Planetary Computer for Landsat). Assets here are found by matching the
   *suffix* of the asset's ``href`` (``_LST.tif``, ``_cloud.tif``) instead
   of an exact key lookup, and scale/offset are read from the GeoTIFF's own
   embedded GDAL metadata at open time, falling back to
   ``_FALLBACK_LST_SCALE``/``_FALLBACK_LST_OFFSET`` (identity: 1.0, 0.0) if
   a file reports no scale at all.

   An earlier version of this fallback used the published ECOSTRESS ATBD
   digital-number formula (scale 0.02) instead of identity - that formula
   is for the older Swath LSTE product's raw counts. Real L2T LSTE tiles
   (confirmed 2026-09 against a live NASA CMR-STAC search + authenticated
   read) embed scale=1.0/offset=0.0 and store LST directly in Kelvin
   already; applying *0.02 on top produced ~5 K ("-267 degC") results. Note
   rasterio/GDAL cannot distinguish "file explicitly says scale=1.0" from
   "file has no scale tag at all" - both report ``(1.0,)`` - so this
   fallback can never really be observed to fire for a real L2T file; it
   exists only as a defensive default, and identity is the correct one.
2. **Auth is a real NASA Earthdata Cloud OAuth flow, not a signed URL.**
   Confirmed by hitting a real asset URL and observing an HTTP redirect to
   ``.../login?code=...`` - ``~/.netrc`` Basic Auth (which works for
   "classic" Earthdata HTTPS downloads) does not complete this. The
   supported non-interactive path is a personal Earthdata "user token"
   (generate once at https://urs.earthdata.nasa.gov/profile -> Generate
   Token) sent as ``Authorization: Bearer <token>``.

   This is fetched with a plain ``requests.get()`` into a temp file rather
   than opened remotely via GDAL's ``vsicurl`` + ``GDAL_HTTP_BEARER``
   (rasterio reads the whole band into memory here regardless, so streaming
   reads buy nothing). Confirmed against real data (2026-09) that vsicurl
   does not work cleanly for this specific host: curl auto-attaches any
   ``~/.netrc`` entry for ``urs.earthdata.nasa.gov`` during the OAuth
   redirect and gets stuck in an infinite ``authorize`` <-> ``login?code=``
   loop instead of using the Bearer header, and even with netrc disabled
   GDAL's own metadata-sidecar probe (a HEAD for a nonexistent
   ``<file>.tif.xml``) can abort the whole open. A single explicit
   ``requests`` call with the Bearer header sidesteps both. Set
   ``EARTHDATA_BEARER_TOKEN`` to use it; no other citycube provider
   needs this, so it is read directly from the environment here rather than
   added to ``ClientConfig``.

A single ECOSTRESS overpass can cover an AOI with more than one MGRS tile
(two items sharing the same acquisition time) - ``execute()`` mosaics same-
time tiles with the existing ``_mosaic_temporal_tiles`` helper, the same
utility Sentinel-2 already uses for the identical situation.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any

import numpy as np
import xarray as xr

from ..catalog import ProductRef
from ..config import AOI
from ..cube import AnalysisGrid
from ..metadata import apply_variable_contract
from ..stac import STACCatalog, STACItem

CMR_STAC_ENDPOINT = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
ECOSTRESS_COLLECTION = "ECO_L2T_LSTE_003"

# Defensive default if a GeoTIFF reports no scale/offset at all (see module
# docstring, point 1) - real L2T LSTE tiles store LST directly in Kelvin, so
# identity is correct, not the older Swath product's ATBD digital-number
# formula.
_FALLBACK_LST_SCALE = 1.0
_FALLBACK_LST_OFFSET = 0.0


def _find_asset(item: STACItem, suffix: str) -> dict[str, Any] | None:
    """Find an asset by the suffix of its href, not its (unpredictable) key."""

    return next((dict(asset) for asset in item.assets.values() if str(asset.get("href", "")).endswith(suffix)), None)


def _stac_item_to_product_ref(item: STACItem) -> ProductRef:
    lst_asset = _find_asset(item, "_LST.tif")
    if lst_asset is None:
        raise ValueError(f"STAC item {item.item_id!r} has no '*_LST.tif' asset")
    return ProductRef(
        product_id=item.item_id,
        name=item.item_id,
        product_type="L2T_LSTE",
        start_datetime=item.properties.get("start_datetime", item.datetime),
        end_datetime=item.properties.get("end_datetime", item.datetime),
        timeliness=None,
        online=True,
        download_url=lst_asset["href"],
        metadata={"attributes": {}, "assets": {a.get("href", ""): a for a in item.assets.values()}, "geometry": item.raw.get("geometry")},
    )


class EcostressCatalog:
    """Search ECOSTRESS L2T LSTE via NASA's CMR-STAC bridge for LP DAAC."""

    def __init__(self, *, endpoint: str = CMR_STAC_ENDPOINT, catalog: STACCatalog | None = None):
        self.catalog = catalog or STACCatalog(endpoint=endpoint)

    def search(
        self, aoi: AOI, start: str | date | datetime, end: str | date | datetime, *, limit: int = 100,
    ) -> list[ProductRef]:
        """Search ECOSTRESS L2T LSTE scenes intersecting an AOI/date range.

        ECOSTRESS rides the ISS (irregular revisit, not a predictable daily
        pass), so a query over an arbitrary past window can genuinely match
        nothing for a small AOI even though the collection has data - widen
        the date range rather than assume the search is broken.
        """

        start = start if isinstance(start, str) else start.isoformat()
        end = end if isinstance(end, str) else end.isoformat()
        aoi.as_extent()  # validates the bounds; raises ValueError if malformed
        items = self.catalog.search(
            collections=[ECOSTRESS_COLLECTION],
            bbox=[aoi.west, aoi.south, aoi.east, aoi.north],
            datetime_range=f"{start}/{end}",
            limit=limit,
        )
        refs = []
        for item in items:
            try:
                refs.append(_stac_item_to_product_ref(item))
            except ValueError:
                continue  # item without a usable LST asset (e.g. a non-data granule)
        return refs


def _read_and_reproject(href: str, grid: AnalysisGrid, *, categorical: bool):
    """Read band 1 of `href` and reproject it onto `grid`. Returns
    (reprojected_array, source_scale, source_offset) - scale/offset come
    from the file's own embedded GDAL metadata when present.

    A remote `href` is fetched with a plain HTTP GET into a temp file rather
    than opened directly through GDAL's vsicurl - see the module docstring's
    point 2 for why vsicurl does not work cleanly against this specific
    host. A local path (as used by tests, or a caller with an
    already-downloaded product) is opened as-is."""

    import contextlib
    import tempfile

    import rasterio
    from rasterio.warp import Resampling, reproject

    with contextlib.ExitStack() as stack:
        if href.startswith("http://") or href.startswith("https://"):
            import requests

            token = os.getenv("EARTHDATA_BEARER_TOKEN")
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            response = requests.get(href, headers=headers, timeout=120)
            response.raise_for_status()
            tmp = stack.enter_context(tempfile.NamedTemporaryFile(suffix=".tif"))
            tmp.write(response.content)
            tmp.flush()
            path = tmp.name
        else:
            path = href

        source = stack.enter_context(rasterio.open(path))
        scale = source.scales[0] if source.scales and source.scales[0] not in (None, 1.0) else None
        offset = source.offsets[0] if source.offsets and source.offsets[0] is not None else None
        data = source.read(1).astype(np.float64)
        if source.nodata is not None:
            data[data == source.nodata] = np.nan
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
        return destination, scale, offset


def read_ecostress_lst(
    product: ProductRef,
    grid: AnalysisGrid,
    *,
    observation_time: str | np.datetime64 | None = None,
) -> xr.Dataset:
    """Read one ECOSTRESS L2T LSTE tile's surface temperature, mask it with
    the product's own `cloud` band, and reproject it directly onto `grid`.

    Requires ``EARTHDATA_BEARER_TOKEN`` in the environment (a personal
    Earthdata user token - see module docstring); raises a clear error
    otherwise rather than failing deep inside an HTTP redirect loop.
    """

    if not os.getenv("EARTHDATA_BEARER_TOKEN"):
        raise RuntimeError(
            "EARTHDATA_BEARER_TOKEN is not set. Generate a personal token at "
            "https://urs.earthdata.nasa.gov/profile (Generate Token) and set it "
            "in the environment - NASA's Earthdata Cloud requires it; a "
            "~/.netrc entry alone is not sufficient for this collection."
        )

    assets = product.metadata.get("assets", {})
    lst_asset = next((a for a in assets.values() if str(a.get("href", "")).endswith("_LST.tif")), None)
    if lst_asset is None:
        raise ValueError(f"Product {product.product_id!r} has no '*_LST.tif' asset")
    cloud_asset = next((a for a in assets.values() if str(a.get("href", "")).endswith("_cloud.tif")), None)

    raw, scale, offset = _read_and_reproject(lst_asset["href"], grid, categorical=False)
    scale = _FALLBACK_LST_SCALE if scale is None else scale
    offset = _FALLBACK_LST_OFFSET if offset is None else offset
    lst_kelvin = raw * scale + offset

    valid = np.isfinite(lst_kelvin)
    if cloud_asset is not None:
        cloud_mask, _, _ = _read_and_reproject(cloud_asset["href"], grid, categorical=True)
        # Nearest-resampled NaNs (edge of the reprojected frame) carry no
        # flag information - treat as "no data" rather than "flagged clear".
        cloud_int = np.where(np.isfinite(cloud_mask), cloud_mask, 1).astype(np.int64)
        valid &= cloud_int == 0

    lst_kelvin = np.where(valid, lst_kelvin, np.nan).astype(np.float32)
    valid = np.isfinite(lst_kelvin)

    time_value = observation_time or product.start_datetime
    time = np.datetime64(str(time_value).replace("Z", ""), "ns")

    dataset = xr.Dataset(
        {"ecostress_lst": (("time", "y", "x"), lst_kelvin[np.newaxis, :, :])},
        coords={"time": [time], "y": grid.y, "x": grid.x},
        attrs={"crs": grid.crs, "grid_id": grid.grid_id, "analysis_shape": "regular_grid"},
    )
    dataset["ecostress_lst"].attrs["units"] = "K"
    dataset["valid_mask"] = (("time", "y", "x"), valid[np.newaxis, :, :])
    dataset = apply_variable_contract(dataset, sensor="ECOSTRESS", product="L2T_LSTE", source=product.product_id)

    from .sentinel3.processing import to_celsius
    return to_celsius(dataset, lst_name="ecostress_lst")
