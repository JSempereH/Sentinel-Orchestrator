"""Sharpening Landsat land surface temperature with Sentinel-2.

Landsat 8/9 measure thermal radiance at about 100 m; the Collection 2
surface temperature product is delivered at 30 m only after cubic
convolution resampling, so its 30 m cells carry no 30 m detail. The same
per-scene protocol used for Sentinel-3 recovers that detail from Sentinel-2:
the 30 m temperature is averaged in ``factor x factor`` blocks (90 m by
default, close to the native resolution), a model relates it to the
Sentinel-2 indices averaged the same way, and predicts from the indices at
30 m; the residual correction makes each block average back to Landsat.

This is how thermal sharpening is usually applied to Landsat with
Sentinel-2 (Gao et al. 2012; Onacillova et al. 2022, Remote Sensing
14:4076). The defaults, a linear model per scene with area-to-point kriging
of the residual, scored best in ``scripts/validate_landsat_sharpening.py``
(``docs/downscaling.md``); tree ensembles fit the coarse scale better but
transfer worse to the fine one.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from .per_scene import downscale_per_scene

# A block counts as observed when at least this share of its fine cells is.
_MIN_BLOCK_COVERAGE = 2 / 3


def block_aggregate(dataset: xr.Dataset, factor: int) -> xr.Dataset:
    """Average ``factor x factor`` blocks of fine cells (trailing rows and columns that do not fill a block are dropped).

    A block is NaN unless at least two thirds of its cells are finite;
    boolean variables become the share of true cells. Keeps ``attrs``.
    """

    if factor < 2:
        raise ValueError("factor must be at least 2")
    ny, nx = dataset.sizes["y"] // factor * factor, dataset.sizes["x"] // factor * factor
    if ny == 0 or nx == 0:
        raise ValueError(f"the grid is smaller than one {factor} x {factor} block")
    trimmed = dataset.isel(y=slice(0, ny), x=slice(0, nx))
    # xarray adds the reductions to Coarsen objects at runtime.
    mean: xr.Dataset = trimmed.coarsen(y=factor, x=factor).mean(skipna=True)  # type: ignore[attr-defined]
    finite: xr.Dataset = trimmed.notnull().coarsen(y=factor, x=factor).mean()  # type: ignore[attr-defined]
    for name in mean.data_vars:
        if trimmed[name].dtype != bool and {"y", "x"} <= set(trimmed[name].dims):
            mean[name] = mean[name].where(finite[name] >= _MIN_BLOCK_COVERAGE)
        mean[name].attrs = dict(trimmed[name].attrs)
    mean.attrs.update(dataset.attrs)
    return mean


def match_sentinel2(landsat_times: np.ndarray, sentinel2: xr.Dataset, *, max_gap: np.timedelta64, min_valid: float = 0.5) -> np.ndarray:
    """For each Landsat time, the closest Sentinel-2 time within ``max_gap`` with enough valid NDVI (NaT if none)."""

    candidates = sentinel2.time.values
    valid = sentinel2["NDVI"].notnull().mean(["y", "x"]).values if "NDVI" in sentinel2 else np.ones(candidates.size)
    usable = candidates[valid >= min_valid]
    matched = np.full(len(landsat_times), np.datetime64("NaT", "ns"))
    for index, time in enumerate(np.asarray(landsat_times, dtype="datetime64[ns]")):
        if usable.size == 0:
            continue
        gaps = np.abs(usable - time)
        best = int(np.argmin(gaps))
        if gaps[best] <= max_gap:
            matched[index] = usable[best]
    return matched


def sharpen_landsat(
    landsat: xr.Dataset,
    sentinel2: xr.Dataset,
    *,
    terrain: xr.Dataset | None = None,
    factor: int = 3,
    max_gap: np.timedelta64 = np.timedelta64(3, "D"),
    predictors: Sequence[str] | None = None,
    model: str = "linear",
    correction: str = "atpk",
    min_samples: int = 30,
    model_options: Mapping[str, Any] | None = None,
    target: str = "landsat_lst",
) -> xr.Dataset:
    """Sharpen Landsat LST from its native ~100 m to the 30 m grid it is delivered on.

    ``landsat`` and ``sentinel2`` are the per-sensor cubes of a workflow
    run at 30 m (``result.predictors["landsat"]`` and
    ``result.predictors["sentinel2"]``), on the same grid; ``terrain``
    optionally adds elevation and slope. Each Landsat scene is paired with
    the closest Sentinel-2 acquisition within ``max_gap``; scenes without
    one, or with fewer than ``min_samples`` complete blocks, are skipped
    and listed in ``downscaling_skipped_scenes``. Returns the same variables
    as ``downscale_per_scene`` (``lst_downscaled`` at 30 m, in degC).
    """

    if target not in landsat:
        raise ValueError(f"landsat has no {target!r} variable")
    if not (np.array_equal(landsat.y.values, sentinel2.y.values) and np.array_equal(landsat.x.values, sentinel2.x.values)):
        raise ValueError("landsat and sentinel2 must be on the same grid (one workflow run, same resolution)")
    names = list(predictors) if predictors else [name for name in ("NDVI", "NDBI", "EVI", "NDMI") if name in sentinel2]
    optical = [name for name in names if name in sentinel2]
    static = [name for name in names if name not in sentinel2 and terrain is not None and name in terrain]
    if not predictors and terrain is not None:
        static = [name for name in ("elevation", "slope") if name in terrain]
        names = optical + static

    matched = match_sentinel2(landsat.time.values, sentinel2, max_gap=max_gap)
    coarse_target = block_aggregate(landsat[[target]], factor)
    layers = []
    for index, time in enumerate(landsat.time.values):
        if np.isnat(matched[index]):
            layers.append(xr.full_like(block_aggregate(sentinel2[optical].isel(time=0, drop=True), factor), np.nan).expand_dims(time=[time]))
        else:
            layers.append(block_aggregate(sentinel2[optical].sel(time=matched[index], drop=True), factor).expand_dims(time=[time]))
    coarse = xr.merge([coarse_target, xr.concat(layers, dim="time")])
    if static and terrain is not None:
        aggregated = block_aggregate(terrain[static].reindex(y=landsat.y, x=landsat.x, method="nearest", tolerance=1e-6), factor)
        coarse = coarse.assign({name: aggregated[name].broadcast_like(coarse[target]) for name in static})
    coarse["sentinel2_matched_time"] = ("time", matched)
    coarse.attrs.update({**landsat.attrs, "crs": landsat.attrs.get("crs") or sentinel2.attrs.get("crs")})
    coarse[target].attrs.setdefault("units", landsat[target].attrs.get("units", "degC"))

    sharpened = downscale_per_scene(
        coarse,
        sentinel2,
        target=target,
        predictors=names,
        model=model,
        min_samples=min_samples,
        predictor_sensor="sentinel2",
        terrain=terrain,
        model_options=model_options,
        correction=correction,
    )
    sharpened.attrs.update({
        "downscaling_source": "landsat",
        "downscaling_block_factor": factor,
        "downscaling_coarse_resolution_m": float(abs(coarse.x.values[1] - coarse.x.values[0])),
    })
    return sharpened
