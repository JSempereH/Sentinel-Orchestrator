"""Classical spatiotemporal fusion: STARFM and ESTARFM."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ..cube import CubeValidationError


def _make_windows(*arrays: np.ndarray, radius: int) -> tuple[np.ndarray, ...]:
    """Pad each 2D array with NaN and return its sliding window view,
    shape (H, W, 2*radius+1, 2*radius+1) - shared by `fuse_starfm` and
    `fuse_estarfm`. NaN padding (not e.g. edge-replication) so a window
    that overhangs the array border correctly excludes those cells via the
    ordinary finite-value check, rather than fabricating real-looking
    neighbors there."""

    from numpy.lib.stride_tricks import sliding_window_view

    pad = ((radius, radius), (radius, radius))
    window_shape = (2 * radius + 1, 2 * radius + 1)
    return tuple(sliding_window_view(np.pad(array, pad, constant_values=np.nan), window_shape) for array in arrays)


def _spatial_weight_grid(radius: int) -> np.ndarray:
    """1 + normalized distance from the window center - STARFM/ESTARFM's
    spatial term: the center pixel itself gets weight 1 (never zero), more
    distant candidates are progressively discounted."""

    offsets = np.arange(-radius, radius + 1)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    return 1.0 + np.sqrt(dy**2 + dx**2) / max(radius, 1)


def fuse_starfm(
    fine_t0: xr.DataArray,
    coarse_t0: xr.DataArray,
    coarse_t1: xr.DataArray,
    *,
    window_radius: int = 10,
    similarity_threshold_multiplier: float = 2.0,
) -> xr.Dataset:
    """STARFM (Gao et al. 2006, 10.1109/TGRS.2006.872081) spatiotemporal
    fusion: predict a fine-resolution map at t1 from a fine observation at
    t0 and the coarse change between t0 and t1.

    This is the single fine/coarse-pair form, the same simplified baseline
    most open STARFM implementations use - not the full multi-pair ensemble
    from the original paper. ESTARFM and FSDAF (`docs/downscaling.md`'s
    other named classical baselines) are not implemented; both are open
    follow-ups, and matter most for exactly the case this simplified form
    handles badly: large or heterogeneous change between t0 and t1 (e.g.
    real land-cover change, not just a temperature swing).

    For each fine pixel, candidate pixels in a `window_radius` window are
    weighted by spectral similarity to the center pixel (in `fine_t0`),
    temporal-change similarity (in `coarse_t1 - coarse_t0`), and spatial
    distance, combined as in the original paper: pixels are excluded from
    the average when they differ from the center pixel in `fine_t0` by more
    than `similarity_threshold_multiplier * std(fine_t0)` - a strategy to
    avoid contamination by a different land-cover class inside the window.
    The unfiltered pixels are then combined by inverse-distance weighting in
    this combined similarity space; the prediction itself is
    `fine_t0 + (coarse_t1 - coarse_t0)` for every candidate, averaged.

    `coarse_t0`/`coarse_t1` are reindexed onto `fine_t0`'s grid via
    nearest-neighbor if their coordinates differ - the usual STARFM
    precondition (coarse data resampled, not interpolated, onto the fine
    grid before the window search).

    This is O(window_radius**2) memory and time per fine pixel (vectorized
    across pixels, not a Python-level pixel loop) - a `window_radius` much
    above the default can be expensive for large grids; downsample the AOI
    or reduce it instead of assuming this scales like the other, O(1)
    downscalers here.
    """

    for name, data_array in (("fine_t0", fine_t0), ("coarse_t0", coarse_t0), ("coarse_t1", coarse_t1)):
        if not {"y", "x"}.issubset(data_array.dims):
            raise CubeValidationError(f"{name} must have y and x dimensions")
    coarse_t0 = coarse_t0.reindex(y=fine_t0.y, x=fine_t0.x, method="nearest")
    coarse_t1 = coarse_t1.reindex(y=fine_t0.y, x=fine_t0.x, method="nearest")

    fine = fine_t0.values.astype(np.float64)
    c0 = coarse_t0.values.astype(np.float64)
    c1 = coarse_t1.values.astype(np.float64)
    radius = window_radius
    epsilon = 1e-6
    threshold = similarity_threshold_multiplier * np.nanstd(fine)

    fine_windows, c0_windows, c1_windows = _make_windows(fine, c0, c1, radius=radius)

    center = fine[:, :, None, None]
    spectral_diff = np.abs(fine_windows - center) + epsilon
    temporal_diff = np.abs(c1_windows - c0_windows) + epsilon
    spatial_weight = _spatial_weight_grid(radius)

    similar = np.abs(fine_windows - center) <= threshold
    finite = np.isfinite(fine_windows) & np.isfinite(c0_windows) & np.isfinite(c1_windows)
    valid = similar & finite

    combined_distance = spectral_diff * temporal_diff * spatial_weight
    weights = np.where(valid, 1.0 / combined_distance, 0.0)
    weight_sum = weights.sum(axis=(-2, -1))
    has_support = weight_sum > 0
    safe_weight_sum = np.where(has_support, weight_sum, 1.0)

    # np.where, not a bare NaN-propagating sum: an invalid (weight=0)
    # window cell can itself be NaN (padding, or masked by the similarity
    # threshold) - 0.0 * NaN is NaN, not 0, and would otherwise poison the
    # weighted sum even though that cell contributes zero weight.
    predicted_local = np.where(valid, fine_windows + (c1_windows - c0_windows), 0.0)
    prediction = (weights * predicted_local).sum(axis=(-2, -1)) / safe_weight_sum
    support = has_support & np.isfinite(fine)
    prediction = np.where(support, prediction, np.nan).astype(np.float32)

    coords = {"y": fine_t0.y, "x": fine_t0.x}
    result = xr.Dataset(
        {
            "lst_downscaled": xr.DataArray(prediction, dims=("y", "x"), coords=coords),
            "downscaled_support": xr.DataArray(support, dims=("y", "x"), coords=coords),
        },
        attrs={**fine_t0.attrs, "downscaled_is_modelled": True},
    )
    result.attrs.update({
        "downscaling_method": "starfm",
        "starfm_window_radius": window_radius,
        "starfm_similarity_threshold": float(threshold),
    })
    return result


def _local_linear_regression(
    x_windows: np.ndarray, y_windows: np.ndarray, valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pixel OLS slope/intercept/residual-variance of y ~ x over each
    pixel's own window of (x, y) pairs - the local fine-coarse conversion
    `fuse_estarfm` uses instead of `fuse_starfm`'s raw additive difference.
    A local *slope*, not just an offset, can differ between land-cover
    classes sharing one window - a raw difference cannot represent that.

    Falls back to identity (slope=1, intercept=0 - STARFM's own behavior)
    wherever a window has fewer than 3 valid points or no x variance to
    regress against; residual_var is NaN there (no fit to be uncertain
    about, not zero uncertainty)."""

    zero = np.zeros_like(x_windows)
    n = valid.sum(axis=(-2, -1))
    sum_x = np.where(valid, x_windows, zero).sum(axis=(-2, -1))
    sum_y = np.where(valid, y_windows, zero).sum(axis=(-2, -1))
    sum_xx = np.where(valid, x_windows * x_windows, zero).sum(axis=(-2, -1))
    sum_xy = np.where(valid, x_windows * y_windows, zero).sum(axis=(-2, -1))
    sum_yy = np.where(valid, y_windows * y_windows, zero).sum(axis=(-2, -1))

    enough = n >= 3
    safe_n = np.where(enough, n, 1)
    mean_x = sum_x / safe_n
    mean_y = sum_y / safe_n
    var_x = sum_xx / safe_n - mean_x**2
    cov_xy = sum_xy / safe_n - mean_x * mean_y

    has_variance = enough & (var_x > 1e-9)
    safe_var_x = np.where(has_variance, var_x, 1.0)
    slope = np.where(has_variance, cov_xy / safe_var_x, 1.0)
    intercept = np.where(has_variance, mean_y - slope * mean_x, 0.0)

    sse = sum_yy - intercept * sum_y - slope * sum_xy
    residual_var = np.where(has_variance, np.maximum(sse, 0.0) / np.maximum(n - 2, 1), np.nan)
    return slope, intercept, residual_var


def fuse_estarfm(
    fine_t0: xr.DataArray,
    coarse_t0: xr.DataArray,
    fine_t2: xr.DataArray,
    coarse_t2: xr.DataArray,
    coarse_t1: xr.DataArray,
    *,
    window_radius: int = 10,
    similarity_threshold_multiplier: float = 2.0,
) -> xr.Dataset:
    """ESTARFM (Zhu et al. 2010, 10.1109/TGRS.2010.2050822) spatiotemporal
    fusion: predict a fine map at t1 from two bracketing fine/coarse pairs
    (t0, t2) and the coarse observation at t1, instead of `fuse_starfm`'s
    single pair.

    The real difference from `fuse_starfm` - not just "one more pair" -
    is *how* each pair's coarse change becomes a fine-scale prediction.
    STARFM adds the raw coarse difference directly onto the fine
    observation, implicitly assuming a 1:1 fine/coarse relationship. This
    fits a local linear regression between fine and coarse values within
    each pixel's window instead (`_local_linear_regression`), separately
    for each pair, and uses that pair's own *slope* to convert its coarse
    change - representing exactly the case docs/downscaling.md flags
    single-pair STARFM as handling badly: a window straddling two surfaces
    whose fine/coarse relationship differs in slope, not just offset.

    The two pairs' resulting predictions are combined weighted by each
    pair's own local regression's residual variance - the pair whose
    window fit fine against coarse more consistently is trusted more, not
    averaged blindly. Candidate-pixel similarity (to keep a heterogeneous
    window from contaminating the regression itself) is judged against
    `fine_t0`, mirroring `fuse_starfm`'s convention.

    `coarse_t0`/`fine_t2`/`coarse_t2`/`coarse_t1` are reindexed onto
    `fine_t0`'s grid via nearest-neighbor if their coordinates differ, same
    as `fuse_starfm`. Same O(window_radius**2) memory/time caveat applies.
    """

    arrays = {"fine_t0": fine_t0, "coarse_t0": coarse_t0, "fine_t2": fine_t2, "coarse_t2": coarse_t2, "coarse_t1": coarse_t1}
    for name, data_array in arrays.items():
        if not {"y", "x"}.issubset(data_array.dims):
            raise CubeValidationError(f"{name} must have y and x dimensions")
    coarse_t0 = coarse_t0.reindex(y=fine_t0.y, x=fine_t0.x, method="nearest")
    fine_t2 = fine_t2.reindex(y=fine_t0.y, x=fine_t0.x, method="nearest")
    coarse_t2 = coarse_t2.reindex(y=fine_t0.y, x=fine_t0.x, method="nearest")
    coarse_t1 = coarse_t1.reindex(y=fine_t0.y, x=fine_t0.x, method="nearest")

    fine0 = fine_t0.values.astype(np.float64)
    c0 = coarse_t0.values.astype(np.float64)
    fine2 = fine_t2.values.astype(np.float64)
    c2 = coarse_t2.values.astype(np.float64)
    c1 = coarse_t1.values.astype(np.float64)
    radius = window_radius
    epsilon = 1e-6
    threshold = similarity_threshold_multiplier * np.nanstd(fine0)

    # c1 is only needed at each pixel's own location (the delta target),
    # never as a window of neighbors, so it is not windowed here.
    fine0_windows, c0_windows, fine2_windows, c2_windows = _make_windows(fine0, c0, fine2, c2, radius=radius)
    center = fine0[:, :, None, None]
    similar = np.abs(fine0_windows - center) <= threshold
    valid_t0 = similar & np.isfinite(fine0_windows) & np.isfinite(c0_windows)
    valid_t2 = similar & np.isfinite(fine2_windows) & np.isfinite(c2_windows)

    # Only the slope is used below: predicting a *delta* (coarse_t1 minus
    # coarse_t0/t2) from a local linear fit needs just its rate of change,
    # not the fit's absolute intercept.
    slope_t0, _intercept_t0, residual_var_t0 = _local_linear_regression(c0_windows, fine0_windows, valid_t0)
    slope_t2, _intercept_t2, residual_var_t2 = _local_linear_regression(c2_windows, fine2_windows, valid_t2)

    predicted_from_t0 = fine0 + slope_t0 * (c1 - c0)
    predicted_from_t2 = fine2 + slope_t2 * (c1 - c2)

    # Reliability weighting: the pair whose window regression fit fine
    # against coarse more consistently (lower residual variance) is
    # trusted more, not averaged blindly. A pair with no local coarse
    # variance to regress against (residual_var is NaN - e.g. a uniform
    # patch, water) fell back to the identity slope, not "no prediction" -
    # weight it neutrally (1.0), not zero, or a whole uniform region would
    # incorrectly come out fully unsupported despite having a perfectly
    # usable (if unrefined) STARFM-equivalent prediction.
    weight_t0 = np.where(np.isfinite(residual_var_t0), 1.0 / (residual_var_t0 + epsilon), 1.0)
    weight_t2 = np.where(np.isfinite(residual_var_t2), 1.0 / (residual_var_t2 + epsilon), 1.0)
    weight_sum = weight_t0 + weight_t2
    has_support = (
        (weight_sum > 0)
        & np.isfinite(fine0) & np.isfinite(fine2)
        & np.isfinite(c0) & np.isfinite(c1) & np.isfinite(c2)
    )
    safe_weight_sum = np.where(has_support, weight_sum, 1.0)
    prediction = (weight_t0 * predicted_from_t0 + weight_t2 * predicted_from_t2) / safe_weight_sum
    prediction = np.where(has_support, prediction, np.nan).astype(np.float32)

    coords = {"y": fine_t0.y, "x": fine_t0.x}
    result = xr.Dataset(
        {
            "lst_downscaled": xr.DataArray(prediction, dims=("y", "x"), coords=coords),
            "downscaled_support": xr.DataArray(has_support, dims=("y", "x"), coords=coords),
        },
        attrs={**fine_t0.attrs, "downscaled_is_modelled": True},
    )
    result.attrs.update({
        "downscaling_method": "estarfm",
        "estarfm_window_radius": window_radius,
        "estarfm_similarity_threshold": float(threshold),
    })
    return result
