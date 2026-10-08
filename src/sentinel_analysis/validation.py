"""Reference-data validation metrics for aligned geospatial variables."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .cube import validate_cube


def compare_to_reference(
    estimate: xr.DataArray,
    reference: xr.DataArray,
    *,
    name: str = "estimate",
) -> dict[str, float | int | str | bool]:
    """Return bias, MAE, RMSE and correlation on finite collocated values."""

    estimate, reference = xr.align(estimate, reference, join="inner")
    values = np.column_stack([np.asarray(estimate.values).ravel(), np.asarray(reference.values).ravel()])
    values = values[np.isfinite(values).all(axis=1)]
    if not len(values):
        raise ValueError("No finite collocated values are available for validation")
    error = values[:, 0] - values[:, 1]
    if len(values) > 1 and np.std(values[:, 0]) > 0 and np.std(values[:, 1]) > 0:
        correlation = float(np.corrcoef(values[:, 0], values[:, 1])[0, 1])
    else:
        correlation = float("nan")
    return {
        "variable": name,
        "samples": int(len(values)),
        "bias": float(np.mean(error)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "correlation": correlation,
    }


def _split_masks(
    dataset: xr.Dataset,
    *,
    validation_fraction: float,
    spatial_block_period: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    validate_cube(dataset, require_time=True)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if spatial_block_period < 2:
        raise ValueError("spatial_block_period must be at least 2")
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    if "time" in dataset.indexes and not dataset.indexes["time"].is_monotonic_increasing:
        # The temporal holdout is "the latest times" by position; on an
        # unsorted cube that silently becomes an arbitrary subset.
        raise ValueError("blocked_spatiotemporal_split requires a chronologically sorted time index; call .sortby('time') first")
    cutoff = max(1, int(np.floor(dataset.sizes["time"] * (1 - validation_fraction))))
    time_train = np.arange(dataset.sizes["time"]) < cutoff
    rows = np.arange(dataset.sizes["y"]) // block_size
    cols = np.arange(dataset.sizes["x"]) // block_size
    spatial_holdout = ((rows[:, None] + cols[None, :]) % spatial_block_period) == 0
    return time_train, spatial_holdout


def blocked_spatiotemporal_split(
    dataset: xr.Dataset,
    *,
    validation_fraction: float = 0.2,
    spatial_block_period: int = 5,
    block_size: int = 1,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Split a cube without mixing future times and spatial blocks.

    The validation set consists of spatially held-out cells from the latest
    time block.  This is deliberately deterministic so model comparisons are
    reproducible.

    Held-out cells form a diagonal pattern of ``block_size`` x ``block_size``
    pixel blocks. The default ``block_size=1`` holds out single pixels whose
    direct neighbours are all in training, so spatial autocorrelation makes
    the holdout optimistic; use a block several pixels wide (e.g. 5 km on a
    1 km thermal grid) for a genuinely spatial test.
    """

    time_train, spatial_holdout = _split_masks(
        dataset, validation_fraction=validation_fraction, spatial_block_period=spatial_block_period, block_size=block_size,
    )
    train_mask = xr.DataArray(
        time_train[:, None, None] & ~spatial_holdout[None, :, :],
        dims=("time", "y", "x"),
    )
    validation_mask = xr.DataArray(
        ~time_train[:, None, None] & spatial_holdout[None, :, :],
        dims=("time", "y", "x"),
    )
    return dataset.where(train_mask), dataset.where(validation_mask)


def blocked_calibration_split(
    dataset: xr.Dataset,
    *,
    validation_fraction: float = 0.2,
    spatial_block_period: int = 5,
    block_size: int = 1,
) -> xr.Dataset:
    """Cells usable to calibrate prediction intervals for the same split.

    These are the spatially held-out blocks at *training* times: excluded
    from ``blocked_spatiotemporal_split``'s training set (so a model has not
    seen them) and earlier than its validation set (so calibrating on them
    does not touch validation data).
    """

    time_train, spatial_holdout = _split_masks(
        dataset, validation_fraction=validation_fraction, spatial_block_period=spatial_block_period, block_size=block_size,
    )
    mask = xr.DataArray(time_train[:, None, None] & spatial_holdout[None, :, :], dims=("time", "y", "x"))
    return dataset.where(mask)


def validate_independent_reference(
    estimate: xr.DataArray,
    reference: xr.DataArray,
    *,
    reference_name: str,
) -> dict[str, float | int | str | bool]:
    """Evaluate an estimate against a named, independent reference product."""

    metrics = compare_to_reference(estimate, reference, name=reference_name)
    metrics["reference"] = reference_name
    metrics["independent_reference"] = True
    return metrics
