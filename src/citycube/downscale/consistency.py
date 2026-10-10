"""Coarse-scale conservation, reaggregation and downscaler validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import numpy as np
import xarray as xr

from ..cube import CubeValidationError, validate_cube
from ..validation import blocked_spatiotemporal_split, compare_to_reference
from .gwr import fit_gwr_downscaler
from .regression import fit_linear_downscaler, fit_random_forest_downscaler, fit_tsharp_downscaler, fit_xgboost_downscaler


class _Downscaler(Protocol):
    def predict(self, predictors: xr.Dataset) -> xr.Dataset: ...


CORRECTIONS = ("block", "smooth")


@dataclass(frozen=True)
class CoarseConsistentDownscaler:
    """Apply a downscaler while conserving the observed coarse target.

    ``correction="block"`` adds each coarse cell's residual (observed minus
    aggregated prediction) uniformly to its fine cells: exact conservation,
    but visible coarse-cell steps where the residual is large.
    ``"smooth"`` interpolates the residual bilinearly and repeats
    aggregate-and-correct ``smoothing_iterations`` times, which removes the
    steps and conserves the observation to within a few hundredths of a
    kelvin (reported as ``coarse_consistency_rmse``).
    """

    base_model: _Downscaler
    target: str = "lst"
    conservation_tolerance: float = 0.25
    correction: str = "block"
    smoothing_iterations: int = 10

    def __post_init__(self) -> None:
        if self.correction not in CORRECTIONS:
            raise ValueError(f"correction must be one of {CORRECTIONS}")

    def predict(self, predictors: xr.Dataset, coarse_reference: xr.Dataset | xr.DataArray) -> xr.Dataset:
        """Predict a fine grid and remove the coarse-scale aggregation residual."""

        raw = self.base_model.predict(predictors)
        coarse = coarse_reference[self.target] if isinstance(coarse_reference, xr.Dataset) else coarse_reference
        coarse_dataset = coarse_reference if isinstance(coarse_reference, xr.Dataset) else xr.Dataset({self.target: coarse}, attrs=coarse.attrs)
        validate_cube(coarse_dataset, required_variables=(self.target,), require_time=True)
        prediction = raw["lst_downscaled"].copy()
        prediction.attrs.update(predictors.attrs)
        aggregated = reaggregate_to_target(prediction, coarse)
        coarse_for_prediction = coarse.reindex(time=prediction.time, method="nearest")
        residual = coarse_for_prediction - aggregated
        if self.correction == "smooth":
            correction = self._smooth_correction(prediction.where(raw["downscaled_support"]), coarse, coarse_for_prediction)
        else:
            correction = residual.reindex(time=prediction.time, method="nearest")
            correction = correction.reindex(y=prediction.y, x=prediction.x, method="nearest").fillna(0)
        corrected = (prediction + correction).where(raw["downscaled_support"])
        corrected_aggregate = reaggregate_to_target(corrected, coarse)
        corrected_aggregate = corrected_aggregate.reindex(time=prediction.time, method="nearest")
        metrics = compare_to_reference(corrected_aggregate, coarse_for_prediction, name="coarse_consistency")
        result = xr.Dataset({
            "lst_downscaled_raw": prediction,
            "lst_downscaled": corrected,
            "lst_downscaled_uncertainty": raw["lst_downscaled_uncertainty"],
            "downscaled_support": raw["downscaled_support"],
            "coarse_consistency_correction": correction,
        })
        if "downscaled_extrapolation" in raw:
            result["downscaled_extrapolation"] = raw["downscaled_extrapolation"]
        result.attrs.update({
            **raw.attrs,
            "downscaling_method": f"coarse_consistent_{raw.attrs.get('downscaling_method', 'model')}",
            "coarse_target": self.target,
            "coarse_consistency_rmse": float(metrics["rmse"]),
            "coarse_consistency_mae": float(metrics["mae"]),
            "coarse_consistency_tolerance": self.conservation_tolerance,
            "coarse_consistency_within_tolerance": bool(float(metrics["rmse"]) <= self.conservation_tolerance),
            "coarse_consistency_correction_method": self.correction,
        })
        return result

    def _smooth_correction(self, prediction: xr.DataArray, coarse: xr.DataArray, coarse_for_prediction: xr.DataArray) -> xr.DataArray:
        """Iteratively add the bilinearly interpolated coarse residual.

        Coarse cells without an observation contribute no residual (zero),
        so the correction fades smoothly across them instead of stepping.
        """

        correction = xr.zeros_like(prediction).fillna(0)
        for _ in range(self.smoothing_iterations):
            aggregated = reaggregate_to_target(prediction + correction, coarse).reindex(time=prediction.time, method="nearest")
            residual = (coarse_for_prediction - aggregated).fillna(0)
            correction = correction + _bilinear_to(residual, prediction)
        return correction


def _bilinear_to(coarse: xr.DataArray, fine: xr.DataArray) -> xr.DataArray:
    """Interpolate a coarse field bilinearly onto a fine grid, nearest beyond the outer cell centres."""

    ordered = coarse.sortby("y").sortby("x")
    interpolated = ordered.interp(y=fine.y, x=fine.x, method="linear")
    edges = ordered.reindex(y=fine.y, x=fine.x, method="nearest")
    return interpolated.combine_first(edges).transpose(*fine.dims)


def reaggregate_to_target(fine: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
    """Aggregate a fine regular grid into the cells of a coarse target grid."""

    for dataset, name in ((fine, "fine"), (target, "target")):
        if not {"y", "x"}.issubset(dataset.dims):
            raise CubeValidationError(f"{name} data must contain y and x dimensions")
    if fine.attrs.get("crs") and target.attrs.get("crs") and fine.attrs["crs"] != target.attrs["crs"]:
        raise CubeValidationError("Reaggregation requires both grids to use the same CRS")
    if fine.x.size < 2 or fine.y.size < 2 or target.x.size < 2 or target.y.size < 2:
        raise CubeValidationError("Reaggregation requires regular grids")
    target_dx = float(abs(target.x.values[1] - target.x.values[0]))
    target_dy = float(abs(target.y.values[1] - target.y.values[0]))
    left = float(target.x.values.min() - target_dx / 2)
    top = float(target.y.values.max() + target_dy / 2)
    cols = np.floor((fine.x.values - left) / target_dx).astype(int)
    rows = np.floor((top - fine.y.values) / target_dy).astype(int)
    inside = (rows[:, None] >= 0) & (rows[:, None] < target.sizes["y"]) & (cols[None, :] >= 0) & (cols[None, :] < target.sizes["x"])
    shape = (target.sizes["y"], target.sizes["x"])

    def aggregate(values: np.ndarray) -> np.ndarray:
        layers = values.reshape((-1, *values.shape[-2:]))
        output = np.full((layers.shape[0], *shape), np.nan, dtype=np.float32)
        for index, layer in enumerate(layers):
            sums = np.zeros(shape, dtype=np.float64)
            counts = np.zeros(shape, dtype=np.int32)
            finite = np.isfinite(layer) & inside
            yy, xx = np.where(finite)
            np.add.at(sums, (rows[yy], cols[xx]), layer[yy, xx])
            np.add.at(counts, (rows[yy], cols[xx]), 1)
            np.divide(sums, counts, out=output[index], where=counts > 0)
        return output.reshape((*values.shape[:-2], *shape))

    # apply_ufunc keeps dask-backed inputs lazy (one task per non-spatial
    # chunk, e.g. per time chunk) instead of loading the whole cube. Each map
    # must be one chunk in y and x (Zarr stores are often tiled spatially).
    if fine.chunks is not None:
        fine = fine.chunk({"y": -1, "x": -1})
    result = xr.apply_ufunc(
        aggregate,
        fine,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y_target", "x_target"]],
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={"output_sizes": {"y_target": shape[0], "x_target": shape[1]}},
        keep_attrs=True,
    )
    result = result.rename({"y_target": "y", "x_target": "x"}).assign_coords(y=target.y.values, x=target.x.values)
    return result.transpose(*(("time",) if "time" in result.dims else ()), *(d for d in result.dims if d not in {"time", "y", "x"}), "y", "x").rename(fine.name)


def validate_reaggregation(
    coarse_observation: xr.DataArray,
    fine_prediction: xr.DataArray,
    *,
    tolerance: float,
) -> dict[str, float | int | bool | str]:
    """Check whether a fine prediction conserves the observed coarse scale."""

    aggregated = reaggregate_to_target(fine_prediction, coarse_observation)
    metrics = compare_to_reference(aggregated, coarse_observation, name="reaggregated_prediction")
    metrics.update({
        "within_tolerance": bool(float(metrics["rmse"]) <= tolerance),
        "tolerance": tolerance,
    })
    return metrics


def validate_downscaler(
    model: Any,
    validation: xr.Dataset,
    *,
    target: str = "lst",
) -> dict[str, float | int | str]:
    """Evaluate a fitted downscaler on a held-out spatiotemporal cube."""

    # Predict once: GWR and tree ensembles are expensive to re-run.
    predicted = model.predict(validation)
    prediction = predicted["lst_downscaled"]
    metrics = compare_to_reference(prediction, validation[target], name=target)
    support = prediction.notnull()
    if "downscaled_extrapolation" in predicted:
        extrapolation = predicted["downscaled_extrapolation"]
        metrics["extrapolation_fraction"] = float(extrapolation.where(support).mean().item())
    if {"lst_downscaled_lower", "lst_downscaled_upper"}.issubset(predicted.data_vars):
        reference = validation[target]
        lower, upper = predicted["lst_downscaled_lower"], predicted["lst_downscaled_upper"]
        scored = lower.notnull() & upper.notnull() & reference.notnull()
        inside = (reference >= lower) & (reference <= upper)
        metrics["interval_coverage"] = float(inside.where(scored).mean().item())
        metrics["interval_mean_width"] = float((upper - lower).where(scored).mean().item())
    return metrics


def split_spatiotemporal(
    dataset: xr.Dataset,
    *,
    validation_fraction: float = 0.2,
    block_period: int = 5,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Create a deterministic time-and-space holdout for model validation."""

    return blocked_spatiotemporal_split(
        dataset,
        validation_fraction=validation_fraction,
        spatial_block_period=block_period,
    )


def _fit_coarse_consistent(
    fit_fn: Callable[..., _Downscaler],
    training: xr.Dataset,
    *,
    target: str,
    conservation_tolerance: float,
    **fit_kwargs: Any,
) -> CoarseConsistentDownscaler:
    """Shared body for every `fit_coarse_consistent_*_downscaler`: fit the
    given base model, then wrap it for coarse-scale conservation. Each public
    wrapper below exists only to expose that specific model's own kwargs
    (predictors vs. predictor, bandwidth/kernel, n_estimators, ...) with
    named parameters and defaults for IDE autocomplete/docs, not because the
    wrapping logic itself differs between them."""

    return CoarseConsistentDownscaler(
        fit_fn(training, target=target, **fit_kwargs),
        target=target,
        conservation_tolerance=conservation_tolerance,
    )


def fit_coarse_consistent_linear_downscaler(
    training: xr.Dataset,
    *,
    target: str = "lst",
    predictors: Sequence[str],
    min_samples: int = 100,
    conservation_tolerance: float = 0.25,
) -> CoarseConsistentDownscaler:
    """Fit an OLS downscaler with a coarse-scale conservation correction."""

    return _fit_coarse_consistent(
        fit_linear_downscaler, training, target=target, conservation_tolerance=conservation_tolerance,
        predictors=predictors, min_samples=min_samples,
    )


def fit_coarse_consistent_random_forest_downscaler(
    training: xr.Dataset,
    *,
    target: str = "lst",
    predictors: Sequence[str],
    n_estimators: int = 200,
    random_state: int = 42,
    min_samples: int = 100,
    conservation_tolerance: float = 0.25,
) -> CoarseConsistentDownscaler:
    """Fit a Random Forest downscaler with a coarse-scale conservation correction."""

    return _fit_coarse_consistent(
        fit_random_forest_downscaler, training, target=target, conservation_tolerance=conservation_tolerance,
        predictors=predictors, n_estimators=n_estimators, random_state=random_state, min_samples=min_samples,
    )


def fit_coarse_consistent_tsharp_downscaler(
    training: xr.Dataset,
    *,
    target: str = "lst",
    predictor: str,
    min_samples: int = 100,
    conservation_tolerance: float = 0.25,
) -> CoarseConsistentDownscaler:
    """Fit TsHARP/DisTrad with coarse-scale conservation."""

    return _fit_coarse_consistent(
        fit_tsharp_downscaler, training, target=target, conservation_tolerance=conservation_tolerance,
        predictor=predictor, min_samples=min_samples,
    )


def fit_coarse_consistent_xgboost_downscaler(
    training: xr.Dataset,
    *,
    target: str = "lst",
    predictors: Sequence[str],
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    random_state: int = 42,
    min_samples: int = 100,
    conservation_tolerance: float = 0.25,
) -> CoarseConsistentDownscaler:
    """Fit XGBoost with coarse-scale conservation."""

    return _fit_coarse_consistent(
        fit_xgboost_downscaler, training, target=target, conservation_tolerance=conservation_tolerance,
        predictors=predictors, n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, random_state=random_state, min_samples=min_samples,
    )


def fit_coarse_consistent_gwr_downscaler(
    training: xr.Dataset,
    *,
    target: str = "lst",
    predictors: Sequence[str],
    bandwidth: float,
    kernel: str = "bisquare",
    min_samples: int = 100,
    min_local_samples: int = 10,
    max_local_samples: int = 200,
    conservation_tolerance: float = 0.25,
) -> CoarseConsistentDownscaler:
    """Fit a geographically weighted regression downscaler with coarse-scale
    conservation."""

    return _fit_coarse_consistent(
        fit_gwr_downscaler, training, target=target, conservation_tolerance=conservation_tolerance,
        predictors=predictors, bandwidth=bandwidth, kernel=kernel,
        min_samples=min_samples, min_local_samples=min_local_samples, max_local_samples=max_local_samples,
    )
