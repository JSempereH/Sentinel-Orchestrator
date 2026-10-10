"""Calibrated, spatially varying prediction intervals for downscalers.

Every downscaler here reports ``lst_downscaled_uncertainty`` as one constant
training-residual standard deviation - the same number for a homogeneous park
and a mixed industrial pixel, and an in-sample one at that. Split conformal
prediction replaces it with intervals that have a finite-sample coverage
guarantee on exchangeable held-out data, and - when the base model exposes a
per-pixel difficulty estimate (the spread of a tree ensemble's members) -
intervals whose width follows that difficulty ("normalized" conformal).

Calibration happens where truth exists: the coarse grid. Applying the
calibrated quantile to fine-resolution predictions assumes the coarse-scale
error distribution transfers to the fine scale; that is an assumption, not
something these intervals can verify, and the result attributes say so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from ..cube import CubeValidationError, validate_cube


def ensemble_spread(model: Any, predictors: xr.Dataset) -> xr.DataArray | None:
    """Per-pixel standard deviation across an ensemble's members, or None.

    Works for any downscaler wrapping a fitted scikit-learn ensemble with
    ``estimators_`` (Random Forest, Extra Trees, bagging). The variance is
    accumulated tree by tree, so memory stays at a few arrays the size of
    one prediction regardless of the number of trees.
    """

    estimator = getattr(model, "model", None)
    members = getattr(estimator, "estimators_", None)
    names = getattr(model, "predictor_names", None)
    if members is None or names is None or len(members) < 2:
        return None
    validate_cube(predictors, required_variables=names, require_time=False)
    variables = xr.Dataset({name: predictors[name] for name in names}).to_array("variable")
    dims = tuple(dim for dim in variables.dims if dim != "variable")
    table = variables.transpose(*dims, "variable").values
    finite = np.isfinite(table).all(axis=-1)
    rows = table[finite]
    total = np.zeros(len(rows))
    total_sq = np.zeros(len(rows))
    for member in members:
        member = member[0] if isinstance(member, np.ndarray) else member  # gradient boosting stores arrays
        values = member.predict(rows)
        total += values
        total_sq += values**2
    count = len(members)
    variance = np.maximum(total_sq / count - (total / count) ** 2, 0.0) * count / (count - 1)
    spread = np.full(finite.shape, np.nan, dtype=np.float32)
    spread[finite] = np.sqrt(variance)
    return xr.DataArray(spread, dims=dims, coords={dim: predictors[dim] for dim in dims}, name="ensemble_spread")


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile with the finite-sample ``(n+1)`` correction."""

    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


@dataclass
class ConformalDownscaler:
    """Wrap a fitted downscaler with calibrated prediction intervals."""

    base_model: Any
    alpha: float
    quantile: float
    normalized: bool
    calibration_samples: int
    spread_floor: float = 0.0

    @property
    def predictor_names(self) -> tuple[str, ...]:
        return tuple(getattr(self.base_model, "predictor_names", ()))

    def _half_width(self, predictors: xr.Dataset, like: xr.DataArray) -> xr.DataArray:
        if not self.normalized:
            return xr.full_like(like, self.quantile, dtype=np.float32)
        spread = ensemble_spread(self.base_model, predictors)
        if spread is None:
            raise CubeValidationError("Normalized conformal intervals need a base model with ensemble members")
        return self.quantile * spread.clip(min=self.spread_floor)

    def predict(self, predictors: xr.Dataset) -> xr.Dataset:
        result = self.base_model.predict(predictors).copy()
        prediction = result["lst_downscaled"]
        half_width = self._half_width(predictors, prediction).where(prediction.notnull())
        result["lst_downscaled_lower"] = prediction - half_width
        result["lst_downscaled_upper"] = prediction + half_width
        result["lst_downscaled_uncertainty"] = half_width
        result["lst_downscaled_uncertainty"].attrs.update({
            "long_name": f"half-width of the {1 - self.alpha:.0%} conformal prediction interval",
        })
        result.attrs.update({
            "uncertainty_method": "normalized split conformal" if self.normalized else "split conformal",
            "uncertainty_coverage_target": 1 - self.alpha,
            "uncertainty_calibration_samples": self.calibration_samples,
            "uncertainty_calibration_scale": "coarse grid; fine-scale coverage assumes the error distribution transfers",
        })
        return result


def fit_conformal_downscaler(
    model: Any,
    calibration: xr.Dataset,
    *,
    target: str = "lst",
    alpha: float = 0.1,
    normalized: bool | None = None,
    min_samples: int = 30,
) -> ConformalDownscaler:
    """Calibrate intervals for a fitted downscaler on held-out coarse cells.

    ``calibration`` must not overlap the model's training data - use
    ``blocked_calibration_split`` for the same split the model was fitted on.
    ``normalized=None`` uses ensemble-spread scaling whenever the base model
    supports it and constant-width intervals otherwise.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    validate_cube(calibration, required_variables=(target,), require_time=False)
    spread = ensemble_spread(model, calibration)
    if normalized is None:
        normalized = spread is not None
    if normalized and spread is None:
        raise CubeValidationError("normalized=True needs a base model with ensemble members")
    prediction = model.predict(calibration)["lst_downscaled"]
    residual = np.abs(prediction - calibration[target])
    if normalized:
        assert spread is not None
        # A floor keeps near-zero spreads (all trees agree) from producing
        # unbounded scores and zero-width intervals.
        floor = float(np.nanpercentile(spread.values, 5)) if np.isfinite(spread.values).any() else 0.0
        floor = max(floor, 1e-6)
        scores = (residual / spread.clip(min=floor)).values
    else:
        floor = 0.0
        scores = residual.values
    scores = scores[np.isfinite(scores)]
    if len(scores) < min_samples:
        raise CubeValidationError(f"Only {len(scores)} calibration samples; need at least {min_samples}")
    return ConformalDownscaler(
        base_model=model,
        alpha=alpha,
        quantile=_conformal_quantile(scores, alpha),
        normalized=bool(normalized),
        calibration_samples=int(len(scores)),
        spread_floor=floor,
    )
