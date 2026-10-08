"""Global regression downscalers: OLS, TsHARP and any scikit-learn-style estimator."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import xarray as xr

from ..cube import CubeValidationError, validate_cube


def training_table(
    training: xr.Dataset,
    *,
    target: str,
    predictors: Sequence[str],
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack a collocated coarse cube into complete (target, *predictors) rows.

    Returns (table, x, y): rows with any non-finite value are dropped, and
    x/y give each kept row's location (GWR needs them).
    """

    validate_cube(training, required_variables=(target, *predictors), require_time=True)
    values = xr.Dataset({name: training[name] for name in (target, *predictors)}).to_array("variable")
    stacked = values.stack(sample=("time", "y", "x")).transpose("sample", "variable")
    table = stacked.values
    finite = np.isfinite(table).all(axis=1)
    if int(finite.sum()) < min_samples:
        raise CubeValidationError(f"Only {int(finite.sum())} complete samples; need at least {min_samples}")
    return table[finite], stacked["x"].values[finite], stacked["y"].values[finite]


@dataclass(frozen=True)
class LinearDownscaler:
    """Linear feature model fitted on a common coarse analysis grid."""

    predictor_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    residual_std: float
    samples: int
    condition_number: float
    method: str = "ordinary least squares"
    predictor_min: tuple[float, ...] | None = None
    predictor_max: tuple[float, ...] | None = None

    def predict(self, predictors: xr.Dataset) -> xr.Dataset:
        """Predict a fine-grid LST and expose uncertainty/support variables.

        Predictors are clipped to the range seen in training and the clipped
        cells are flagged in ``downscaled_extrapolation``. Indices such as
        EVI can take extreme values at 100 m that coarse 1 km aggregates
        never reach; an unbounded linear model extrapolates them without
        limit (one real Guadalajara run produced a 737 K RMSE that way).
        """

        validate_cube(predictors, required_variables=self.predictor_names, require_time=False)
        prediction = xr.zeros_like(predictors[self.predictor_names[0]], dtype=float) + self.intercept
        support = xr.ones_like(prediction, dtype=bool)
        extrapolation = xr.zeros_like(prediction, dtype=bool)
        for index, (name, coefficient) in enumerate(zip(self.predictor_names, self.coefficients)):
            values = predictors[name]
            support &= xr.apply_ufunc(np.isfinite, values)
            if self.predictor_min is not None and self.predictor_max is not None:
                low, high = self.predictor_min[index], self.predictor_max[index]
                extrapolation |= (values < low) | (values > high)
                values = values.clip(low, high)
            prediction = prediction + coefficient * values
        result = xr.Dataset({
            "lst_downscaled": prediction.where(support),
            "lst_downscaled_uncertainty": xr.full_like(prediction, self.residual_std).where(support),
            "downscaled_support": support,
            "downscaled_extrapolation": extrapolation & support,
        })
        result.attrs.update(predictors.attrs)
        result.attrs.update({
            "downscaling_method": self.method,
            "downscaling_predictors": list(self.predictor_names),
            "downscaling_training_samples": self.samples,
            "downscaling_residual_std": self.residual_std,
            "downscaling_condition_number": self.condition_number,
            "downscaled_is_modelled": True,
        })
        return result


@dataclass
class SklearnDownscaler:
    """Downscaler around any fitted estimator exposing predict(X).

    Random Forest and XGBoost are thin named subclasses; pass any other
    scikit-learn-compatible regressor to fit_sklearn_downscaler.
    """

    predictor_names: tuple[str, ...]
    model: Any
    residual_std: float
    samples: int
    random_state: int | None = None
    method: str = "sklearn_estimator"

    def predict(self, predictors: xr.Dataset) -> xr.Dataset:
        validate_cube(predictors, required_variables=self.predictor_names, require_time=False)
        variables = xr.Dataset({name: predictors[name] for name in self.predictor_names}).to_array("variable")
        dims = tuple(dim for dim in variables.dims if dim != "variable")
        table = variables.transpose(*dims, "variable").values
        finite = np.isfinite(table).all(axis=-1)
        prediction_values = np.full(finite.shape, np.nan, dtype=np.float32)
        prediction_values[finite] = self.model.predict(table[finite])
        prediction = xr.DataArray(prediction_values, dims=dims, coords={dim: predictors[dim] for dim in dims}, name="lst_downscaled")
        support = xr.DataArray(finite, dims=dims, coords=prediction.coords, name="downscaled_support")
        result = xr.Dataset({
            "lst_downscaled": prediction,
            "lst_downscaled_uncertainty": xr.full_like(prediction, self.residual_std).where(support),
            "downscaled_support": support,
        }, attrs={**predictors.attrs, "downscaled_is_modelled": True})
        result.attrs.update({
            "downscaling_method": self.method,
            "downscaling_predictors": list(self.predictor_names),
            "downscaling_training_samples": self.samples,
            "downscaling_residual_std": self.residual_std,
        })
        if self.random_state is not None:
            result.attrs["downscaling_random_state"] = self.random_state
        return result


@dataclass
class RandomForestDownscaler(SklearnDownscaler):
    """Optional non-linear baseline with reproducible training parameters."""

    method: str = "random_forest"


@dataclass
class XGBoostDownscaler(SklearnDownscaler):
    """Optional gradient-boosted tree downscaler."""

    method: str = "xgboost"


def fit_linear_downscaler(
    training: xr.Dataset,
    *,
    target: str = "lst",
    predictors: Sequence[str],
    min_samples: int = 100,
) -> LinearDownscaler:
    """Fit a small auditable baseline on a coarse, collocated feature cube."""

    table, _, _ = training_table(training, target=target, predictors=predictors, min_samples=min_samples)
    design = np.column_stack([np.ones(table.shape[0]), table[:, 1:]])
    target_values = table[:, 0]
    coefficients, _, _, _ = np.linalg.lstsq(design, target_values, rcond=None)
    residuals = target_values - design @ coefficients
    condition_number = float(np.linalg.cond(design))
    return LinearDownscaler(
        predictor_names=tuple(predictors),
        coefficients=tuple(float(value) for value in coefficients[1:]),
        intercept=float(coefficients[0]),
        residual_std=float(np.std(residuals, ddof=1)),
        samples=int(table.shape[0]),
        condition_number=condition_number,
        predictor_min=tuple(float(value) for value in table[:, 1:].min(axis=0)),
        predictor_max=tuple(float(value) for value in table[:, 1:].max(axis=0)),
    )


def fit_tsharp_downscaler(
    training: xr.Dataset,
    *,
    target: str = "lst",
    predictor: str,
    min_samples: int = 100,
) -> LinearDownscaler:
    """Fit the TsHARP/DisTrad linear relation for one fine-scale predictor.

    TsHARP and DisTrad require a coarse target-to-predictor relation, then
    apply that relation to the fine predictor.  The caller supplies a cube
    where both variables are already collocated at the coarse training grid.
    """

    model = fit_linear_downscaler(training, target=target, predictors=(predictor,), min_samples=min_samples)
    return dataclasses.replace(model, method="ts_harp_dis_trad")


def fit_sklearn_downscaler(
    training: xr.Dataset,
    estimator: Any,
    *,
    target: str = "lst",
    predictors: Sequence[str],
    min_samples: int = 100,
    method: str = "sklearn_estimator",
    random_state: int | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    downscaler_class: type[SklearnDownscaler] = SklearnDownscaler,
) -> SklearnDownscaler:
    """Fit an unfitted scikit-learn-style regressor on a collocated coarse cube."""

    table, _, _ = training_table(training, target=target, predictors=predictors, min_samples=min_samples)
    estimator.fit(table[:, 1:], table[:, 0], **(fit_kwargs or {}))
    residuals = table[:, 0] - estimator.predict(table[:, 1:])
    return downscaler_class(
        predictor_names=tuple(predictors),
        model=estimator,
        residual_std=float(np.std(residuals, ddof=1)),
        samples=int(len(table)),
        random_state=random_state,
        method=method,
    )


def fit_random_forest_downscaler(
    training: xr.Dataset,
    *,
    target: str = "lst",
    predictors: Sequence[str],
    n_estimators: int = 200,
    random_state: int = 42,
    min_samples: int = 100,
) -> RandomForestDownscaler:
    """Fit an optional non-linear baseline on a collocated coarse cube."""

    try:
        from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Install sentinel-analysis[ml] to use RandomForestDownscaler") from exc
    model = fit_sklearn_downscaler(
        training,
        RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1),
        target=target, predictors=predictors, min_samples=min_samples,
        method="random_forest", random_state=random_state, downscaler_class=RandomForestDownscaler,
    )
    assert isinstance(model, RandomForestDownscaler)
    return model


def fit_xgboost_downscaler(
    training: xr.Dataset,
    *,
    target: str = "lst",
    predictors: Sequence[str],
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    random_state: int = 42,
    min_samples: int = 100,
) -> XGBoostDownscaler:
    """Fit an optional XGBoost regressor on a collocated feature cube."""

    try:
        from xgboost import XGBRegressor  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Install sentinel-analysis[ml] to use XGBoostDownscaler") from exc
    estimator = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )
    model = fit_sklearn_downscaler(
        training, estimator, target=target, predictors=predictors, min_samples=min_samples,
        method="xgboost", random_state=random_state, fit_kwargs={"verbose": False}, downscaler_class=XGBoostDownscaler,
    )
    assert isinstance(model, XGBoostDownscaler)
    return model
