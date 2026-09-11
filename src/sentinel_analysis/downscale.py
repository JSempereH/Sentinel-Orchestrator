"""Transparent baseline downscaling with explicit residual uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import numpy as np
import xarray as xr

from .cube import CubeValidationError, validate_cube
from .validation import blocked_spatiotemporal_split, compare_to_reference


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

    def predict(self, predictors: xr.Dataset) -> xr.Dataset:
        """Predict a fine-grid LST and expose uncertainty/support variables."""

        validate_cube(predictors, required_variables=self.predictor_names, require_time=False)
        prediction = xr.zeros_like(predictors[self.predictor_names[0]], dtype=float) + self.intercept
        support = xr.ones_like(prediction, dtype=bool)
        for name, coefficient in zip(self.predictor_names, self.coefficients):
            values = predictors[name]
            support &= xr.apply_ufunc(np.isfinite, values)
            prediction = prediction + coefficient * values
        result = xr.Dataset({
            "lst_downscaled": prediction.where(support),
            "lst_downscaled_uncertainty": xr.full_like(prediction, self.residual_std).where(support),
            "downscaled_support": support,
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
class RandomForestDownscaler:
    """Optional non-linear baseline with reproducible training parameters."""

    predictor_names: tuple[str, ...]
    model: Any
    residual_std: float
    samples: int
    random_state: int

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
            "downscaling_method": "random_forest",
            "downscaling_predictors": list(self.predictor_names),
            "downscaling_training_samples": self.samples,
            "downscaling_residual_std": self.residual_std,
            "downscaling_random_state": self.random_state,
        })
        return result


@dataclass
class XGBoostDownscaler:
    """Optional gradient-boosted tree downscaler."""

    predictor_names: tuple[str, ...]
    model: Any
    residual_std: float
    samples: int
    random_state: int

    def predict(self, predictors: xr.Dataset) -> xr.Dataset:
        validate_cube(predictors, required_variables=self.predictor_names, require_time=False)
        variables = xr.Dataset({name: predictors[name] for name in self.predictor_names}).to_array("variable")
        dims = tuple(dim for dim in variables.dims if dim != "variable")
        table = variables.transpose(*dims, "variable").values
        finite = np.isfinite(table).all(axis=-1)
        prediction_values = np.full(finite.shape, np.nan, dtype=np.float32)
        prediction_values[finite] = self.model.predict(table[finite])
        prediction = xr.DataArray(
            prediction_values,
            dims=dims,
            coords={dim: predictors[dim] for dim in dims},
            name="lst_downscaled",
        )
        support = xr.DataArray(finite, dims=dims, coords=prediction.coords, name="downscaled_support")
        result = xr.Dataset({
            "lst_downscaled": prediction,
            "lst_downscaled_uncertainty": xr.full_like(prediction, self.residual_std).where(support),
            "downscaled_support": support,
        }, attrs={**predictors.attrs, "downscaled_is_modelled": True})
        result.attrs.update({
            "downscaling_method": "xgboost",
            "downscaling_predictors": list(self.predictor_names),
            "downscaling_training_samples": self.samples,
            "downscaling_residual_std": self.residual_std,
            "downscaling_random_state": self.random_state,
        })
        return result


def _kernel_weights(distances: np.ndarray, bandwidth: float, kernel: str) -> np.ndarray:
    """Spatial kernel weight as a function of distance/bandwidth. Bisquare
    (the common GWR default, e.g. Fotheringham et al.) gives exactly zero
    weight beyond the bandwidth radius, unlike Gaussian's soft tail."""

    ratio = distances / bandwidth
    if kernel == "gaussian":
        return np.exp(-0.5 * ratio**2)
    if kernel == "bisquare":
        return np.where(ratio < 1.0, (1.0 - ratio**2) ** 2, 0.0)
    raise ValueError(f"Unknown kernel {kernel!r}; use 'bisquare' or 'gaussian'")


def _weighted_least_squares(design: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    sqrt_weights = np.sqrt(weights)
    beta, *_ = np.linalg.lstsq(design * sqrt_weights[:, None], target * sqrt_weights, rcond=None)
    return beta


def _distinct_locations(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Group training rows by distinct (x, y) location.

    GWR training data is stacked over (time, y, x) (see `fit_gwr_downscaler`),
    so a pixel observed at T valid times appears T times at the *same*
    (x, y). Searching for nearest neighbors directly over these rows (as an
    earlier version of this module did) lets a training block with many
    time steps exhaust `max_local_samples` on time-duplicates of only a
    handful of the geographically nearest pixels - confirmed empirically:
    with ~20-28 training time steps, `max_local_samples=200` reached only
    8-10 distinct locations barely 2 pixels away, regardless of `bandwidth`
    (widening it 3x changed nothing, because the neighbor *set* never grew).

    Returns `(distinct_x, distinct_y, rows_per_location)`: one entry per
    distinct location, where `rows_per_location[i]` lists every row-index
    in the original `x`/`y` arrays sharing `distinct_x[i]`/`distinct_y[i]` -
    so a caller can search over locations for "how far to look" while still
    using every time-replicate at a chosen location for the actual fit.
    """

    coords = np.column_stack([x, y])
    _, first_index, inverse = np.unique(coords, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    order = np.argsort(inverse, kind="stable")
    boundaries = np.searchsorted(inverse[order], np.arange(len(first_index) + 1))
    rows_per_location = [order[boundaries[i]:boundaries[i + 1]] for i in range(len(first_index))]
    return x[first_index], y[first_index], rows_per_location


@dataclass(frozen=True)
class GWRDownscaler:
    """Geographically weighted regression - a locally-linear downscaler that
    refits its coefficients at every prediction location from the nearest
    coarse training samples, weighted by spatial distance.

    Unlike LinearDownscaler's single global relation, this directly models
    the local non-stationarity urban heat exhibits (a park's NDVI-LST slope
    is not a city's), the motivation cited for geographically weighted
    baselines like MFGWML in `docs/downscaling.md`. The cost is that
    ``predict`` solves one small weighted least-squares problem per fine
    pixel (using only its ``max_local_samples`` nearest training points, not
    the full training set - true GWR literature calls this "adaptive
    bandwidth via nearest neighbors") rather than reusing one fixed
    coefficient vector, so it is markedly slower than the other downscalers
    here for large grids.
    """

    predictor_names: tuple[str, ...]
    training_target: np.ndarray
    training_predictors: np.ndarray
    training_x: np.ndarray
    training_y: np.ndarray
    bandwidth: float
    kernel: str = "bisquare"
    min_local_samples: int = 10
    max_local_samples: int = 200
    residual_std: float = float("nan")
    samples: int = 0
    method: str = "geographically_weighted_regression"

    def predict(self, predictors: xr.Dataset) -> xr.Dataset:
        """Predict a fine grid. Local coefficients depend only on (y, x), so
        they are computed once per pixel and reused across every time step
        present in `predictors`, rather than recomputed per time slice."""

        validate_cube(predictors, required_variables=self.predictor_names, require_time=False)
        try:
            from scipy.spatial import cKDTree
        except ImportError as exc:
            raise RuntimeError("Install sentinel-analysis[ml] to use GWRDownscaler") from exc

        y_values = predictors["y"].values
        x_values = predictors["x"].values
        yy, xx = np.meshgrid(y_values, x_values, indexing="ij")
        height, width = yy.shape
        n_coefficients = len(self.predictor_names) + 1

        # Search over *distinct* training locations, not raw rows - see
        # _distinct_locations' docstring for why searching rows directly
        # silently caps the spatial reach far below `bandwidth` once a
        # training block spans many time steps.
        distinct_x, distinct_y, rows_per_location = _distinct_locations(self.training_x, self.training_y)
        tree = cKDTree(np.column_stack([distinct_x, distinct_y]))
        k = min(self.max_local_samples, len(distinct_x))
        _, location_indices = tree.query(np.column_stack([xx.ravel(), yy.ravel()]), k=k)
        if k == 1:
            location_indices = location_indices[:, None]

        coefficients = np.full((height * width, n_coefficients), np.nan, dtype=np.float64)
        supported = np.zeros(height * width, dtype=bool)
        flat_x, flat_y = xx.ravel(), yy.ravel()
        for pixel in range(height * width):
            rows = np.concatenate([rows_per_location[location] for location in location_indices[pixel]])
            pixel_distances = np.hypot(self.training_x[rows] - flat_x[pixel], self.training_y[rows] - flat_y[pixel])
            weights = _kernel_weights(pixel_distances, self.bandwidth, self.kernel)
            valid = weights > 0
            if valid.sum() < self.min_local_samples:
                continue
            local_index = rows[valid]
            design = np.column_stack([np.ones(valid.sum()), self.training_predictors[local_index]])
            coefficients[pixel] = _weighted_least_squares(design, self.training_target[local_index], weights[valid])
            supported[pixel] = True

        coefficients = coefficients.reshape(height, width, n_coefficients)
        supported = supported.reshape(height, width)
        coords = {"y": predictors.y, "x": predictors.x}
        support = xr.DataArray(supported, dims=("y", "x"), coords=coords)
        prediction = xr.DataArray(coefficients[..., 0], dims=("y", "x"), coords=coords)
        for index, name in enumerate(self.predictor_names):
            coefficient_map = xr.DataArray(coefficients[..., index + 1], dims=("y", "x"), coords=coords)
            values = predictors[name]
            support = support & xr.apply_ufunc(np.isfinite, values)
            prediction = prediction + coefficient_map * values

        result = xr.Dataset({
            "lst_downscaled": prediction.where(support),
            "lst_downscaled_uncertainty": xr.full_like(prediction, self.residual_std).where(support),
            "downscaled_support": support,
        })
        result.attrs.update(predictors.attrs)
        result.attrs.update({
            "downscaling_method": self.method,
            "downscaling_predictors": list(self.predictor_names),
            "downscaling_training_samples": self.samples,
            "downscaling_residual_std": self.residual_std,
            "gwr_bandwidth": self.bandwidth,
            "gwr_kernel": self.kernel,
            "downscaled_is_modelled": True,
        })
        return result


class _Downscaler(Protocol):
    def predict(self, predictors: xr.Dataset) -> xr.Dataset: ...


@dataclass(frozen=True)
class CoarseConsistentDownscaler:
    """Apply a downscaler while conserving the observed coarse target."""

    base_model: _Downscaler
    target: str = "lst"
    conservation_tolerance: float = 0.25

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
        result.attrs.update({
            **raw.attrs,
            "downscaling_method": f"coarse_consistent_{raw.attrs.get('downscaling_method', 'model')}",
            "coarse_target": self.target,
            "coarse_consistency_rmse": float(metrics["rmse"]),
            "coarse_consistency_mae": float(metrics["mae"]),
            "coarse_consistency_tolerance": self.conservation_tolerance,
            "coarse_consistency_within_tolerance": bool(float(metrics["rmse"]) <= self.conservation_tolerance),
        })
        return result


def fit_linear_downscaler(
    training: xr.Dataset,
    *,
    target: str = "lst",
    predictors: Sequence[str],
    min_samples: int = 100,
) -> LinearDownscaler:
    """Fit a small auditable baseline on a coarse, collocated feature cube."""

    validate_cube(training, required_variables=(target, *predictors), require_time=True)
    values = xr.Dataset({name: training[name] for name in (target, *predictors)}).to_array("variable")
    table = values.stack(sample=("time", "y", "x")).transpose("sample", "variable").values
    finite = np.isfinite(table).all(axis=1)
    table = table[finite]
    if table.shape[0] < min_samples:
        raise CubeValidationError(f"Only {table.shape[0]} complete samples; need at least {min_samples}")
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
    return LinearDownscaler(
        predictor_names=model.predictor_names,
        coefficients=model.coefficients,
        intercept=model.intercept,
        residual_std=model.residual_std,
        samples=model.samples,
        condition_number=model.condition_number,
        method="ts_harp_dis_trad",
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
    validate_cube(training, required_variables=(target, *predictors), require_time=True)
    values = xr.Dataset({name: training[name] for name in (target, *predictors)}).to_array("variable")
    table = values.stack(sample=("time", "y", "x")).transpose("sample", "variable").values
    finite = np.isfinite(table).all(axis=1)
    table = table[finite]
    if table.shape[0] < min_samples:
        raise CubeValidationError(f"Only {table.shape[0]} complete samples; need at least {min_samples}")
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(table[:, 1:], table[:, 0])
    residuals = table[:, 0] - model.predict(table[:, 1:])
    return RandomForestDownscaler(tuple(predictors), model, float(np.std(residuals, ddof=1)), int(len(table)), random_state)


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
    validate_cube(training, required_variables=(target, *predictors), require_time=True)
    values = xr.Dataset({name: training[name] for name in (target, *predictors)}).to_array("variable")
    table = values.stack(sample=("time", "y", "x")).transpose("sample", "variable").values
    finite = np.isfinite(table).all(axis=1)
    table = table[finite]
    if table.shape[0] < min_samples:
        raise CubeValidationError(f"Only {table.shape[0]} complete samples; need at least {min_samples}")
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(table[:, 1:], table[:, 0], verbose=False)
    residuals = table[:, 0] - model.predict(table[:, 1:])
    return XGBoostDownscaler(tuple(predictors), model, float(np.std(residuals, ddof=1)), int(len(table)), random_state)


def fit_gwr_downscaler(
    training: xr.Dataset,
    *,
    target: str = "lst",
    predictors: Sequence[str],
    bandwidth: float,
    kernel: str = "bisquare",
    min_samples: int = 100,
    min_local_samples: int = 10,
    max_local_samples: int = 200,
) -> GWRDownscaler:
    """Fit a geographically weighted regression baseline on a collocated
    coarse cube.

    `bandwidth` is in the same units as the cube's x/y coordinates (usually
    metres, for a projected AnalysisGrid) and controls how far a prediction
    location looks for training samples - too small starves local fits of
    data (see `min_local_samples`), too large converges toward the single
    global relation `fit_linear_downscaler` already gives directly.

    `residual_std` is a leave-one-out cross-validated residual (each
    training sample's own local fit excludes it), not the optimistic
    in-sample residual the other downscalers here report, since GWR's local
    fits would otherwise partly explain a point using itself.
    """

    validate_cube(training, required_variables=(target, *predictors), require_time=True)
    values = xr.Dataset({name: training[name] for name in (target, *predictors)}).to_array("variable")
    stacked = values.stack(sample=("time", "y", "x")).transpose("sample", "variable")
    table = stacked.values
    finite = np.isfinite(table).all(axis=1)
    x_values = stacked["x"].values[finite]
    y_values = stacked["y"].values[finite]
    table = table[finite]
    if table.shape[0] < min_samples:
        raise CubeValidationError(f"Only {table.shape[0]} complete samples; need at least {min_samples}")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")

    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("Install sentinel-analysis[ml] to use GWRDownscaler") from exc

    training_target = table[:, 0].astype(np.float64)
    training_predictors = table[:, 1:].astype(np.float64)

    # Search over distinct locations, not raw rows - see _distinct_locations'
    # docstring. Excluding the held-out sample's whole *location* (not just
    # its own row) also fixes a related leak: a different training time step
    # at the exact same pixel sits at distance 0 and would otherwise get the
    # maximum possible weight in its own "held-out" fit.
    distinct_x, distinct_y, rows_per_location = _distinct_locations(x_values, y_values)
    location_of_row = np.empty(len(training_target), dtype=int)
    for location, rows in enumerate(rows_per_location):
        location_of_row[rows] = location

    tree = cKDTree(np.column_stack([distinct_x, distinct_y]))
    k = min(max_local_samples + 1, len(distinct_x))  # +1 so excluding self's location still leaves max_local_samples
    _, location_neighbor_indices = tree.query(np.column_stack([distinct_x, distinct_y]), k=k)
    if k == 1:
        location_neighbor_indices = location_neighbor_indices[:, None]

    residuals = []
    for sample in range(len(training_target)):
        own_location = location_of_row[sample]
        neighbor_locations = [loc for loc in location_neighbor_indices[own_location] if loc != own_location]
        if not neighbor_locations:
            continue
        rows = np.concatenate([rows_per_location[location] for location in neighbor_locations])
        pixel_distances = np.hypot(x_values[rows] - x_values[sample], y_values[rows] - y_values[sample])
        weights = _kernel_weights(pixel_distances, bandwidth, kernel)
        valid = weights > 0
        if valid.sum() < min_local_samples:
            continue
        local_index = rows[valid]
        design = np.column_stack([np.ones(valid.sum()), training_predictors[local_index]])
        beta = _weighted_least_squares(design, training_target[local_index], weights[valid])
        predicted = beta[0] + training_predictors[sample] @ beta[1:]
        residuals.append(training_target[sample] - predicted)
    residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float("nan")

    return GWRDownscaler(
        predictor_names=tuple(predictors),
        training_target=training_target,
        training_predictors=training_predictors,
        training_x=x_values.astype(np.float64),
        training_y=y_values.astype(np.float64),
        bandwidth=float(bandwidth),
        kernel=kernel,
        min_local_samples=min_local_samples,
        max_local_samples=max_local_samples,
        residual_std=residual_std,
        samples=int(table.shape[0]),
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
    values = fine.transpose("time", "y", "x").values if "time" in fine.dims else fine.values[None, ...]
    output = np.full((values.shape[0], target.sizes["y"], target.sizes["x"]), np.nan, dtype=np.float32)
    for index, layer in enumerate(values):
        sums = np.zeros(output.shape[1:], dtype=np.float64)
        counts = np.zeros(output.shape[1:], dtype=np.int32)
        finite = np.isfinite(layer) & inside
        yy, xx = np.where(finite)
        np.add.at(sums, (rows[yy], cols[xx]), layer[yy, xx])
        np.add.at(counts, (rows[yy], cols[xx]), 1)
        np.divide(sums, counts, out=output[index], where=counts > 0)
    coords = {"y": target.y, "x": target.x}
    if "time" in fine.dims:
        coords["time"] = fine.time
        return xr.DataArray(output, dims=("time", "y", "x"), coords=coords, name=fine.name, attrs=fine.attrs)
    return xr.DataArray(output[0], dims=("y", "x"), coords=coords, name=fine.name, attrs=fine.attrs)


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

    prediction = model.predict(validation)["lst_downscaled"]
    metrics = compare_to_reference(prediction, validation[target], name=target)
    support = prediction.notnull()
    if "downscaled_extrapolation" in model.predict(validation):
        extrapolation = model.predict(validation)["downscaled_extrapolation"]
        metrics["extrapolation_fraction"] = float(extrapolation.where(support).mean().item())
    return metrics


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
