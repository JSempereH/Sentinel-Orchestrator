"""Geographically weighted regression downscaler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import xarray as xr

from ..cube import validate_cube
from .regression import training_table


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
        extrapolation = xr.zeros_like(support)
        for index, name in enumerate(self.predictor_names):
            coefficient_map = xr.DataArray(coefficients[..., index + 1], dims=("y", "x"), coords=coords)
            values = predictors[name]
            support = support & xr.apply_ufunc(np.isfinite, values)
            # Local fits are linear too: clip to the training range (see
            # LinearDownscaler.predict) and flag the clipped cells.
            low, high = float(np.min(self.training_predictors[:, index])), float(np.max(self.training_predictors[:, index]))
            extrapolation = extrapolation | (values < low) | (values > high)
            prediction = prediction + coefficient_map * values.clip(low, high)

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
            "gwr_bandwidth": self.bandwidth,
            "gwr_kernel": self.kernel,
            "downscaled_is_modelled": True,
        })
        return result


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

    table, x_values, y_values = training_table(training, target=target, predictors=predictors, min_samples=min_samples)
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
