"""Local moving-window regression, after the Data Mining Sharpener (pyDMS).

The multi-city benchmark found pyDMS's local mode, trained per scene, the
most accurate downscaler (``docs/downscaling.md``). This is the same idea
with this library's data model:

* a global model for the whole scene plus local models in a grid of
  windows of ``window`` coarse cells; each local model trains on its window
  extended by a quarter of its size on every side, so neighbours overlap;
* every model is a bag of regression trees whose leaves hold a ridge
  regression, clipped to the range seen in the leaf plus a margin, so it
  bends with the data but does not extrapolate without limit;
* local and global predictions are blended per coarse cell with weights
  1/residual^2 (each model's error on that cell), interpolated to the fine
  grid; temperatures are blended as radiance (T^4);
* coarse training cells are weighted by how homogeneous their fine
  predictors are (inverse coefficient of variation), when the fine
  predictors are given.

Memory stays bounded: windows are fitted and predicted one at a time, each
over its own part of the fine grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import xarray as xr

from ..cube import validate_cube

# pyDMS defaults: leaf counts of local and global trees, and how far a leaf
# may extrapolate beyond the target range it saw, as a share of that range.
LOCAL_MAX_LEAVES = 10
GLOBAL_MAX_LEAVES = 30
LEAF_EXTRAPOLATION_RATIO = 0.25
WINDOW_EXTENSION = 0.25


def _weighted_ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, float]:
    """Coefficients and intercept of a weighted ridge regression (as scikit-learn's ``Ridge``).

    Solved in closed form: a leaf has a few dozen samples and a handful of
    predictors, and building thousands of estimator objects per scene cost
    more than the arithmetic.
    """

    total = w.sum()
    x_mean = (w[:, None] * X).sum(axis=0) / total
    y_mean = float((w * y).sum() / total)
    Xc, yc = X - x_mean, y - y_mean
    gram = (Xc * w[:, None]).T @ Xc + alpha * np.eye(X.shape[1])
    coef = np.linalg.solve(gram, (Xc * w[:, None]).T @ yc)
    return coef, y_mean - float(x_mean @ coef)


class LinearLeafTreeEnsemble:
    """Bagged regression trees with a ridge regression in every leaf."""

    def __init__(self, *, max_leaf_nodes: int, n_estimators: int = 20, min_samples_leaf: int = 10, random_state: int = 42):
        self.max_leaf_nodes = max_leaf_nodes
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        # Per member: the tree and, indexed by tree node id, each leaf's
        # coefficients, intercept, clip range and whether it has a regression.
        self.members: list[tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "LinearLeafTreeEnsemble":
        from sklearn.tree import DecisionTreeRegressor

        rng = np.random.default_rng(self.random_state)
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.members = []
        for member in range(self.n_estimators):
            sample = rng.integers(0, len(y), len(y))  # bootstrap
            Xs, ys, ws = X[sample], y[sample], weights[sample]
            tree = DecisionTreeRegressor(max_leaf_nodes=self.max_leaf_nodes, min_samples_leaf=min(self.min_samples_leaf, max(1, len(y) // 4)), random_state=self.random_state + member)
            tree.fit(Xs, ys, sample_weight=ws)
            nodes = tree.tree_.node_count
            coefs = np.zeros((nodes, X.shape[1]))
            intercepts = np.zeros(nodes)
            low = np.full(nodes, -np.inf)
            high = np.full(nodes, np.inf)
            has_regression = np.zeros(nodes, dtype=bool)
            assigned = tree.apply(Xs)
            for leaf in np.unique(assigned):
                in_leaf = assigned == leaf
                lowest, highest = float(ys[in_leaf].min()), float(ys[in_leaf].max())
                margin = LEAF_EXTRAPOLATION_RATIO * (highest - lowest)
                low[leaf], high[leaf] = lowest - margin, highest + margin
                if in_leaf.sum() > X.shape[1] + 1 and ws[in_leaf].sum() > 0:
                    coefs[leaf], intercepts[leaf] = _weighted_ridge(Xs[in_leaf], ys[in_leaf], ws[in_leaf])
                    has_regression[leaf] = True
            self.members.append((tree, coefs, intercepts, low, high, has_regression))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.members:
            raise RuntimeError("fit the ensemble before predicting")
        total = np.zeros(len(X))
        for tree, coefs, intercepts, low, high, has_regression in self.members:
            leaf = tree.apply(X)
            linear = np.einsum("ij,ij->i", X, coefs[leaf]) + intercepts[leaf]
            # Leaves without a regression predict their mean.
            total += np.where(has_regression[leaf], np.clip(linear, low[leaf], high[leaf]), tree.predict(X))
        return total / len(self.members)


@dataclass
class LocalWindowDownscaler:
    """Global plus moving-window local models, blended by local accuracy."""

    predictor_names: tuple[str, ...]
    global_model: LinearLeafTreeEnsemble
    # (row0, row1, col0, col1) of each window's own coarse cells, and its model (None: too few samples).
    windows: list[tuple[tuple[int, int, int, int], LinearLeafTreeEnsemble | None]]
    local_weight: xr.DataArray  # per coarse cell, weight of the local model in [0, 1]
    residual_std: float
    samples: int
    temperature_offset: float | None = None  # add to the target to get kelvin when blending as radiance
    method: str = "local_trees"
    extra_attrs: dict[str, Any] = field(default_factory=dict)

    def predict(self, predictors: xr.Dataset) -> xr.Dataset:
        validate_cube(predictors, required_variables=self.predictor_names, require_time=False)
        from .consistency import _bilinear_to

        stack = xr.Dataset({name: predictors[name] for name in self.predictor_names}).to_array("variable")
        dims = tuple(dim for dim in stack.dims if dim != "variable")
        if dims[-2:] != ("y", "x"):
            stack = stack.transpose(..., "y", "x", "variable")
            dims = tuple(dim for dim in stack.dims if dim != "variable")
        # (leading, y, x, variable), with any time axis folded into "leading".
        table = stack.transpose(*dims, "variable").values
        shape = table.shape[:-1]
        table = table.reshape((-1, *shape[-2:], table.shape[-1]))
        finite = np.isfinite(table).all(axis=-1)
        global_values = np.full(finite.shape, np.nan)
        global_values[finite] = self.global_model.predict(table[finite])
        local_values = np.full(finite.shape, np.nan)
        coarse = self.local_weight
        y_edges, x_edges = _cell_edges(coarse.y.values), _cell_edges(coarse.x.values)
        for (r0, r1, c0, c1), model in self.windows:
            if model is None:
                continue
            rows = np.flatnonzero(_between(predictors.y.values, y_edges[r0], y_edges[r1]))
            cols = np.flatnonzero(_between(predictors.x.values, x_edges[c0], x_edges[c1]))
            if not rows.size or not cols.size:
                continue
            window_table = table[:, rows][:, :, cols]
            window_finite = np.isfinite(window_table).all(axis=-1)
            values = np.full(window_finite.shape, np.nan)
            values[window_finite] = model.predict(window_table[window_finite])
            local_values[:, rows[:, None], cols[None, :]] = values
        global_values, local_values, finite = (array.reshape(shape) for array in (global_values, local_values, finite))
        template = predictors[self.predictor_names[0]]
        weight = _bilinear_to(coarse.fillna(0.0), template.isel(time=0, drop=True) if "time" in template.dims else template).clip(0, 1).values
        weight = np.where(np.isfinite(local_values), weight, 0.0)
        blended = _blend(local_values, global_values, weight, self.temperature_offset)
        prediction = xr.DataArray(blended.astype(np.float32), dims=dims, coords={dim: predictors[dim] for dim in dims}, name="lst_downscaled")
        support = xr.DataArray(finite, dims=dims, coords=prediction.coords, name="downscaled_support")
        result = xr.Dataset({
            "lst_downscaled": prediction.where(support),
            "lst_downscaled_uncertainty": xr.full_like(prediction, self.residual_std).where(support),
            "downscaled_support": support,
        }, attrs={**predictors.attrs, "downscaled_is_modelled": True})
        result.attrs.update({
            "downscaling_method": self.method,
            "downscaling_predictors": list(self.predictor_names),
            "downscaling_training_samples": self.samples,
            "downscaling_residual_std": self.residual_std,
            "downscaling_windows": len(self.windows),
            "downscaling_local_windows_fitted": sum(model is not None for _, model in self.windows),
            **self.extra_attrs,
        })
        return result


def fit_local_window_downscaler(
    training: xr.Dataset,
    *,
    target: str = "lst",
    predictors: Sequence[str],
    window: int = 10,
    min_samples: int = 30,
    min_window_samples: int = 10,
    n_estimators: int = 20,
    random_state: int = 42,
    fine: xr.Dataset | None = None,
) -> LocalWindowDownscaler:
    """Fit global and moving-window local linear-leaf tree ensembles on one coarse scene.

    ``window`` is in coarse cells. A window with fewer than
    ``min_window_samples`` complete cells gets no local model and uses the
    global one. ``fine``, the fine predictors of the same scene, weights
    coarse samples by the homogeneity of the fine pixels inside them.
    """

    from .regression import training_table

    if window < 1:
        raise ValueError("window must be at least one coarse cell")
    predictors = tuple(predictors)
    table, x, y = training_table(training, target=target, predictors=predictors, min_samples=min_samples)
    target_values, features = table[:, 0], table[:, 1:]
    weights = _homogeneity_weights(training, fine, predictors, x, y) if fine is not None else np.ones(len(target_values))

    global_model = LinearLeafTreeEnsemble(max_leaf_nodes=GLOBAL_MAX_LEAVES, n_estimators=n_estimators, random_state=random_state)
    global_model.fit(features, target_values, sample_weight=weights)

    grid = training[target].isel(time=0) if "time" in training[target].dims else training[target]
    rows_of = {value: index for index, value in enumerate(grid.y.values)}
    cols_of = {value: index for index, value in enumerate(grid.x.values)}
    sample_rows = np.array([rows_of[value] for value in y])
    sample_cols = np.array([cols_of[value] for value in x])
    n_rows, n_cols = grid.sizes["y"], grid.sizes["x"]
    extension = max(1, int(round(window * WINDOW_EXTENSION)))

    local_error = np.full((n_rows, n_cols), np.nan)
    global_error = np.full((n_rows, n_cols), np.nan)
    global_error[sample_rows, sample_cols] = np.abs(global_model.predict(features) - target_values)
    windows: list[tuple[tuple[int, int, int, int], LinearLeafTreeEnsemble | None]] = []
    for r0 in range(0, n_rows, window):
        for c0 in range(0, n_cols, window):
            r1, c1 = min(r0 + window, n_rows), min(c0 + window, n_cols)
            in_training = (
                (sample_rows >= r0 - extension) & (sample_rows < r1 + extension)
                & (sample_cols >= c0 - extension) & (sample_cols < c1 + extension)
            )
            model = None
            if in_training.sum() >= min_window_samples:
                model = LinearLeafTreeEnsemble(max_leaf_nodes=LOCAL_MAX_LEAVES, n_estimators=n_estimators, random_state=random_state)
                model.fit(features[in_training], target_values[in_training], sample_weight=weights[in_training])
                own = (sample_rows >= r0) & (sample_rows < r1) & (sample_cols >= c0) & (sample_cols < c1)
                if own.any():
                    local_error[sample_rows[own], sample_cols[own]] = np.abs(model.predict(features[own]) - target_values[own])
            windows.append(((r0, r1, c0, c1), model))

    # pyDMS's blending weight; a cell without a local model keeps weight 0.
    floor = 1e-6
    inverse_local = 1.0 / np.maximum(local_error, floor) ** 2
    inverse_global = 1.0 / np.maximum(global_error, floor) ** 2
    weight = np.where(np.isfinite(local_error) & np.isfinite(global_error), inverse_local / (inverse_local + inverse_global), np.nan)
    local_weight = xr.DataArray(weight, dims=("y", "x"), coords={"y": grid.y, "x": grid.x}, name="local_weight")
    # Cells without their own sample borrow the mean weight of their window.
    for (r0, r1, c0, c1), model in windows:
        block = local_weight.values[r0:r1, c0:c1]
        fill = float(np.nanmean(block)) if model is not None and np.isfinite(block).any() else 0.0
        block[~np.isfinite(block)] = fill

    residual = target_values - global_model.predict(features)
    units = str(training[target].attrs.get("units", ""))
    offset = 273.15 if units in {"degC", "celsius", "°C"} else 0.0 if units in {"K", "kelvin"} else None
    return LocalWindowDownscaler(
        predictor_names=predictors,
        global_model=global_model,
        windows=windows,
        local_weight=local_weight,
        residual_std=float(np.std(residual)),
        samples=len(target_values),
        temperature_offset=offset,
        extra_attrs={"downscaling_window_cells": window},
    )


def _homogeneity_weights(training: xr.Dataset, fine: xr.Dataset, predictors: Sequence[str], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """pyDMS sample weights: inverse mean coefficient of variation of the fine predictors in each coarse cell."""

    from .consistency import reaggregate_to_target

    template = training[predictors[0]].isel(time=0, drop=True) if "time" in training[predictors[0]].dims else training[predictors[0]]
    template = template.assign_attrs(crs=training.attrs.get("crs"))
    variation = []
    for name in predictors:
        if name not in fine:
            continue
        values = fine[name].isel(time=0, drop=True) if "time" in fine[name].dims else fine[name]
        values = values.assign_attrs(crs=fine.attrs.get("crs", training.attrs.get("crs")))
        mean = reaggregate_to_target(values, template)
        square = reaggregate_to_target(values**2, template)
        mean_values = mean.values
        std = np.sqrt(np.maximum(square.values - mean_values**2, 0))
        variation.append(std / np.where(np.abs(mean_values) > 1e-6, np.abs(mean_values), np.nan))
    if not variation:
        return np.ones(len(x))
    cv = np.nanmean(np.stack(variation), axis=0)
    rows = {value: index for index, value in enumerate(template.y.values)}
    cols = {value: index for index, value in enumerate(template.x.values)}
    sample_cv = cv[[rows[value] for value in y], [cols[value] for value in x]]
    weights = 1.0 / np.where(np.isfinite(sample_cv) & (sample_cv > 0), sample_cv, np.nan)
    finite = np.isfinite(weights)
    if not finite.any():
        return np.ones(len(x))
    low, high = np.nanmin(weights), np.nanmax(weights)
    scaled = (weights - low) / (high - low) if high > low else np.ones_like(weights)
    return np.where(finite, np.clip(scaled, 0.05, 1.0), 0.05)


def _blend(local: np.ndarray, global_: np.ndarray, weight: np.ndarray, offset: float | None) -> np.ndarray:
    local_filled = np.where(np.isfinite(local), local, global_)
    if offset is None:
        return weight * local_filled + (1 - weight) * global_
    # Temperatures mix as emitted radiance, not linearly.
    radiance = weight * (local_filled + offset) ** 4 + (1 - weight) * (global_ + offset) ** 4
    return radiance ** 0.25 - offset


def _cell_edges(centres: np.ndarray) -> np.ndarray:
    step = centres[1] - centres[0] if centres.size > 1 else 1.0
    return np.concatenate([centres - step / 2, [centres[-1] + step / 2]])


def _between(values: np.ndarray, a: float, b: float) -> np.ndarray:
    low, high = min(a, b), max(a, b)
    return (values >= low) & (values < high)

