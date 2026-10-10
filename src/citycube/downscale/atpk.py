"""Area-to-point kriging (ATPK) of coarse residuals.

The residual correction of area-to-point regression kriging (ATPRK, Wang et
al. 2015, Remote Sensing of Environment 166:191-204; Kyriakidis 2004,
Geographical Analysis 36:259-289): after a regression model predicts the
fine grid, the coarse residual (observation minus aggregated prediction) is
spread onto the fine cells by kriging that accounts for the size of the
coarse cells, instead of being added as a block or interpolated bilinearly.

Steps:

1. The empirical semivariogram of the coarse residuals is computed.
2. A point-support exponential covariance is fitted so that, averaged over
   pairs of coarse cells (regularised), it reproduces that semivariogram:
   the deconvolution step. Its sill is linear in the fit and is solved in
   closed form for each candidate range.
3. Each fine cell gets a simple-kriging combination of the residuals of
   the coarse cells in a ``(2 * radius + 1)`` square window around its own
   cell, using area-to-area and area-to-point covariances. The mean is
   taken as zero, as it is for regression residuals: far from any
   observation the correction fades to zero instead of carrying the local
   residual into a gap (which a blocked holdout showed to hurt). Kriging systems
   are cached by which neighbours are observed, so a mostly clear scene
   solves only a handful of distinct systems.

ATPK is coherent: averaged over a coarse cell, its prediction returns that
cell's residual. Discretising each cell with a few sub-points makes that
approximate, so the small remaining mismatch is added per block and the
result conserves the observation exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Sub-points per side used to represent a coarse cell when averaging covariances.
_CELL_DISCRETISATION = 5
# Positions per side at which fine cells inside a coarse cell are distinguished.
_MAX_POSITION_BINS = 16
# Candidate point ranges, in coarse-cell sizes.
_RANGE_CANDIDATES = np.geomspace(0.25, 30.0, 48)


@dataclass(frozen=True)
class PointCovariance:
    """Point-support exponential covariance ``sill * exp(-h / range_m)``."""

    sill: float
    range_m: float

    def __call__(self, distance: np.ndarray) -> np.ndarray:
        return self.sill * np.exp(-np.asarray(distance) / self.range_m)


def _sub_points(dy: float, dx: float, n: int = _CELL_DISCRETISATION) -> np.ndarray:
    """Centres of an ``n x n`` partition of a cell, relative to its centre, in metres (y, x)."""

    offsets = (np.arange(n) + 0.5) / n - 0.5
    yy, xx = np.meshgrid(offsets * dy, offsets * dx, indexing="ij")
    return np.column_stack([yy.ravel(), xx.ravel()])


def _area_to_area(offsets: np.ndarray, dy: float, dx: float, range_m: float) -> np.ndarray:
    """Mean unit-sill covariance between a cell and the cells at ``offsets`` (rows, cols)."""

    points = _sub_points(dy, dx)
    shifts = offsets * np.array([dy, dx])
    # (offset, p, q) distances between sub-point p of the centre cell and q of the shifted cell.
    delta = points[None, :, None, :] - (points[None, None, :, :] + shifts[:, None, None, :])
    distance = np.sqrt((delta ** 2).sum(-1))
    return np.exp(-distance / range_m).mean(axis=(1, 2))


def fit_point_covariance(residual: np.ndarray, dy: float, dx: float, *, max_lag: int = 8) -> PointCovariance | None:
    """Deconvolve a point covariance from a 2-D coarse residual field (NaN = unobserved).

    Returns ``None`` when there are too few residuals or no variability.
    """

    valid = np.isfinite(residual)
    if valid.sum() < 10 or float(np.nanvar(residual)) <= 1e-12:
        return None
    ny, nx = residual.shape
    max_lag = max(1, min(max_lag, max(ny, nx) // 2))
    lags, gammas, counts = [], [], []
    for di in range(0, max_lag + 1):
        for dj in range(-max_lag, max_lag + 1):
            if (di == 0 and dj <= 0) or di * di + dj * dj > max_lag * max_lag:
                continue
            # Cell pairs separated by (di, dj): a at (i, j), b at (i + di, j + dj).
            a = residual[: ny - di, max(0, -dj):nx - max(0, dj)]
            b = residual[di:, max(0, dj):nx - max(0, -dj)]
            pair = np.isfinite(a) & np.isfinite(b)
            n = int(pair.sum())
            if n < 5:
                continue
            lags.append((di, dj))
            gammas.append(0.5 * float(np.mean((a[pair] - b[pair]) ** 2)))
            counts.append(n)
    if len(lags) < 3:
        return None
    offsets = np.array(lags, dtype=float)
    gamma = np.array(gammas)
    weight = np.array(counts, dtype=float)
    cell = max(dy, dx)
    best: tuple[float, float, float] | None = None
    for factor in _RANGE_CANDIDATES:
        range_m = factor * cell
        within = _area_to_area(np.zeros((1, 2)), dy, dx, range_m)[0]
        between = _area_to_area(offsets, dy, dx, range_m)
        shape = within - between  # regularised semivariogram per unit point sill
        denominator = float((weight * shape * shape).sum())
        if denominator <= 0:
            continue
        sill = float((weight * gamma * shape).sum()) / denominator
        if sill <= 0:
            continue
        error = float((weight * (gamma - sill * shape) ** 2).sum())
        if best is None or error < best[0]:
            best = (error, sill, range_m)
    if best is None:
        return None
    return PointCovariance(sill=best[1], range_m=best[2])


def atpk_residual_field(
    residual: np.ndarray,
    coarse_y: np.ndarray,
    coarse_x: np.ndarray,
    fine_y: np.ndarray,
    fine_x: np.ndarray,
    *,
    radius: int = 2,
    covariance: PointCovariance | None = None,
) -> np.ndarray:
    """Spread a coarse residual field onto a fine grid by area-to-point kriging.

    ``residual`` is ``(len(coarse_y), len(coarse_x))`` with NaN where the
    coarse cell was not observed; coordinates are cell centres in metres.
    Returns a ``(len(fine_y), len(fine_x))`` array; fine cells outside the
    coarse grid get NaN, cells with no observed neighbour get 0. Raises
    ``ValueError`` when no covariance can be fitted (the caller falls back).
    """

    coarse_y, coarse_x = np.asarray(coarse_y, float), np.asarray(coarse_x, float)
    fine_y, fine_x = np.asarray(fine_y, float), np.asarray(fine_x, float)
    dy, dx = float(abs(coarse_y[1] - coarse_y[0])), float(abs(coarse_x[1] - coarse_x[0]))
    if covariance is None:
        covariance = fit_point_covariance(residual, dy, dx)
    if covariance is None:
        raise ValueError("too few coarse residuals to fit a covariance for ATPK")

    # Coarse cell and position within it of every fine row and column (coarse rows run north to south).
    top, left = float(coarse_y.max() + dy / 2), float(coarse_x.min() - dx / 2)
    ascending_y = coarse_y[0] < coarse_y[-1]
    row_position = (top - fine_y) / dy
    col_position = (fine_x - left) / dx
    fine_rows = np.floor(row_position).astype(int)
    fine_cols = np.floor(col_position).astype(int)
    ny, nx = residual.shape
    # Work north-up internally.
    field = residual[::-1] if ascending_y else residual
    bins_v = min(_MAX_POSITION_BINS, max(1, int(round(dy / max(abs(fine_y[1] - fine_y[0]), 1e-9)))))
    bins_u = min(_MAX_POSITION_BINS, max(1, int(round(dx / max(abs(fine_x[1] - fine_x[0]), 1e-9)))))
    row_bin = np.clip(np.floor((row_position - fine_rows) * bins_v).astype(int), 0, bins_v - 1)
    col_bin = np.clip(np.floor((col_position - fine_cols) * bins_u).astype(int), 0, bins_u - 1)

    window = np.array([(di, dj) for di in range(-radius, radius + 1) for dj in range(-radius, radius + 1)], dtype=float)
    # Covariances between window cells (area-to-area), indexed by window position pairs.
    pair_offsets = (window[:, None, :] - window[None, :, :]).reshape(-1, 2)
    unique_offsets, inverse = np.unique(pair_offsets, axis=0, return_inverse=True)
    c_vv = (covariance.sill * _area_to_area(unique_offsets, dy, dx, covariance.range_m))[inverse].reshape(len(window), len(window))
    # Covariances between a point at each position bin of the centre cell and each window cell (area-to-point).
    centres_v = ((np.arange(bins_v) + 0.5) / bins_v - 0.5) * dy
    centres_u = ((np.arange(bins_u) + 0.5) / bins_u - 0.5) * dx
    points = np.array([(v, u) for v in centres_v for u in centres_u])
    cell_points = _sub_points(dy, dx)
    targets = window * np.array([dy, dx])
    delta = points[:, None, None, :] - (targets[None, :, None, :] + cell_points[None, None, :, :])
    c_xv = covariance(np.sqrt((delta ** 2).sum(-1))).mean(-1)  # (bins, window)

    output = np.full((fine_y.size, fine_x.size), np.nan)
    inside_rows = (fine_rows >= 0) & (fine_rows < ny)
    inside_cols = (fine_cols >= 0) & (fine_cols < nx)
    padded = np.pad(field, radius, constant_values=np.nan)
    cache: dict[bytes, np.ndarray | None] = {}
    rows_of = {i: np.flatnonzero(inside_rows & (fine_rows == i)) for i in range(ny)}
    cols_of = {j: np.flatnonzero(inside_cols & (fine_cols == j)) for j in range(nx)}
    for i in range(ny):
        fr = rows_of[i]
        if fr.size == 0:
            continue
        for j in range(nx):
            fc = cols_of[j]
            if fc.size == 0:
                continue
            values = padded[i:i + 2 * radius + 1, j:j + 2 * radius + 1].ravel()
            observed = np.isfinite(values)
            if not observed.any():
                output[np.ix_(fr, fc)] = 0.0
                continue
            key = observed.tobytes()
            weights = cache.get(key, False)
            if weights is False:
                index = np.flatnonzero(observed)
                try:
                    weights = np.linalg.solve(c_vv[np.ix_(index, index)], c_xv[:, index].T).T  # (bins, n)
                except np.linalg.LinAlgError:
                    weights = None
                cache[key] = weights
            if weights is None:
                output[np.ix_(fr, fc)] = values[radius * (2 * radius + 1) + radius] if np.isfinite(values[radius * (2 * radius + 1) + radius]) else 0.0
                continue
            per_bin = (weights @ values[observed]).reshape(bins_v, bins_u)
            output[np.ix_(fr, fc)] = per_bin[np.ix_(row_bin[fr], col_bin[fc])]
    return output
