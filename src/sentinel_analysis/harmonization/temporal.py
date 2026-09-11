"""Temporal matching policies used by the workflow runner."""

from __future__ import annotations

import numpy as np
import xarray as xr


def nearest_time(dataset: xr.Dataset, target_time: xr.DataArray, tolerance: np.timedelta64) -> xr.Dataset:
    """Match a feature cube to target times with an explicit tolerance."""

    return dataset.reindex(time=target_time, method="nearest", tolerance=str(tolerance))
