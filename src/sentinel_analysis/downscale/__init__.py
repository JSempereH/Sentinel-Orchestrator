"""Coarse-to-fine thermal downscaling and spatiotemporal fusion.

- regression: OLS, TsHARP and any scikit-learn-style estimator
  (Random Forest, XGBoost, ...).
- gwr: geographically weighted regression.
- consistency: coarse-scale conservation wrappers, reaggregation and
  validation helpers.
- spatiotemporal: STARFM and ESTARFM.
"""

from .consistency import (
    CoarseConsistentDownscaler,
    fit_coarse_consistent_gwr_downscaler,
    fit_coarse_consistent_linear_downscaler,
    fit_coarse_consistent_random_forest_downscaler,
    fit_coarse_consistent_tsharp_downscaler,
    fit_coarse_consistent_xgboost_downscaler,
    reaggregate_to_target,
    split_spatiotemporal,
    validate_downscaler,
    validate_reaggregation,
)
from .gwr import GWRDownscaler, fit_gwr_downscaler
from .regression import (
    LinearDownscaler,
    RandomForestDownscaler,
    SklearnDownscaler,
    XGBoostDownscaler,
    fit_linear_downscaler,
    fit_random_forest_downscaler,
    fit_sklearn_downscaler,
    fit_tsharp_downscaler,
    fit_xgboost_downscaler,
)
from .spatiotemporal import fuse_estarfm, fuse_starfm
from .uncertainty import ConformalDownscaler, ensemble_spread, fit_conformal_downscaler

__all__ = [
    "CoarseConsistentDownscaler",
    "ConformalDownscaler",
    "ensemble_spread",
    "fit_conformal_downscaler",
    "GWRDownscaler",
    "LinearDownscaler",
    "RandomForestDownscaler",
    "SklearnDownscaler",
    "XGBoostDownscaler",
    "fit_coarse_consistent_gwr_downscaler",
    "fit_coarse_consistent_linear_downscaler",
    "fit_coarse_consistent_random_forest_downscaler",
    "fit_coarse_consistent_tsharp_downscaler",
    "fit_coarse_consistent_xgboost_downscaler",
    "fit_gwr_downscaler",
    "fit_linear_downscaler",
    "fit_random_forest_downscaler",
    "fit_sklearn_downscaler",
    "fit_tsharp_downscaler",
    "fit_xgboost_downscaler",
    "fuse_estarfm",
    "fuse_starfm",
    "reaggregate_to_target",
    "split_spatiotemporal",
    "validate_downscaler",
    "validate_reaggregation",
]
