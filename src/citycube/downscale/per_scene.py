"""Per-scene downscaling: one model per coarse acquisition, anchored to it.

The multi-city benchmark (``docs/downscaling.md``, "Benchmark Results
(multi-city)") found this to be the most accurate protocol: a model trained
on other days cannot know today's overall temperature level, while a model
fitted on the scene it sharpens (its own 1 km cells and their aggregated
predictors) only has to learn the spatial pattern. The coarse-consistency
correction then makes each sharpened scene average back to the observed
coarse scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import xarray as xr

from ..cube import CubeValidationError
from .consistency import CORRECTIONS, CoarseConsistentDownscaler
from .local import fit_local_window_downscaler
from .regression import fit_linear_downscaler, fit_random_forest_downscaler, fit_xgboost_downscaler

logger = logging.getLogger(__name__)

PER_SCENE_MODELS: dict[str, Callable[..., Any]] = {
    "linear": fit_linear_downscaler,
    "random_forest": fit_random_forest_downscaler,
    "xgboost": fit_xgboost_downscaler,
    # Global plus moving-window local models, after pyDMS (downscale/local.py).
    "local_trees": fit_local_window_downscaler,
}
# Models that also use the scene's fine predictors when fitting.
_USES_FINE = {"local_trees"}
# Sentinel-2 indices plus the terrain predictors (present only with
# terrain_predictors=True); the predictor set the benchmark found best.
DEFAULT_PREDICTORS = ("NDVI", "NDBI", "EVI", "NDMI", "elevation", "slope", "cos_incidence")
# Time-dependent terrain predictor: recomputed at each coarse acquisition time.
_ILLUMINATION = "cos_incidence"


@dataclass(frozen=True)
class DownscaleSpec:
    """Request-level settings for per-scene downscaling."""

    model: str = "local_trees"
    target: str = "lst"
    predictors: tuple[str, ...] = ()
    coarse_consistent: bool = True
    min_samples: int = 30
    predictor_sensor: str = "sentinel2"
    model_options: Mapping[str, Any] = field(default_factory=dict)
    correction: str = "smooth"
    mask_unobserved: bool = True

    def __post_init__(self) -> None:
        if self.model not in PER_SCENE_MODELS:
            raise ValueError(f"downscale model must be one of {tuple(PER_SCENE_MODELS)}")
        if self.correction not in CORRECTIONS:
            raise ValueError(f"downscale correction must be one of {CORRECTIONS}")
        if self.min_samples < 2:
            raise ValueError("downscale min_samples must be at least 2")
        object.__setattr__(self, "predictors", tuple(self.predictors))
        object.__setattr__(self, "model_options", dict(self.model_options))

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "target": self.target,
            "predictors": list(self.predictors),
            "coarse_consistent": self.coarse_consistent,
            "min_samples": self.min_samples,
            "predictor_sensor": self.predictor_sensor,
            "model_options": dict(self.model_options),
            "correction": self.correction,
            "mask_unobserved": self.mask_unobserved,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DownscaleSpec":
        return cls(
            model=value.get("model", "random_forest"),
            target=value.get("target", "lst"),
            predictors=tuple(value.get("predictors", ())),
            coarse_consistent=bool(value.get("coarse_consistent", True)),
            min_samples=int(value.get("min_samples", 30)),
            predictor_sensor=value.get("predictor_sensor", "sentinel2"),
            model_options=dict(value.get("model_options", {})),
            correction=value.get("correction", "smooth"),
            mask_unobserved=bool(value.get("mask_unobserved", True)),
        )


def downscale_per_scene(
    cube: xr.Dataset,
    fine_predictors: xr.Dataset,
    *,
    target: str = "lst",
    predictors: Sequence[str] | None = None,
    model: str = "local_trees",
    coarse_consistent: bool = True,
    min_samples: int = 30,
    predictor_sensor: str = "sentinel2",
    terrain: xr.Dataset | None = None,
    model_options: Mapping[str, Any] | None = None,
    correction: str = "smooth",
    mask_unobserved: bool = True,
    store: str | Path | None = None,
    domain: xr.DataArray | None = None,
) -> xr.Dataset:
    """Sharpen every coarse scene of ``cube`` onto ``fine_predictors``' grid.

    ``cube`` is the fused coarse cube (``AnalysisResult.cube``): the target
    plus the predictors aggregated onto the coarse grid and
    ``<predictor_sensor>_matched_time``, the fine acquisition matched to each
    coarse time. ``fine_predictors`` is that sensor's own cube at fine
    resolution (``AnalysisResult.predictors[predictor_sensor]``). ``terrain``
    (``AnalysisResult.terrain``, static, no time axis) supplies ``elevation``
    and ``slope`` and is used to recompute ``cos_incidence`` at each coarse
    acquisition time.

    For each coarse time a model is fitted on that scene alone and predicts
    its fine predictors. Scenes without a matched fine acquisition or with
    fewer than ``min_samples`` complete cells are skipped and listed in the
    ``downscaling_skipped_scenes`` attribute. Raises ``ValueError`` when no
    scene can be downscaled.

    ``correction`` is the coarse-consistency correction (see
    ``CoarseConsistentDownscaler``): ``"smooth"`` avoids coarse-cell steps.
    With ``mask_unobserved`` fine cells whose coarse cell had no valid
    observation (cloud, quality) are left empty: the model would otherwise
    present clear-sky estimates there as if they were observed.
    ``coarse_observed`` marks the cells that had an observation.

    ``domain`` is a boolean (y, x) mask on the fine grid (``aoi_mask`` of a
    polygon AOI): downscaled values outside it are blanked.

    With ``store`` each scene is appended to that Zarr store as soon as it
    is done and the result is returned opened lazily from it, so memory
    holds one scene at a time however many scenes there are.
    """

    if model not in PER_SCENE_MODELS:
        raise ValueError(f"model must be one of {tuple(PER_SCENE_MODELS)}")
    if target not in cube:
        raise ValueError(f"Coarse cube has no {target!r} variable to downscale")
    matched_name = f"{predictor_sensor}_matched_time"
    if matched_name not in cube:
        raise ValueError(f"Coarse cube has no {matched_name!r}; was {predictor_sensor} fused into it?")
    selected = _available_predictors(cube, fine_predictors, terrain, predictors)
    fit = PER_SCENE_MODELS[model]
    options = dict(model_options or {})
    sensor_names = [name for name in selected if name in fine_predictors]
    # Static terrain on the fine grid once; it has no time axis to select.
    static_terrain = None
    terrain_names = [name for name in selected if name not in fine_predictors and name != _ILLUMINATION]
    if terrain_names and terrain is not None:
        static_terrain = terrain[terrain_names].reindex(y=fine_predictors.y, x=fine_predictors.x, method="nearest", tolerance=1e-6)

    scenes: list[xr.Dataset] = []
    written = 0
    skipped: list[str] = []
    for index, timestamp in enumerate(cube.time.values):
        label = str(np.datetime_as_string(timestamp, unit="s"))
        matched = cube[matched_name].values[index]
        if np.isnat(matched):
            skipped.append(f"{label}: no {predictor_sensor} acquisition within the temporal tolerance")
            continue
        scene = cube.isel(time=[index])
        fine = fine_predictors[sensor_names].sel(time=[matched]).assign_coords(time=[timestamp])
        fine.attrs.update(fine_predictors.attrs)
        if static_terrain is not None:
            fine = fine.assign({name: static_terrain[name].expand_dims(time=[timestamp]) for name in terrain_names})
        if _ILLUMINATION in selected and terrain is not None:
            fine[_ILLUMINATION] = _illumination(terrain, timestamp, fine)
        extra = {"fine": fine} if model in _USES_FINE else {}
        try:
            fitted = fit(scene, target=target, predictors=selected, min_samples=min_samples, **options, **extra)
        except CubeValidationError as exc:
            skipped.append(f"{label}: {exc}")
            continue
        result = CoarseConsistentDownscaler(fitted, target=target, correction=correction).predict(fine, scene) if coarse_consistent else fitted.predict(fine)
        observed = scene[target].notnull().reindex(y=fine.y, x=fine.x, method="nearest").fillna(False).astype(bool)
        result["coarse_observed"] = observed
        if mask_unobserved:
            for name in ("lst_downscaled", "lst_downscaled_raw", "lst_downscaled_uncertainty"):
                if name in result:
                    result[name] = result[name].where(observed)
        if domain is not None:
            for name in ("lst_downscaled", "lst_downscaled_raw", "lst_downscaled_uncertainty"):
                if name in result:
                    result[name] = result[name].where(domain)
        units = cube[target].attrs.get("units")
        for name in ("lst_downscaled", "lst_downscaled_raw"):
            if name in result and units:
                result[name].attrs["units"] = units
        for name in ("lst_downscaled_uncertainty", "coarse_consistency_correction"):
            if name in result:
                result[name].attrs["units"] = "K"
        crs = fine_predictors.attrs.get("crs")
        if crs:
            # On each map too, so a variable taken out of the dataset can still be masked or exported.
            for variable in result.data_vars.values():
                if {"y", "x"} <= set(variable.dims):
                    variable.attrs["crs"] = crs
        # Per-scene diagnostics become variables: as attributes they would
        # conflict between scenes and be dropped by the concat below.
        result["coarse_consistency_rmse"] = xr.DataArray([result.attrs.pop("coarse_consistency_rmse", np.nan)], dims="time", coords={"time": [timestamp]})
        result["downscaling_training_samples"] = xr.DataArray([int(result.attrs.pop("downscaling_training_samples", 0))], dims="time", coords={"time": [timestamp]})
        if store is not None:
            _append_scene(result, store, first=written == 0)
            written += 1
        else:
            scenes.append(result)
    done = written if store is not None else len(scenes)
    if not done:
        raise ValueError("No scene could be downscaled: " + "; ".join(skipped))
    if skipped:
        logger.warning("Per-scene downscaling skipped %d of %d scenes", len(skipped), cube.sizes["time"])
    attrs = {
        "crs": fine_predictors.attrs.get("crs"),
        "downscaling_protocol": "per_scene",
        "downscaling_model": model,
        "downscaling_predictors": list(selected),
        "downscaling_target": target,
        "downscaling_coarse_consistent": coarse_consistent,
        "downscaling_correction": correction if coarse_consistent else "none",
        "downscaling_mask_unobserved": mask_unobserved,
        "downscaling_scenes": done,
        "downscaling_skipped_scenes": skipped,
    }
    if store is not None:
        import zarr

        from ..storage import open_zarr

        zarr.open_group(str(store), mode="a").attrs.update(attrs)
        return open_zarr(store)
    downscaled = xr.concat(scenes, dim="time", combine_attrs="drop_conflicts").sortby("time")
    downscaled.attrs.update(attrs)
    return downscaled


def _append_scene(scene: xr.Dataset, store: str | Path, *, first: bool) -> None:
    """Write one scene to ``store``: create it for the first, append after."""

    from ..storage import write_zarr

    scene = scene.copy()
    scene.attrs = {}  # run-level attributes are written once, at the end
    if first:
        write_zarr(scene, store)
    else:
        scene.to_zarr(store, mode="a", append_dim="time", zarr_format=2, consolidated=False)


def _available_predictors(cube: xr.Dataset, fine_predictors: xr.Dataset, terrain: xr.Dataset | None, requested: Sequence[str] | None) -> list[str]:
    """Resolve the predictor list against what the coarse and fine data hold.

    Explicitly requested predictors must all be available; the default set
    silently drops whatever this run did not produce (e.g. terrain).
    """

    def available(name: str) -> bool:
        if name not in cube:
            return False
        if name == _ILLUMINATION:
            return terrain is not None
        return name in fine_predictors or (terrain is not None and name in terrain)

    if requested:
        missing = [name for name in requested if not available(name)]
        if missing:
            raise ValueError(f"Predictors not available on both the coarse and fine cubes: {missing}")
        return list(requested)
    selected = [name for name in DEFAULT_PREDICTORS if available(name)]
    if not selected:
        raise ValueError(f"None of the default predictors {DEFAULT_PREDICTORS} is available on both cubes")
    return selected


def _illumination(terrain: xr.Dataset, timestamp: np.datetime64, like: xr.Dataset) -> xr.DataArray:
    from ..sensors.terrain import terrain_predictors

    illumination = terrain_predictors(terrain, [timestamp], crs=terrain.attrs["crs"])[_ILLUMINATION]
    return illumination.reindex(y=like.y, x=like.x, method="nearest", tolerance=1e-6).assign_coords(time=like.time)
