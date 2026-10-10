"""Regression tests for ordering, footprint and catalogue-selection bugs."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from citycube import AOI, AnalysisRequest, AnalysisWorkflow, Sentinel2Catalog, align_features
from citycube.fusion import TemporalMatch
from citycube.harmonization.spatial import harmonize_spatial


def _cube(name: str, times: list[str], values: list[float], *, x=(0.0, 100.0), y=(100.0, 0.0), grid_id: str = "g") -> xr.Dataset:
    data = np.stack([np.full((len(y), len(x)), value) for value in values])
    return xr.Dataset(
        {name: (("time", "y", "x"), data)},
        coords={"time": np.array(times, dtype="datetime64[ns]"), "y": list(y), "x": list(x)},
        attrs={"crs": "EPSG:32633", "grid_id": grid_id},
    )


def test_align_features_matches_unsorted_feature_times_by_date():
    target = _cube("lst", ["2025-06-10", "2025-06-20"], [30.0, 31.0])
    # Feature times arrive in cloud-cover order, not chronological order.
    features = _cube("NDVI", ["2025-06-21", "2025-06-11"], [0.9, 0.1])

    aligned = align_features(target, features, match=TemporalMatch(np.timedelta64(2, "D")))

    assert aligned["NDVI"].sel(time="2025-06-10").values.ravel()[0] == pytest.approx(0.1)
    assert aligned["NDVI"].sel(time="2025-06-20").values.ravel()[0] == pytest.approx(0.9)
    assert aligned["matched_time"].values[0] == np.datetime64("2025-06-11", "ns")


def test_workflow_run_returns_chronological_cube_for_unsorted_inputs():
    target = _cube("lst", ["2025-06-20", "2025-06-10"], [31.0, 30.0])
    feature = _cube("NDVI", ["2025-06-21", "2025-06-11"], [0.9, 0.1], grid_id="feature")
    request = AnalysisRequest(aoi=AOI(0, 0, 1, 1), start="2025-06-10", end="2025-06-21", sensors=("sentinel3", "sentinel2"))

    result = AnalysisWorkflow(request).run({"sentinel3": target, "sentinel2": feature})

    times = result.cube.time.values
    assert np.all(times[:-1] < times[1:])
    assert result.cube["lst"].isel(time=0).values.ravel()[0] == pytest.approx(30.0)
    assert result.cube["NDVI"].isel(time=0).values.ravel()[0] == pytest.approx(0.1)


def test_same_crs_harmonization_does_not_extrapolate_outside_source_footprint():
    target = _cube("lst", ["2025-06-10"], [30.0], x=(0.0, 100.0, 200.0, 300.0), y=(100.0, 0.0))
    # The source only covers the western half of the target.
    source = _cube("NDVI", ["2025-06-10"], [0.5], x=(0.0, 100.0), y=(100.0, 0.0), grid_id="source")

    result = harmonize_spatial(target, source)

    row = result["NDVI"].isel(time=0, y=0).values
    assert np.isfinite(row[:2]).all()
    assert np.isnan(row[2:]).all()


def test_harmonized_boolean_masks_stay_boolean_and_false_outside_footprint():
    target = _cube("lst", ["2025-06-10"], [30.0], x=(0.0, 100.0, 200.0, 300.0), y=(100.0, 0.0))
    source = _cube("NDVI", ["2025-06-10"], [0.5], x=(0.0, 100.0), y=(100.0, 0.0), grid_id="source")
    source["valid_mask"] = xr.ones_like(source["NDVI"], dtype=bool)

    result = harmonize_spatial(target, source)

    assert result["valid_mask"].dtype == bool
    assert result["valid_mask"].isel(time=0, y=0).values.tolist() == [True, True, False, False]


def test_request_validates_dates_chronologically_not_lexically():
    # "2025-06-12 10:00:00" < "2025-06-12T09:00" as strings, so the old
    # str() comparison accepted this inverted range.
    with pytest.raises(ValueError, match="start must not be after end"):
        AnalysisRequest(aoi=AOI(0, 0, 1, 1), start=datetime(2025, 6, 12, 10), end="2025-06-12T09:00")
    # A date-only end covers the whole day.
    AnalysisRequest(aoi=AOI(0, 0, 1, 1), start="2025-06-12T10:00", end="2025-06-12")


def _odata_item(index: int, cloud: float) -> dict:
    return {
        "Id": f"id-{index}",
        "Name": f"S2A_MSIL2A_202506{index + 10:02d}T100000_N0511_R022_T33UUU_X.SAFE",
        "Online": True,
        "ContentDate": {"Start": f"2025-06-{index + 10:02d}T10:00:00Z", "End": f"2025-06-{index + 10:02d}T10:00:10Z"},
        "Attributes": [{"Name": "cloudCover", "Value": cloud}, {"Name": "productType", "Value": "S2MSI2A"}],
    }


class _PagedSession:
    def __init__(self, pages: list[list[dict]]):
        self.pages = pages
        self.calls = 0

    def get(self, url, *, params=None, timeout=None):
        page = self.pages[self.calls]
        self.calls += 1
        payload = {"value": page}
        if self.calls < len(self.pages):
            payload["@odata.nextLink"] = f"next-{self.calls}"

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return payload

        return Response()


def test_workflow_discovery_ranks_all_candidates_before_truncating(monkeypatch):
    # Early acquisitions are cloudy, the clear ones come last in date order.
    pages = [[_odata_item(0, 90.0), _odata_item(1, 80.0)], [_odata_item(2, 5.0), _odata_item(3, 1.0)]]
    session = _PagedSession(pages)
    monkeypatch.setattr("citycube.sensors.sentinel2.http_session", lambda: session)
    request = AnalysisRequest(aoi=AOI(13.2, 52.4, 13.6, 52.6), start="2025-06-10", end="2025-06-13", sensors=("sentinel2",), max_products_per_sensor=2)

    selected = AnalysisWorkflow(request).discover()["sentinel2"]

    assert [product.cloud_cover for product in selected] == [1.0, 5.0]


def test_sentinel2_catalog_search_still_honours_its_own_limit():
    session = _PagedSession([[_odata_item(0, 90.0), _odata_item(1, 80.0)], [_odata_item(2, 5.0)]])
    products = Sentinel2Catalog(session=session).search(AOI(13.2, 52.4, 13.6, 52.6), "2025-06-10", "2025-06-13", limit=2)
    assert len(products) == 2


def test_asset_cache_trusts_unchanged_stat_and_rehashes_only_on_change(tmp_path, monkeypatch):
    import os

    from citycube.cache import AssetCache

    asset = tmp_path / "product.zip"
    asset.write_bytes(b"original")
    cache = AssetCache(tmp_path / "cache")
    calls = {"count": 0}
    original = AssetCache.checksum

    def counting(path):
        calls["count"] += 1
        return original(path)

    monkeypatch.setattr(AssetCache, "checksum", staticmethod(counting))
    digest = cache.record("p", asset)
    assert calls["count"] == 1
    assert cache.record("p", asset) == digest  # unchanged file: hash reused
    assert cache.valid("p", asset) and cache.valid("p", asset)
    assert calls["count"] == 1

    assert cache.valid("p", asset, verify=True)
    assert calls["count"] == 2

    asset.write_bytes(b"corrupted")  # different size -> must re-hash and reject
    assert not cache.valid("p", asset)

    # Same size and mtime but different content is only caught by verify=True.
    asset.write_bytes(b"original")
    cache.record("p", asset)
    stat = asset.stat()
    asset.write_bytes(b"ORIGINAL")
    os.utime(asset, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert cache.valid("p", asset)
    assert not cache.valid("p", asset, verify=True)


def test_asset_cache_upgrades_legacy_entries_without_stat(tmp_path):
    import json

    from citycube.cache import AssetCache

    asset = tmp_path / "file.nc"
    asset.write_bytes(b"data")
    cache = AssetCache(tmp_path / "cache")
    cache.manifest_path.write_text(json.dumps({"k": {"path": str(asset), "sha256": AssetCache.checksum(asset), "metadata": {}}}))

    assert cache.valid("k", asset)
    assert "mtime_ns" in json.loads(cache.manifest_path.read_text())["k"]


def _linear_training(n_time: int = 120) -> xr.Dataset:
    rng = np.random.default_rng(42)
    ndvi = rng.random((n_time, 2, 2))
    return xr.Dataset(
        {"lst": (("time", "y", "x"), 20 + 5 * ndvi), "NDVI": (("time", "y", "x"), ndvi)},
        coords={"time": np.arange(n_time).astype("datetime64[D]"), "y": [0, 1], "x": [0, 1]},
        attrs={"crs": "EPSG:32613"},
    )


def test_fit_sklearn_downscaler_wraps_any_estimator_and_validates_with_one_prediction(monkeypatch):
    sklearn_linear = pytest.importorskip("sklearn.linear_model")
    from citycube import RandomForestDownscaler, SklearnDownscaler, fit_random_forest_downscaler, fit_sklearn_downscaler, validate_downscaler

    training = _linear_training()
    model = fit_sklearn_downscaler(training, sklearn_linear.LinearRegression(), predictors=("NDVI",), method="ols_sklearn")
    result = model.predict(training.isel(time=slice(0, 1)))
    assert isinstance(model, SklearnDownscaler)
    assert result.attrs["downscaling_method"] == "ols_sklearn"
    assert float(result["lst_downscaled"].isel(time=0, y=0, x=0)) == pytest.approx(float(training["lst"].isel(time=0, y=0, x=0)), abs=1e-6)

    forest = fit_random_forest_downscaler(training, predictors=("NDVI",), n_estimators=5)
    assert isinstance(forest, RandomForestDownscaler)
    assert forest.predict(training.isel(time=slice(0, 1))).attrs["downscaling_method"] == "random_forest"

    calls = {"count": 0}
    original = SklearnDownscaler.predict

    def counting(self, predictors):
        calls["count"] += 1
        return original(self, predictors)

    monkeypatch.setattr(SklearnDownscaler, "predict", counting)
    metrics = validate_downscaler(model, training.isel(time=slice(0, 5)))
    assert calls["count"] == 1
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-6)


def test_regridding_stays_lazy_for_dask_inputs_and_matches_eager_results():
    pytest.importorskip("dask")
    pytest.importorskip("rasterio")
    from citycube import reaggregate_to_target

    rng = np.random.default_rng(0)
    times = np.array(["2025-06-10", "2025-06-11", "2025-06-12", "2025-06-13"], dtype="datetime64[ns]")
    fine = xr.Dataset(
        {"NDVI": (("time", "y", "x"), rng.random((4, 8, 8)).astype(np.float32)), "valid_mask": (("time", "y", "x"), rng.random((4, 8, 8)) > 0.2)},
        coords={"time": times, "y": 4_000_075.0 - 10.0 * np.arange(8), "x": 500005.0 + 10.0 * np.arange(8)},
        attrs={"crs": "EPSG:32633"},
    )
    coarse = xr.Dataset(
        {"lst": (("time", "y", "x"), np.zeros((4, 2, 2)))},
        coords={"time": times, "y": [4_000_060.0, 4_000_020.0], "x": [500020.0, 500060.0]},
        attrs={"crs": "EPSG:32633"},
    )
    lazy_fine = fine.chunk({"time": 1})

    eager = reaggregate_to_target(fine["NDVI"], coarse["lst"])
    lazy = reaggregate_to_target(lazy_fine["NDVI"], coarse["lst"])
    assert lazy.chunks is not None
    np.testing.assert_allclose(lazy.compute().values, eager.values)
    assert lazy.dims == ("time", "y", "x")

    # Same CRS (mean path) and a CRS change (rasterio warp path; ETRS89 /
    # UTM 33N overlaps WGS84 / UTM 33N to within a metre, so real data lands).
    for target in (coarse, coarse.assign_attrs(crs="EPSG:25833")):
        eager_h = harmonize_spatial(target, fine)
        lazy_h = harmonize_spatial(target, lazy_fine)
        assert lazy_h["NDVI"].chunks is not None
        assert np.isfinite(eager_h["NDVI"].values).mean() > 0.5
        np.testing.assert_allclose(lazy_h["NDVI"].compute().values, eager_h["NDVI"].values, equal_nan=True)
        assert lazy_h["valid_mask"].compute().dtype == bool


def test_blocked_split_holds_out_whole_blocks_and_rejects_unsorted_time():
    from citycube import blocked_calibration_split, blocked_spatiotemporal_split

    cube = xr.Dataset(
        {"lst": (("time", "y", "x"), np.ones((10, 8, 8)))},
        coords={"time": np.arange(10).astype("datetime64[D]").astype("datetime64[ns]"), "y": np.arange(8.0), "x": np.arange(8.0)},
        attrs={"crs": "EPSG:32633"},
    )
    train, validation = blocked_spatiotemporal_split(cube, validation_fraction=0.2, spatial_block_period=2, block_size=4)
    held = np.isfinite(validation["lst"].isel(time=-1).values)
    # 4x4 blocks in a checkerboard: top-left and bottom-right blocks held out.
    assert held[:4, :4].all() and held[4:, 4:].all() and not held[:4, 4:].any()
    calibration = blocked_calibration_split(cube, validation_fraction=0.2, spatial_block_period=2, block_size=4)
    assert np.isfinite(calibration["lst"].isel(time=0).values)[:4, :4].all()
    assert np.isnan(calibration["lst"].isel(time=-1).values).all()
    assert not (np.isfinite(train["lst"].values) & np.isfinite(calibration["lst"].values)).any()

    with pytest.raises(ValueError, match="sorted"):
        blocked_spatiotemporal_split(cube.isel(time=[3, 1, 2, 0, 4, 5, 6, 7, 8, 9]))


def test_conformal_intervals_reach_target_coverage_and_widen_where_the_model_is_unsure():
    pytest.importorskip("sklearn")
    from citycube import (
        blocked_calibration_split,
        blocked_spatiotemporal_split,
        fit_conformal_downscaler,
        fit_linear_downscaler,
        fit_random_forest_downscaler,
        validate_downscaler,
    )

    rng = np.random.default_rng(7)
    shape = (60, 12, 12)
    ndvi = rng.random(shape)
    noise = rng.normal(size=shape) * (0.1 + 3.0 * ndvi**2)  # heteroscedastic: noisy where NDVI is high
    cube = xr.Dataset(
        {"lst": (("time", "y", "x"), 30 - 8 * ndvi + noise), "NDVI": (("time", "y", "x"), ndvi)},
        coords={"time": np.arange(shape[0]).astype("datetime64[D]").astype("datetime64[ns]"), "y": np.arange(12.0), "x": np.arange(12.0)},
        attrs={"crs": "EPSG:32633"},
    )
    split = {"validation_fraction": 0.3, "spatial_block_period": 2, "block_size": 3}
    train, validation = blocked_spatiotemporal_split(cube, **split)
    calibration = blocked_calibration_split(cube, **split)

    forest = fit_random_forest_downscaler(train, predictors=("NDVI",), n_estimators=60, min_samples=50)
    normalized = fit_conformal_downscaler(forest, calibration, alpha=0.1)
    constant = fit_conformal_downscaler(fit_linear_downscaler(train, predictors=("NDVI",)), calibration, alpha=0.1)
    assert normalized.normalized and not constant.normalized

    for model in (normalized, constant):
        metrics = validate_downscaler(model, validation)
        assert 0.85 <= metrics["interval_coverage"] <= 0.96

    widths = normalized.predict(validation)["lst_downscaled_uncertainty"]
    ndvi_values = validation["NDVI"]
    assert float(widths.where(ndvi_values > 0.8).mean()) > 1.5 * float(widths.where(ndvi_values < 0.2).mean())
    assert normalized.predict(validation).attrs["uncertainty_method"] == "normalized split conformal"


def test_linear_models_clip_predictors_to_the_training_range_and_flag_it():
    from citycube import fit_linear_downscaler, validate_downscaler

    model = fit_linear_downscaler(_linear_training(), predictors=("NDVI",))  # NDVI in [0, 1), lst = 20 + 5 * NDVI
    fine = xr.Dataset(
        {"NDVI": (("y", "x"), np.array([[0.5, 40.0]]))},  # 40: an EVI-like outlier never seen in training
        coords={"y": [0.0], "x": [0.0, 1.0]},
        attrs={"crs": "EPSG:32613"},
    )
    result = model.predict(fine)
    assert result["lst_downscaled"].values[0, 0] == pytest.approx(22.5, abs=1e-6)
    assert result["lst_downscaled"].values[0, 1] == pytest.approx(20 + 5 * model.predictor_max[0], abs=1e-6)
    assert result["downscaled_extrapolation"].values.tolist() == [[False, True]]

    validation = fine.assign(lst=(("y", "x"), np.array([[22.5, 25.0]]))).expand_dims(time=[np.datetime64("2025-06-10", "ns")])
    assert validate_downscaler(model, validation)["extrapolation_fraction"] == pytest.approx(0.5)


@pytest.mark.parametrize("ascending", [True, False])
def test_reprojection_keeps_north_up_for_ascending_and_descending_latitudes(ascending):
    """grid_s5p (and some ERA5/CAMS files) produce ascending y; the warp used
    to place row 0 at the top regardless and flipped them north-south."""

    pytest.importorskip("rasterio")
    from citycube import AnalysisGrid

    lat = np.arange(52.305, 52.6, 0.01)
    lat = lat if ascending else lat[::-1]
    lon = np.arange(13.205, 13.55, 0.01)
    times = np.array(["2025-06-10"], dtype="datetime64[ns]")
    source = xr.Dataset(
        {"field": (("time", "y", "x"), np.broadcast_to(lat[:, None], (lat.size, lon.size))[None].copy())},
        coords={"time": times, "y": lat, "x": lon},
        attrs={"crs": "EPSG:4326"},
    )
    grid = AnalysisGrid.for_aoi(AOI(13.25, 52.35, 13.5, 52.55), resolution_m=1000)
    target = xr.Dataset({"lst": (("time", "y", "x"), np.zeros((1, grid.height, grid.width)))}, coords={"time": times, "y": grid.y, "x": grid.x}, attrs={"crs": grid.crs})

    column = harmonize_spatial(target, source)["field"].isel(time=0, x=grid.width // 2).values
    assert np.all(np.diff(column[np.isfinite(column)]) < 0)  # grid.y runs north to south


def test_asset_cache_keeps_every_entry_under_concurrent_writers(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from citycube.cache import AssetCache

    files = []
    for index in range(40):
        path = tmp_path / f"asset-{index}.nc"
        path.write_bytes(bytes([index]) * 100)
        files.append(path)

    def record(path):
        # A fresh instance per call, like AcquisitionContext.subset() does.
        AssetCache(tmp_path / "cache").record(path.name, path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(record, files))

    cache = AssetCache(tmp_path / "cache")
    assert all(cache.valid(path.name, path) for path in files)
    assert not list((tmp_path / "cache").glob("*.part"))
