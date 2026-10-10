"""Overpass filtering, per-product error isolation, request limits and per-scene downscaling."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from citycube import AOI, AnalysisRequest, AnalysisWorkflow, DownscaleSpec, downscale_per_scene, filter_overpass, local_solar_hour
from citycube.catalog import ProductRef
from citycube.cube import AnalysisGrid
from citycube.workflow import ProductAcquisitionError, RequestLimits, RequestTooLargeError, check_request, estimate_request
from citycube.workflow.adapters import AcquisitionContext

CRS = "EPSG:32633"


def _product(name: str, start: str | None, cloud: float = 10.0) -> ProductRef:
    return ProductRef(
        product_id=name, name=name, product_type="SL_2_LST___", start_datetime=start, end_datetime=start,
        timeliness="NT", online=True, download_url="", metadata={"attributes": {"cloudCover": cloud}},
    )


def _request(**kwargs) -> AnalysisRequest:
    values = {"aoi": AOI(13.2, 52.4, 13.6, 52.6), "start": "2025-06-10", "end": "2025-06-13", "sensors": ("sentinel3",)}
    return AnalysisRequest(**{**values, **kwargs})


# --- thermal overpass -----------------------------------------------------


def test_local_solar_hour_shifts_utc_by_longitude():
    assert local_solar_hour("2025-06-10T09:30:00Z", 15.0) == pytest.approx(10.5)
    assert local_solar_hour("2025-06-10T23:00:00Z", 30.0) == pytest.approx(1.0)
    assert local_solar_hour("2025-06-10T12:00:00+00:00", -90.0) == pytest.approx(6.0)


def test_filter_overpass_separates_day_and_night_and_drops_unknown_times():
    # Berlin (~13.4 E): 09:30 UTC is ~10:24 solar (day), 20:30 UTC ~21:24 (night).
    products = [_product("day", "2025-06-10T09:30:00Z"), _product("night", "2025-06-10T20:30:00Z"), _product("unknown", None)]

    assert [p.name for p in filter_overpass(products, longitude=13.4, overpass="day")] == ["day"]
    assert [p.name for p in filter_overpass(products, longitude=13.4, overpass="night")] == ["night"]
    assert [p.name for p in filter_overpass(products, longitude=13.4, overpass="any")] == ["day", "night", "unknown"]
    with pytest.raises(ValueError):
        filter_overpass(products, longitude=13.4, overpass="dusk")


def test_discovery_filters_night_passes_before_ranking(monkeypatch):
    # The clearest products are night passes; a day-only request must keep
    # the best *daytime* products instead of losing its slots to them.
    candidates = [
        _product("night-clear", "2025-06-10T20:30:00Z", cloud=0.0),
        _product("night-clear-2", "2025-06-11T20:30:00Z", cloud=1.0),
        _product("day-cloudy", "2025-06-10T09:30:00Z", cloud=50.0),
        _product("day-clear", "2025-06-11T09:30:00Z", cloud=5.0),
    ]
    from citycube.workflow import runner

    class FakeAdapter:
        def search(self, request, *, limit):
            return candidates

    monkeypatch.setitem(runner.SENSOR_ADAPTERS, "sentinel3", FakeAdapter())  # type: ignore[arg-type]

    selected = AnalysisWorkflow(_request(thermal_overpass="day", max_products_per_sensor=2)).discover()["sentinel3"]

    assert [p.name for p in selected] == ["day-clear", "day-cloudy"]


# --- per-product error isolation ------------------------------------------


def _context(tmp_path: Path, **request_kwargs) -> AcquisitionContext:
    def no_downloader():
        raise AssertionError("not needed")

    return AcquisitionContext(request=_request(**request_kwargs), output_dir=tmp_path, downloader_factory=no_downloader)


def _read(reference: ProductRef) -> str:
    if reference.name == "corrupt":
        raise OSError("truncated zip")
    return reference.name


def test_map_products_skips_and_records_a_failing_product(tmp_path):
    context = _context(tmp_path)
    references = [_product("a", "2025-06-10T09:30:00Z"), _product("corrupt", "2025-06-11T09:30:00Z"), _product("b", "2025-06-12T09:30:00Z")]

    assert context.map_products("sentinel3", references, _read) == ["a", "b"]
    assert context.failures == [{"sensor": "sentinel3", "product": "corrupt", "error": "OSError: truncated zip"}]


def test_map_products_raise_policy_aborts_on_the_first_failure(tmp_path):
    context = _context(tmp_path, on_product_error="raise")

    with pytest.raises(OSError, match="truncated zip"):
        context.map_products("sentinel3", [_product("corrupt", None)], _read)


def test_map_products_fails_when_every_product_fails(tmp_path):
    context = _context(tmp_path)

    with pytest.raises(ProductAcquisitionError, match="All 2 sentinel3 products failed.*truncated zip"):
        context.map_products("sentinel3", [_product("corrupt", None), _product("corrupt", None)], _read)


def test_execute_reports_skipped_products_in_provenance(tmp_path, monkeypatch):
    from citycube.workflow import runner

    class FlakyAdapter:
        def search(self, request, *, limit):
            return [_product("good", "2025-06-10T09:30:00Z"), _product("corrupt", "2025-06-11T09:30:00Z")]

        def acquire(self, references, context):
            def read(reference):
                if reference.name == "corrupt":
                    raise OSError("truncated zip")
                return _coarse_scene(np.datetime64("2025-06-10T09:30"))

            return context.map_products("sentinel3", references, read)[0]

    monkeypatch.setitem(runner.SENSOR_ADAPTERS, "sentinel3", FlakyAdapter())  # type: ignore[arg-type]

    result = AnalysisWorkflow(_request()).execute(tmp_path)

    assert [item["product"] for item in result.provenance["failed_products"]] == ["corrupt"]


# --- request limits -------------------------------------------------------


def _gridded_request(resolution_m: float, **kwargs) -> AnalysisRequest:
    aoi = AOI(13.2, 52.4, 13.6, 52.6)
    grid = AnalysisGrid.for_aoi(aoi, resolution_m=resolution_m)
    thermal = AnalysisGrid.for_aoi(aoi, resolution_m=1000, crs=grid.crs)
    return _request(aoi=aoi, sensors=("sentinel3", "sentinel2"), grid=grid, predictor_grid=grid, thermal_grid=thermal, **kwargs)


def test_estimate_scales_with_resolution_and_products():
    coarse = estimate_request(_gridded_request(100, max_products_per_sensor=10))
    fine = estimate_request(_gridded_request(10, max_products_per_sensor=10))
    more = estimate_request(_gridded_request(100, max_products_per_sensor=20))

    assert 550 < coarse.aoi_km2 < 650  # 0.4 x 0.2 degrees at 52.5 N
    assert fine.cells["sentinel2"] == pytest.approx(100 * coarse.cells["sentinel2"], rel=0.05)
    assert fine.estimated_bytes > 90 * coarse.estimated_bytes
    assert more.estimated_bytes == pytest.approx(2 * coarse.estimated_bytes, rel=0.01)


def test_check_request_lists_every_exceeded_limit():
    request = _gridded_request(10, max_products_per_sensor=100)

    with pytest.raises(RequestTooLargeError) as error:
        check_request(request, RequestLimits(max_aoi_km2=100, max_products_per_sensor=50, max_estimated_gb=1))

    message = str(error.value)
    assert "AOI is" in message and "max_products_per_sensor is 100" in message and "estimated cube size" in message
    assert check_request(_gridded_request(100, max_products_per_sensor=10)).estimated_gb < 1


def test_execute_rejects_an_oversized_request_before_any_download(tmp_path, monkeypatch):
    from citycube.workflow import runner

    monkeypatch.setattr(runner.AnalysisWorkflow, "discover", lambda self: pytest.fail("discovery must not run"))

    with pytest.raises(RequestTooLargeError):
        AnalysisWorkflow(_gridded_request(10)).execute(tmp_path, limits=RequestLimits(max_estimated_gb=0.5))


# --- request serialization ------------------------------------------------


def test_new_request_fields_round_trip_and_validate():
    request = _request(
        sensors=("sentinel3", "sentinel2"), thermal_overpass="day", on_product_error="raise",
        downscale={"model": "linear", "predictors": ["NDVI"], "min_samples": 10},
    )

    restored = AnalysisRequest.from_dict(request.to_dict())

    assert restored.thermal_overpass == "day"
    assert restored.on_product_error == "raise"
    assert restored.downscale == DownscaleSpec(model="linear", predictors=("NDVI",), min_samples=10)
    with pytest.raises(ValueError, match="thermal_overpass"):
        _request(thermal_overpass="dusk")
    with pytest.raises(ValueError, match="on_product_error"):
        _request(on_product_error="ignore")
    with pytest.raises(ValueError, match="predictor sensor 'sentinel2'"):
        _request(downscale=DownscaleSpec())
    with pytest.raises(ValueError, match="downscale model"):
        DownscaleSpec(model="cnn")


# --- per-scene downscaling ------------------------------------------------

# 10 x 10 coarse cells of 1 km over 100 x 100 fine cells of 100 m.
_FINE_X = np.arange(50.0, 10_000, 100)
_FINE_Y = _FINE_X[::-1].copy()
_COARSE_X = np.arange(500.0, 10_000, 1000)
_COARSE_Y = _COARSE_X[::-1].copy()


def _ndvi(seed: int) -> np.ndarray:
    yy, xx = np.meshgrid(_FINE_Y, _FINE_X, indexing="ij")
    rng = np.random.default_rng(seed)
    return 0.5 + 0.3 * np.sin(xx / 900.0 + seed) * np.cos(yy / 1300.0) + 0.05 * rng.standard_normal(xx.shape)


def _block_mean(values: np.ndarray) -> np.ndarray:
    return values.reshape(10, 10, 10, 10).mean(axis=(1, 3))


def _fine_predictors(times: list[np.datetime64]) -> xr.Dataset:
    ndvi = np.stack([_ndvi(index) for index in range(len(times))])
    return xr.Dataset({"NDVI": (("time", "y", "x"), ndvi)}, coords={"time": times, "y": _FINE_Y, "x": _FINE_X}, attrs={"crs": CRS})


def _coarse_scene(time: np.datetime64, *, level: float = 40.0, slope: float = -20.0, seed: int = 0) -> xr.Dataset:
    ndvi = _ndvi(seed)
    lst = _block_mean(level + slope * ndvi)
    return xr.Dataset(
        {"lst": (("time", "y", "x"), lst[None]), "NDVI": (("time", "y", "x"), _block_mean(ndvi)[None])},
        coords={"time": [time], "y": _COARSE_Y, "x": _COARSE_X},
        attrs={"crs": CRS},
    )


def test_downscale_per_scene_fits_each_scene_to_its_own_level_and_conserves_it():
    # Two scenes with different temperature levels *and* slopes: a pooled
    # model would blur both; per-scene fits recover each exactly.
    times = [np.datetime64("2025-06-10T10:00", "ns"), np.datetime64("2025-06-12T10:00", "ns")]
    s2_times = [np.datetime64("2025-06-10T10:30", "ns"), np.datetime64("2025-06-12T10:30", "ns")]
    cube = xr.concat([_coarse_scene(times[0], level=40, slope=-20, seed=0), _coarse_scene(times[1], level=25, slope=-8, seed=1)], dim="time")
    cube["sentinel2_matched_time"] = ("time", np.array(s2_times))

    downscaled = downscale_per_scene(cube, _fine_predictors(s2_times), predictors=["NDVI"], model="linear", min_samples=10, correction="block")

    assert list(downscaled.time.values) == times
    for index, (level, slope) in enumerate(((40, -20), (25, -8))):
        expected = level + slope * _ndvi(index)
        # Fine NDVI beyond the coarse training range is clipped (and flagged)
        # by the linear model; everywhere else the relation is recovered.
        inside = ~downscaled["downscaled_extrapolation"].isel(time=index).values
        assert inside.mean() > 0.8
        assert np.allclose(downscaled["lst_downscaled_raw"].isel(time=index).values[inside], expected[inside], atol=1e-6)
        aggregated = _block_mean(downscaled["lst_downscaled"].isel(time=index).values)
        assert np.allclose(aggregated, cube["lst"].isel(time=index).values, atol=1e-6)
    assert downscaled.attrs["downscaling_protocol"] == "per_scene"
    assert downscaled.attrs["downscaling_skipped_scenes"] == []
    assert downscaled["coarse_consistency_rmse"].values.max() < 1e-6


def test_downscale_per_scene_skips_unmatched_and_too_cloudy_scenes():
    times = [np.datetime64(f"2025-06-1{day}T10:00", "ns") for day in (0, 1, 2)]
    s2_time = np.datetime64("2025-06-10T10:30", "ns")
    cloudy = _coarse_scene(times[1])
    cloudy["lst"][:] = np.nan
    cube = xr.concat([_coarse_scene(times[0]), cloudy, _coarse_scene(times[2])], dim="time")
    cube["sentinel2_matched_time"] = ("time", np.array([s2_time, s2_time, np.datetime64("NaT", "ns")]))

    downscaled = downscale_per_scene(cube, _fine_predictors([s2_time]), predictors=["NDVI"], model="linear", min_samples=10)

    assert list(downscaled.time.values) == [times[0]]
    skipped = downscaled.attrs["downscaling_skipped_scenes"]
    assert len(skipped) == 2 and "complete samples" in skipped[0] and "no sentinel2 acquisition" in skipped[1]


def test_downscale_per_scene_requires_explicit_predictors_to_exist():
    time = np.datetime64("2025-06-10T10:00", "ns")
    cube = _coarse_scene(time)
    cube["sentinel2_matched_time"] = ("time", np.array([time]))

    with pytest.raises(ValueError, match="EVI"):
        downscale_per_scene(cube, _fine_predictors([time]), predictors=["NDVI", "EVI"])
    # The default predictor set silently drops what the run did not produce.
    result = downscale_per_scene(cube, _fine_predictors([time]), model="linear", min_samples=10)
    assert result.attrs["downscaling_predictors"] == ["NDVI"]


def test_workflow_run_downscales_and_saves_the_result(tmp_path):
    pytest.importorskip("zarr")
    times = [np.datetime64("2025-06-10T10:00", "ns"), np.datetime64("2025-06-12T10:00", "ns")]
    s2_times = [np.datetime64("2025-06-10T10:30", "ns"), np.datetime64("2025-06-12T10:30", "ns")]
    thermal = xr.concat([_coarse_scene(t, seed=i)[["lst"]] for i, t in enumerate(times)], dim="time")
    request = _request(sensors=("sentinel3", "sentinel2"), downscale=DownscaleSpec(model="linear", predictors=("NDVI",), min_samples=10))

    result = AnalysisWorkflow(request).run({"sentinel3": thermal, "sentinel2": _fine_predictors(s2_times)})

    assert result.downscaled is not None
    assert result.downscaled.sizes == {"time": 2, "y": 100, "x": 100}
    assert result.provenance["downscaling"]["scenes"] == 2
    saved = result.save(tmp_path / "result")
    assert (saved / "downscaled.zarr").exists()


def test_experimental_sentinel1_backends_warn_but_are_accepted():
    with pytest.warns(UserWarning, match="hyp3_rtc.*experimental"):
        _request(sensors=("sentinel1",), sentinel1_backend="hyp3_rtc")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _request(sensors=("sentinel3",), sentinel1_backend="hyp3_rtc")  # unused backend: no warning


# --- build info and credential checks ---------------------------------------


def test_provenance_records_the_software_version_and_commit(monkeypatch):
    from citycube import version

    version.build_info.cache_clear()
    monkeypatch.setenv(version.COMMIT_ENV, "abc123")
    try:
        times = [np.datetime64("2025-06-10T10:00", "ns")]
        result = AnalysisWorkflow(_request()).run({"sentinel3": _coarse_scene(times[0])[["lst"]]})
    finally:
        version.build_info.cache_clear()
    assert result.provenance["software"]["git_commit"] == "abc123"
    assert result.provenance["software"]["version"]


def test_credential_checks_report_expiry_and_never_raise(monkeypatch):
    import base64
    import json as json_module
    from datetime import datetime, timedelta, timezone

    import requests

    from citycube import credentials

    def jwt(expires: datetime) -> str:
        payload = base64.urlsafe_b64encode(json_module.dumps({"exp": int(expires.timestamp())}).encode()).decode().rstrip("=")
        return f"header.{payload}.signature"

    class Response:
        def __init__(self, status):
            self.status_code = status

    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if "openaq" in url:
            raise requests.ConnectionError("down")
        return Response(200)

    for name in ("CDSE_CLIENT_ID", "CDSE_CLIENT_SECRET", "SH_CLIENT_ID", "SH_CLIENT_SECRET", "CDSE_USERNAME", "CDSE_PASSWORD",
                 "CDS_API_URL", "CDS_API_KEY", "CDSAPI_URL", "CDSAPI_KEY", "CAMS_API_URL", "CAMS_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CDSAPI_RC", "/nonexistent")
    monkeypatch.setenv("CDSE_CLIENT_ID", "id")
    monkeypatch.setenv("CDSE_CLIENT_SECRET", "secret")
    monkeypatch.setenv("EARTHDATA_BEARER_TOKEN", jwt(datetime.now(timezone.utc) + timedelta(days=3)))
    monkeypatch.setenv("OPENAQ_API_KEY", "key")
    monkeypatch.setattr(credentials.requests, "get", fake_get)
    monkeypatch.setattr(credentials.requests, "post", lambda url, **kwargs: Response(401))

    results = credentials.check_credentials(env_file=None)

    assert results["cdse_oauth_client"].status == "error" and results["cdse_oauth_client"].detail == "rejected: HTTP 401"
    assert results["cdse_account"].status == "not_configured"
    assert results["earthdata"].status == "warning" and "expires in" in results["earthdata"].detail
    assert results["cds"].status == "not_configured"
    assert results["openaq"].status == "error" and results["openaq"].detail == "unreachable: ConnectionError"
    assert credentials._expiry("x", datetime.now(timezone.utc) - timedelta(days=1)).status == "error"


# --- auxiliary providers ----------------------------------------------------


def test_cams_forecast_requests_two_runs_with_short_leads_only():
    from citycube import CAMSConfig, CAMSProvider

    provider = CAMSProvider(CAMSConfig(dataset="cams-global-atmospheric-composition-forecasts"))
    request = provider.build_request(AOI(13.2, 52.4, 13.6, 52.6), "2026-08-01", "2026-08-02", variables=("NO2",))

    assert request["time"] == ["00:00", "12:00"]
    assert request["leadtime_hour"] == ["0", "3", "6", "9"]
    assert request["type"] == ["forecast"]


def test_forecast_axes_become_one_time_axis_keeping_the_freshest_forecast():
    from citycube.providers.base import forecast_to_time

    runs = np.array(["2026-08-01T00", "2026-08-01T12"], dtype="datetime64[ns]")
    leads = np.array([0, 3, 6, 9, 12, 15], dtype="timedelta64[h]").astype("timedelta64[ns]")
    # value = lead in hours, so the kept value says which forecast survived
    values = np.broadcast_to((leads / np.timedelta64(1, "h"))[:, None, None, None], (6, 2, 1, 1)).copy()
    dataset = xr.Dataset(
        {"tcno2": (("forecast_period", "forecast_reference_time", "latitude", "longitude"), values)},
        coords={"forecast_period": leads, "forecast_reference_time": runs, "latitude": [52.5], "longitude": [13.4],
                "valid_time": (("forecast_period", "forecast_reference_time"), runs[None, :] + leads[:, None])},
    )

    flat = forecast_to_time(dataset)

    expected = np.arange(np.datetime64("2026-08-01T00", "ns"), np.datetime64("2026-08-02T04", "ns"), np.timedelta64(3, "h"))
    assert np.array_equal(flat.time.values, expected)
    by_time = dict(zip(flat.time.values, flat["tcno2"].values.ravel()))
    assert by_time[np.datetime64("2026-08-01T12", "ns")] == 0  # the 12 UTC run's analysis, not the 00 run's +12 h
    assert by_time[np.datetime64("2026-08-01T09", "ns")] == 9
    assert forecast_to_time(flat) is flat  # no forecast axes: unchanged


def test_write_netcdf_coerces_attributes_netcdf_cannot_store(tmp_path):
    from citycube import write_netcdf

    cube = _coarse_scene(np.datetime64("2025-06-10T10:00", "ns"))
    cube.attrs.update({"flag": True, "nothing": None, "skipped": ["a", "b"], "counts": [1, 2], "bools": [True, False]})
    cube["lst"].attrs["reconstructed"] = np.bool_(True)

    path = write_netcdf(cube, tmp_path / "cube.nc")

    with xr.open_dataset(path) as reopened:
        assert reopened.attrs["flag"] == 1 and "nothing" not in reopened.attrs
        assert reopened.attrs["skipped"] == "['a', 'b']"
        assert list(reopened.attrs["bools"]) == [1, 0]
        assert reopened["lst"].attrs["reconstructed"] == 1
        assert np.allclose(reopened["lst"].values, cube["lst"].values)


def test_sentinel1_indices_keep_the_metadata_contract_the_fusion_step_validates():
    from citycube.metadata import validate_variable_contract
    from citycube.sensors.sentinel1 import sentinel1_indices

    attrs = {"units": "linear", "sensor": "Sentinel-1", "product": "sentinel-1-rtc", "aggregation_method": "mean", "standard_name": "surface_backwards_scattering_coefficient_of_radar_wave"}
    band = np.full((1, 2, 2), 0.05)
    dataset = xr.Dataset(
        {"gamma0_VV": (("time", "y", "x"), band, attrs), "gamma0_VH": (("time", "y", "x"), band / 5, attrs)},
        coords={"time": [np.datetime64("2026-08-01", "ns")], "y": [1.0, 0.0], "x": [0.0, 1.0]},
        attrs={"crs": CRS},
    )

    indices = sentinel1_indices(dataset)

    validate_variable_contract(indices)  # previously failed on VH_VV_ratio_dB/RVI/radar_span
    assert indices["RVI"].attrs["sensor"] == "Sentinel-1"


def test_hourly_auxiliary_data_does_not_inflate_the_predictors():
    # An outer time join with hourly ERA5 used to stretch every Sentinel-2
    # variable to 24 steps a day: a 0.1 GB request needed over 6 GB.
    from citycube import AuxiliarySpec

    times = [np.datetime64("2025-06-10T10:00", "ns"), np.datetime64("2025-06-11T10:00", "ns")]
    thermal = xr.concat([_coarse_scene(t, seed=i)[["lst"]] for i, t in enumerate(times)], dim="time")
    hours = np.arange(np.datetime64("2025-06-10T00", "ns"), np.datetime64("2025-06-12T00", "ns"), np.timedelta64(1, "h"))
    era5 = xr.Dataset(
        {"air_temperature_2m": (("time", "y", "x"), np.full((hours.size, 10, 10), 290.0), {"units": "K", "sensor": "era5", "product": "era5", "aggregation_method": "native"})},
        coords={"time": hours, "y": _COARSE_Y, "x": _COARSE_X},
        attrs={"crs": CRS, "analysis_shape": "regular_grid"},
    )
    request = _request(sensors=("sentinel3", "sentinel2"), auxiliary=(AuxiliarySpec("era5", variables=("air_temperature_2m",)),))

    result = AnalysisWorkflow(request).run({"sentinel3": thermal, "sentinel2": _fine_predictors([np.datetime64("2025-06-10T10:30", "ns")]), "era5": era5})

    assert result.predictors["sentinel2"].sizes["time"] == 1
    assert not any(name.startswith("auxiliary_") for name in result.predictors["sentinel2"].data_vars)
    assert "auxiliary_era5_air_temperature_2m" in result.cube
    assert result.auxiliary["era5"].sizes["time"] == 48


def test_estimate_counts_hourly_auxiliary_sources():
    from citycube import AuxiliarySpec

    plain = _gridded_request(100, max_products_per_sensor=10)
    with_era5 = dataclasses.replace(plain, auxiliary=(AuxiliarySpec("era5", variables=("air_temperature_2m", "boundary_layer_height")),))
    extra = estimate_request(with_era5).estimated_bytes - estimate_request(plain).estimated_bytes
    days, steps, cells, variables = 4, 24, with_era5.thermal_grid.width * with_era5.thermal_grid.height, 2
    assert extra == days * steps * cells * variables * 8


def test_zarr_round_trips_microsecond_times_and_gaps_exactly(tmp_path):
    pytest.importorskip("zarr")
    from citycube import open_zarr, write_netcdf, write_zarr

    times = np.array(["2026-08-09T08:57:17.033925", "2026-08-10T09:00:00.000001"], dtype="datetime64[ns]")
    matched = np.array(["2026-08-09T10:15:59.024000", "NaT"], dtype="datetime64[ns]")  # no match on day two
    cube = xr.Dataset({"sentinel2_matched_time": ("time", matched), "lst": ("time", [30.0, 31.0])}, coords={"time": times})

    write_zarr(cube, tmp_path / "direct.zarr")
    # The same cube after a NetCDF round trip, which leaves its own encodings behind.
    write_zarr(xr.open_dataset(write_netcdf(cube, tmp_path / "cube.nc")).load(), tmp_path / "via_netcdf.zarr")

    for store in ("direct.zarr", "via_netcdf.zarr"):
        back = open_zarr(tmp_path / store).load()
        assert np.array_equal(back.time.values, times)
        assert back["sentinel2_matched_time"].values[0] == matched[0]
        assert np.isnat(back["sentinel2_matched_time"].values[1])


def test_sentinel5p_pixels_fill_their_footprint_not_just_one_cell():
    from citycube import grid_s5p

    # Two TROPOMI-sized pixels (5.5 x 3.5 km) over a 0.01 degree (~1 km) grid.
    swath = xr.Dataset(
        {"NO2": (("time", "observation"), [[1.0, 2.0]]), "valid_mask": (("time", "observation"), [[True, True]])},
        coords={"time": [np.datetime64("2026-08-03T11:12", "ns")], "latitude": (("time", "observation"), [[52.47, 52.53]]), "longitude": (("time", "observation"), [[13.35, 13.45]])},
        attrs={"gas": "NO2"},
    )
    aoi = AOI(13.30, 52.42, 13.50, 52.58)

    centres_only = grid_s5p(swath, resolution_deg=0.01, aoi=aoi, footprint_radius_km=None)
    filled = grid_s5p(swath, resolution_deg=0.01, aoi=aoi)

    assert int(centres_only["NO2"].notnull().sum()) == 2
    cells = filled["NO2"].isel(time=0)
    assert int(cells.notnull().sum()) > 40  # each 3.5 km footprint covers dozens of 1 km cells
    assert float(cells.sel(y=52.47, x=13.35, method="nearest")) == 1.0
    assert float(cells.sel(y=52.53, x=13.45, method="nearest")) == 2.0
    assert np.isnan(float(cells.sel(y=52.425, x=13.495, method="nearest")))  # far from both pixels


def test_smooth_correction_removes_coarse_steps_and_still_conserves():
    time, s2_time = np.datetime64("2025-06-10T10:00", "ns"), np.datetime64("2025-06-10T10:30", "ns")
    cube = _coarse_scene(time)
    # Departures from the NDVI relation that only the correction can explain.
    cube["lst"] = cube["lst"] + np.random.default_rng(3).normal(0, 2, cube["lst"].shape)
    cube["sentinel2_matched_time"] = ("time", np.array([s2_time]))

    def run(correction):
        out = downscale_per_scene(cube, _fine_predictors([s2_time]), predictors=["NDVI"], model="linear", min_samples=10, correction=correction)
        correction_field = out["coarse_consistency_correction"].isel(time=0).values
        edge_jump = np.abs(np.diff(correction_field, axis=1))[:, 9::10].mean()  # across 1 km cell edges
        return out, edge_jump

    block, block_jump = run("block")
    smooth, smooth_jump = run("smooth")

    assert smooth_jump < block_jump / 2
    assert float(smooth["coarse_consistency_rmse"].values[0]) < 0.1
    aggregated = _block_mean(smooth["lst_downscaled"].isel(time=0).values)
    assert np.abs(aggregated - cube["lst"].isel(time=0).values).max() < 0.2
    assert smooth.attrs["downscaling_correction"] == "smooth"


def test_atpk_correction_conserves_exactly_and_recovers_a_smooth_unexplained_field():
    # A warm plume the predictors cannot explain: only the residual step can place it.
    time, s2_time = np.datetime64("2025-06-10T10:00", "ns"), np.datetime64("2025-06-10T10:30", "ns")
    ndvi = _ndvi(0)
    xx, yy = np.meshgrid(_FINE_X, _FINE_Y)
    plume = 4 * np.exp(-(((xx - 3_000) / 2_500) ** 2 + ((yy - 6_000) / 2_500) ** 2))
    truth = 40 - 20 * ndvi + plume
    cube = _coarse_scene(time)
    cube["lst"] = (("time", "y", "x"), _block_mean(truth)[None])
    cube["sentinel2_matched_time"] = ("time", np.array([s2_time]))

    def run(correction):
        out = downscale_per_scene(cube, _fine_predictors([s2_time]), predictors=["NDVI"], model="linear", min_samples=10, correction=correction)
        values = out["lst_downscaled"].isel(time=0).values
        edge_jump = np.abs(np.diff(out["coarse_consistency_correction"].isel(time=0).values, axis=1))[:, 9::10].mean()
        return out, float(np.sqrt(np.mean((values - truth) ** 2))), edge_jump

    _, block_rmse, block_jump = run("block")
    smooth, smooth_rmse, _ = run("smooth")
    atpk, atpk_rmse, atpk_jump = run("atpk")

    assert atpk_rmse < block_rmse and atpk_rmse <= smooth_rmse * 1.05
    assert atpk_jump < block_jump / 2
    np.testing.assert_allclose(_block_mean(atpk["lst_downscaled"].isel(time=0).values), cube["lst"].isel(time=0).values, atol=1e-6)
    assert atpk.attrs["downscaling_correction"] == "atpk"


def test_atpk_falls_back_to_smooth_when_too_few_cells_are_observed():
    time, s2_time = np.datetime64("2025-06-10T10:00", "ns"), np.datetime64("2025-06-10T10:30", "ns")
    cube = _coarse_scene(time)
    cube["lst"][:, :, 3:] = np.nan  # only 30 cells observed, and most of them in one strip
    cube["lst"][:, 3:, :] = np.nan  # 9 cells: too few to fit a covariance
    cube["sentinel2_matched_time"] = ("time", np.array([s2_time]))
    out = downscale_per_scene(cube, _fine_predictors([s2_time]), predictors=["NDVI"], model="linear", min_samples=5, correction="atpk")
    assert np.isfinite(out["lst_downscaled"].isel(time=0, y=slice(0, 30), x=slice(0, 30))).all()


def test_unobserved_coarse_cells_are_masked_unless_asked_to_fill():
    time, s2_time = np.datetime64("2025-06-10T10:00", "ns"), np.datetime64("2025-06-10T10:30", "ns")
    cube = _coarse_scene(time)
    cube["lst"][:, :3, :] = np.nan  # a cloud over the northern three rows of 1 km cells
    cube["lst"].attrs["units"] = "degC"
    cube["sentinel2_matched_time"] = ("time", np.array([s2_time]))
    fine = _fine_predictors([s2_time])

    masked = downscale_per_scene(cube, fine, predictors=["NDVI"], model="linear", min_samples=10)
    filled = downscale_per_scene(cube, fine, predictors=["NDVI"], model="linear", min_samples=10, mask_unobserved=False)

    assert masked["lst_downscaled"].isel(time=0, y=slice(0, 30)).isnull().all()
    assert masked["lst_downscaled"].isel(time=0, y=slice(30, None)).notnull().all()
    assert not masked["coarse_observed"].isel(time=0, y=0).any()
    assert filled["lst_downscaled"].isel(time=0).notnull().all()
    assert masked["lst_downscaled"].attrs["units"] == "degC"
    assert masked["coarse_consistency_correction"].attrs["units"] == "K"


def test_era5_land_rejects_boundary_layer_height_before_queueing(tmp_path):
    from citycube import AuxiliarySpec, ERA5Provider

    with pytest.raises(ValueError, match="reanalysis-era5-single-levels"):
        ERA5Provider().download(AOI(13.2, 52.4, 13.6, 52.6), "2026-08-01", "2026-08-02", AuxiliarySpec("era5", variables=("boundary_layer_height",)), tmp_path)


def test_a_single_row_coarse_source_still_regrids_onto_the_aoi():
    # A city-sized AOI can fall inside one 0.25 degree ERA5 row.
    pytest.importorskip("rasterio")
    from citycube.workflow.adapters import to_grid

    era5 = xr.Dataset(
        {"air_temperature_2m": (("time", "y", "x"), [[[290.0, 292.0]]])},
        coords={"time": [np.datetime64("2026-08-01T12", "ns")], "y": [52.5], "x": [13.25, 13.5]},
        attrs={"crs": "EPSG:4326"},
    )
    grid = AnalysisGrid.for_aoi(AOI(13.2, 52.42, 13.55, 52.58), resolution_m=1000)

    regridded = to_grid(era5, grid)

    values = regridded["air_temperature_2m"].isel(time=0).values
    assert np.isfinite(values).all()
    assert set(np.unique(values)) <= {290.0, 292.0} or (values.min() >= 290.0 and values.max() <= 292.0)
    assert regridded.sizes["y"] == grid.height


def test_station_collocation_matches_nearest_time_and_reports_unusable_stations():
    from citycube import collocate_stations

    cube = xr.Dataset(
        {"no2": (("time", "y", "x"), np.array([[[20.0, 30.0]], [[25.0, 35.0]]]))},
        coords={"time": np.array(["2026-08-02T10:16:35.048924", "2026-08-03T09:50:23.896256"], dtype="datetime64[ns]"), "y": [52.5], "x": [13.3, 13.4]},
        attrs={"crs": "EPSG:4326"},
    )
    hours = np.arange(np.datetime64("2026-08-02T00", "ns"), np.datetime64("2026-08-04T00", "ns"), np.timedelta64(1, "h"))
    stations = xr.Dataset(
        {"NO2": (("station", "time"), np.vstack([np.linspace(10, 40, hours.size), np.full(hours.size, 50.0)]))},
        coords={"station": ["city", "far"], "time": hours, "latitude": ("station", [52.5, 48.0]), "longitude": ("station", [13.3, 2.3])},
    )

    report = collocate_stations(cube, stations, variable="no2", station_variable="NO2")

    assert report["city"]["samples"] == 2  # matched to 10:00 and 10:00 (nearest hour), not dropped for inexact times
    assert report["far"]["samples"] == 0 and "outside" in report["far"]["note"]
    strict = collocate_stations(cube, stations.isel(time=slice(0, 3)), variable="no2", station_variable="NO2")
    assert strict["city"]["samples"] == 0 and "no observation within" in strict["city"]["note"]


def test_products_without_cloud_cover_are_sampled_across_the_whole_period():
    from citycube import select_product_refs

    radar = [
        ProductRef(product_id=f"p{day:02d}", name=f"S1_{day:02d}", product_type="GRD", start_datetime=f"2026-08-{day:02d}T05:00:00Z",
                   end_datetime=None, timeliness=None, online=True, download_url="", metadata={})
        for day in range(1, 22)
    ]

    chosen = select_product_refs(radar, limit=4)

    assert [p.start_datetime[8:10] for p in chosen] == ["01", "08", "14", "21"]


def test_whole_orbit_products_are_judged_by_their_mid_time_for_day_or_night():
    # Sentinel-5P orbits last ~100 minutes; the start of a day-side orbit over
    # Berlin is still early morning in local time, its middle is not.
    def orbit(name, start, end):
        return ProductRef(product_id=name, name=name, product_type="L2__NO2___", start_datetime=start, end_datetime=end,
                          timeliness=None, online=True, download_url="", metadata={})

    products = [orbit("day", "2026-08-01T10:08:59Z", "2026-08-01T11:50:29Z"), orbit("night", "2026-08-01T01:41:31Z", "2026-08-01T03:23:01Z")]

    assert [p.name for p in filter_overpass(products, longitude=13.4, overpass="day")] == ["day"]


@pytest.mark.parametrize("bounds, message", [
    ((13.5, 52.4, 13.3, 52.6), "longitudes"),   # inverted west/east: used to reach the worker as an HTTP 500
    ((200.0, 52.4, 210.0, 52.6), "longitudes"),
    ((13.3, 52.6, 13.5, 52.4), "latitudes"),
    ((13.3, float("nan"), 13.5, 52.6), "finite"),
    (("13.3", 52.4, 13.5, 52.6), "numbers"),
])
def test_invalid_aois_are_rejected_on_creation(bounds, message):
    with pytest.raises(ValueError, match=message):
        AOI(*bounds)


def test_aoi_accepts_numpy_numbers_and_sensors_must_be_a_list():
    assert AOI(np.float64(13.3), np.int64(52), 13.5, 52.6).south == 52
    with pytest.raises(ValueError, match="list of sensor names"):
        _request(sensors="sentinel3")


_THREADED_NETCDF_SCRIPT = """
import sys
from pathlib import Path
import numpy as np, xarray as xr
from citycube import AOI, AnalysisRequest
from citycube.catalog import ProductRef
from citycube.workflow.adapters import AcquisitionContext, _locked_read

work = Path(sys.argv[1])
rng = np.random.default_rng(0)
source = work / "source.nc"
xr.Dataset({f"v{i}": (("y", "x"), rng.random((300, 300)).astype("float32")) for i in range(6)}).to_netcdf(source)
request = AnalysisRequest(aoi=AOI(13.2, 52.4, 13.6, 52.6), start="2026-08-01", end="2026-08-02", sensors=("sentinel3",), raw_retention="keep")
context = AcquisitionContext(request=request, output_dir=work, downloader_factory=lambda: None, max_workers=4)
products = [ProductRef(product_id=f"p{i}", name=f"product_{i}", product_type="x", start_datetime=None, end_datetime=None,
                       timeliness=None, online=True, download_url="", metadata={}) for i in range(24)]

def acquire(product):
    # build() reads a NetCDF the way the adapters do, subset() writes it, and a
    # second call per product reads the stored subset back: all threaded.
    for _ in range(2):
        context.subset("sentinel3", product, lambda: (_locked_read(xr.open_dataset, source), []))
    return product.name

assert len(context.map_products("sentinel3", products, acquire)) == 24
print("ok")
"""


def test_threaded_netcdf_reads_and_writes_do_not_crash_the_process(tmp_path):
    # HDF5 is not thread safe: max_workers > 1 segfaulted acquisitions before
    # NETCDF_LOCK. Run in a subprocess so a crash fails the test, not pytest.
    import subprocess
    import sys

    completed = subprocess.run([sys.executable, "-c", _THREADED_NETCDF_SCRIPT, str(tmp_path)], capture_output=True, text=True, timeout=300)

    assert completed.returncode == 0, f"exit {completed.returncode}: {completed.stderr[-500:]}"
    assert completed.stdout.strip() == "ok"


def test_masks_stay_boolean_when_a_source_misses_some_times():
    from citycube import TemporalMatch, align_features

    target = xr.Dataset({"lst": (("time", "y", "x"), np.ones((2, 1, 1)))},
                        coords={"time": np.array(["2026-08-01", "2026-08-10"], dtype="datetime64[ns]"), "y": [0.0], "x": [0.0]}, attrs={"crs": CRS})
    feature = xr.Dataset({"valid_mask": (("time", "y", "x"), np.ones((1, 1, 1), dtype=bool))},
                         coords={"time": np.array(["2026-08-01"], dtype="datetime64[ns]"), "y": [0.0], "x": [0.0]}, attrs={"crs": CRS})

    aligned = align_features(target, feature, match=TemporalMatch(np.timedelta64(1, "D")))

    assert aligned["valid_mask"].dtype == bool
    assert aligned["valid_mask"].values.ravel().tolist() == [True, False]


def test_downscaling_streams_scenes_to_a_store_and_reads_static_terrain(tmp_path):
    pytest.importorskip("zarr")
    times = [np.datetime64("2025-06-10T10:00", "ns"), np.datetime64("2025-06-12T10:00", "ns")]
    s2_times = [np.datetime64("2025-06-10T10:30", "ns"), np.datetime64("2025-06-12T10:30", "ns")]
    cube = xr.concat([_coarse_scene(t, seed=i) for i, t in enumerate(times)], dim="time")
    cube["sentinel2_matched_time"] = ("time", np.array(s2_times))
    cube["elevation"] = (("time", "y", "x"), np.broadcast_to(_block_mean(np.add.outer(_FINE_Y, _FINE_X) / 100), (2, 10, 10)).copy())
    terrain = xr.Dataset({"elevation": (("y", "x"), np.add.outer(_FINE_Y, _FINE_X) / 100)}, coords={"y": _FINE_Y, "x": _FINE_X}, attrs={"crs": CRS})

    in_memory = downscale_per_scene(cube, _fine_predictors(s2_times), predictors=["NDVI", "elevation"], model="linear", min_samples=10, terrain=terrain)
    streamed = downscale_per_scene(cube, _fine_predictors(s2_times), predictors=["NDVI", "elevation"], model="linear", min_samples=10, terrain=terrain, store=tmp_path / "d.zarr")

    assert streamed["lst_downscaled"].chunks is not None  # opened lazily from the store
    np.testing.assert_allclose(streamed["lst_downscaled"].values, in_memory["lst_downscaled"].values, equal_nan=True)
    assert list(streamed.time.values) == times
    assert streamed.attrs["downscaling_scenes"] == 2 and streamed.attrs["downscaling_predictors"] == ["NDVI", "elevation"]


def test_aoi_coverage_comes_from_footprints_and_unions_tiles_of_one_pass():
    from citycube.catalog import annotate_aoi_coverage

    aoi = AOI(13.0, 52.0, 14.0, 53.0)

    def product(name, start, west, east):
        geometry = {"type": "Polygon", "coordinates": [[[west, 51.5], [east, 51.5], [east, 53.5], [west, 53.5], [west, 51.5]]]}
        return ProductRef(product_id=name, name=name, product_type="x", start_datetime=start, end_datetime=start,
                          timeliness=None, online=True, download_url="", metadata={"GeoFootprint": geometry})

    annotated = {p.name: p.coverage for p in annotate_aoi_coverage([
        product("sliver", "2026-08-01T10:00:00Z", 13.9, 15.0),       # 10 % of the AOI
        product("tile_a", "2026-08-02T10:00:00Z", 12.0, 13.5),       # one pass split in two tiles
        product("tile_b", "2026-08-02T10:00:00Z", 13.5, 15.0),
        ProductRef(product_id="nogeo", name="nogeo", product_type="x", start_datetime=None, end_datetime=None,
                   timeliness=None, online=True, download_url="", metadata={}),
    ], aoi)}

    assert annotated["sliver"] == pytest.approx(0.1, abs=0.01)
    assert annotated["tile_a"] == annotated["tile_b"] == pytest.approx(1.0)
    assert annotated["nogeo"] is None


def test_discovery_drops_products_that_barely_touch_the_aoi(monkeypatch):
    from citycube.workflow import runner

    def product(name, west, east, cloud):
        geometry = {"type": "Polygon", "coordinates": [[[west, 52.0], [east, 52.0], [east, 53.0], [west, 53.0], [west, 52.0]]]}
        return ProductRef(product_id=name, name=name, product_type="x", start_datetime=f"2026-08-0{len(name) % 9 + 1}T09:30:00Z",
                          end_datetime=None, timeliness="NT", online=True, download_url="", metadata={"GeoFootprint": geometry, "attributes": {"cloudCover": cloud}})

    class FakeAdapter:
        def search(self, request, *, limit):
            # The clearest granule only grazes the AOI; it used to rank first.
            return [product("sliver", 13.55, 14.5, 0.0), product("full", 13.0, 14.0, 20.0)]

    monkeypatch.setitem(runner.SENSOR_ADAPTERS, "sentinel3", FakeAdapter())  # type: ignore[arg-type]

    selected = AnalysisWorkflow(_request(aoi=AOI(13.2, 52.4, 13.6, 52.6))).discover()["sentinel3"]

    assert [p.name for p in selected] == ["full"]
    assert AnalysisRequest.from_dict(_request(min_aoi_coverage=0.7).to_dict()).min_aoi_coverage == 0.7


# --- Sentinel-3 cloud probe -------------------------------------------------


def _probe_files(root: Path, *, cloudy_rows: int) -> Path:
    """geodetic_in.nc and flags_in.nc for a 10 x 10 swath over Berlin, the first rows clouded."""
    root.mkdir(parents=True, exist_ok=True)
    lat = np.repeat(np.linspace(52.58, 52.42, 10)[:, None], 10, axis=1)
    lon = np.repeat(np.linspace(13.22, 13.58, 10)[None, :], 10, axis=0)
    xr.Dataset({"latitude_in": (("rows", "columns"), lat), "longitude_in": (("rows", "columns"), lon)}).to_netcdf(root / "geodetic_in.nc")
    bayes = np.zeros((10, 10), dtype=np.uint8)
    bayes[:cloudy_rows] = 2  # bit 1 = single_moderate
    flags = xr.Dataset({"bayes_in": (("rows", "columns"), bayes, {"flag_masks": [1, 2], "flag_meanings": "single_low single_moderate"})})
    flags.to_netcdf(root / "flags_in.nc")
    return root


def test_aoi_clear_fraction_uses_the_bayesian_mask_inside_the_aoi_only(tmp_path):
    from citycube.sensors.sentinel3.reader import aoi_clear_fraction

    root = _probe_files(tmp_path / "p", cloudy_rows=3)

    assert aoi_clear_fraction(root, AOI(13.2, 52.4, 13.6, 52.6)) == pytest.approx(0.7)
    assert aoi_clear_fraction(root, AOI(13.2, 52.55, 13.6, 52.6)) == pytest.approx(0.0)  # only clouded rows inside
    assert aoi_clear_fraction(root, AOI(0.0, 0.0, 1.0, 1.0)) is None


def test_cloud_probe_skips_clouded_products_and_fills_the_quota_from_spares(tmp_path):
    from citycube.sensors.sentinel3.reader import SLSTR_PROBE_FILES
    from citycube.workflow.adapters import AcquisitionContext, Sentinel3Adapter

    clouded_rows = {"a": 9, "b": 0, "c": 8, "d": 1, "e": 0}

    class FakeDownloader:
        def download_files(self, reference, names, output):
            assert tuple(names) == SLSTR_PROBE_FILES  # only the probe files, never the LST
            return _probe_files(Path(output) / reference.name, cloudy_rows=clouded_rows[reference.name])

    request = _request(max_products_per_sensor=2, min_clear_fraction=0.5)
    context = AcquisitionContext(request=request, output_dir=tmp_path, downloader_factory=FakeDownloader)
    references = [_product(name, f"2026-08-0{i + 1}T09:30:00Z") for i, name in enumerate("abcde")]

    kept = Sentinel3Adapter()._select_clear(references, context, tmp_path / "downloads")

    assert [r.name for r in kept] == ["b", "d"]  # quota of 2 reached from the spares
    assert [item["product"] for item in context.probed_out] == ["a", "c"]
    assert context.probed_out[0]["aoi_clear_fraction"] == pytest.approx(0.1)
    assert not (tmp_path / "downloads" / "a").exists()  # rejected probe files are not kept
    assert not any(r.name == "e" for r in kept)  # stops probing once the quota is full


def test_cloud_probe_judges_cached_subsets_by_their_own_clouds(tmp_path):
    # A cached subset used to be accepted unprobed, so a rerun kept fully clouded scenes.
    from citycube.workflow.adapters import AcquisitionContext, Sentinel3Adapter

    class NoDownloads:
        def download_files(self, reference, names, output):
            raise AssertionError("cached products must not be downloaded again")

    request = _request(max_products_per_sensor=2, min_clear_fraction=0.5)
    context = AcquisitionContext(request=request, output_dir=tmp_path, downloader_factory=NoDownloads)
    references = [_product(name, f"2026-08-0{i + 1}T09:30:00Z") for i, name in enumerate("abc")]
    lat = np.repeat(np.linspace(52.58, 52.42, 10)[:, None], 10, axis=1)
    lon = np.repeat(np.linspace(13.22, 13.58, 10)[None, :], 10, axis=0)
    for reference, cloudy_rows in zip(references, (10, 0, 2)):
        cloud = np.zeros((1, 10, 10), dtype=bool)
        cloud[:, :cloudy_rows] = True
        path = context.subset_path("sentinel3", reference)
        path.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset({
            "latitude": (("time", "y", "x"), lat[None]), "longitude": (("time", "y", "x"), lon[None]),
            "cloud_mask": (("time", "y", "x"), cloud),
        }).to_netcdf(path)

    kept = Sentinel3Adapter()._select_clear(references, context, tmp_path / "downloads")

    assert [r.name for r in kept] == ["b", "c"]
    assert context.probed_out == [{"product": "a", "aoi_clear_fraction": 0.0}]
    assert (tmp_path / "subsets").exists() and context.subset_path("sentinel3", references[0]).exists()  # cache kept


def test_probe_counts_only_pixels_within_the_view_angle_cut(tmp_path):
    # A clear scene seen at 58 degrees is all rejected later by the quality screening.
    from citycube.sensors.sentinel3.reader import aoi_clear_fraction, subset_clear_fraction

    root = _probe_files(tmp_path / "p", cloudy_rows=0)
    with xr.open_dataset(root / "geodetic_in.nc") as geodetic:
        geodetic = geodetic.load()
    geodetic.attrs.update({"track_offset": 0, "resolution": "[ 1000 1000 ]"})
    geodetic.to_netcdf(root / "geodetic_in.nc", mode="w")
    zenith = np.repeat(np.linspace(30, 60, 2)[None, :], 10, axis=0)  # tie points at columns 0 and 16 km
    xr.Dataset({"sat_zenith_tn": (("rows", "tie_columns"), zenith)},
               attrs={"track_offset": 0, "resolution": "[ 16000 1000 ]"}).to_netcdf(root / "geometry_tn.nc")
    aoi = AOI(13.2, 52.4, 13.6, 52.6)

    # Column i sits at 30 + 30 * i / 16 degrees: only column 9 (46.9) is beyond 45.
    assert aoi_clear_fraction(root, aoi) == pytest.approx(0.9)
    assert aoi_clear_fraction(root, aoi, max_view_zenith=None) == pytest.approx(1.0)

    lat = np.repeat(np.linspace(52.58, 52.42, 10)[:, None], 10, axis=1)
    lon = np.repeat(np.linspace(13.22, 13.58, 10)[None, :], 10, axis=0)
    xr.Dataset({
        "latitude": (("time", "y", "x"), lat[None]), "longitude": (("time", "y", "x"), lon[None]),
        "cloud_mask": (("time", "y", "x"), np.zeros((1, 10, 10), dtype=bool)),
        "sat_zenith": (("time", "y", "x"), np.full((1, 10, 10), 58.0)),
    }).to_netcdf(tmp_path / "subset.nc")
    assert subset_clear_fraction(tmp_path / "subset.nc", aoi) == pytest.approx(0.0)


# --- local moving-window downscaling ----------------------------------------


def _two_regime_scene(time):
    """West half: LST falls with NDVI; east half: it rises. One global relation cannot fit both."""
    ndvi = _ndvi(0)
    xx = np.broadcast_to(_FINE_X[None, :], ndvi.shape)
    lst_fine = np.where(xx < 5000, 40 - 20 * ndvi, 20 + 20 * ndvi)
    coarse = xr.Dataset(
        {"lst": (("time", "y", "x"), _block_mean(lst_fine)[None], {"units": "degC"}), "NDVI": (("time", "y", "x"), _block_mean(ndvi)[None])},
        coords={"time": [time], "y": _COARSE_Y, "x": _COARSE_X}, attrs={"crs": CRS},
    )
    fine = xr.Dataset({"NDVI": (("time", "y", "x"), ndvi[None])}, coords={"time": [time], "y": _FINE_Y, "x": _FINE_X}, attrs={"crs": CRS})
    return coarse, fine, lst_fine


def test_local_windows_recover_relations_a_global_model_cannot():
    pytest.importorskip("sklearn")
    from citycube import fit_local_window_downscaler

    time = np.datetime64("2025-06-10T10:00", "ns")
    coarse, fine, truth = _two_regime_scene(time)

    local = fit_local_window_downscaler(coarse, predictors=["NDVI"], window=5, min_samples=20, min_window_samples=10, fine=fine)
    global_only = fit_local_window_downscaler(coarse, predictors=["NDVI"], window=100, min_samples=20, min_window_samples=10_000)

    def rmse(model):
        return float(np.sqrt(np.nanmean((model.predict(fine)["lst_downscaled"].isel(time=0).values - truth) ** 2)))

    assert local.predict(fine).attrs["downscaling_local_windows_fitted"] == 4
    assert global_only.predict(fine).attrs["downscaling_local_windows_fitted"] == 0
    assert rmse(local) < 0.6 * rmse(global_only)


def test_local_trees_run_per_scene_and_conserve_the_coarse_observation():
    pytest.importorskip("sklearn")
    time = np.datetime64("2025-06-10T10:00", "ns")
    coarse, fine, _ = _two_regime_scene(time)
    coarse["sentinel2_matched_time"] = ("time", np.array([time]))

    result = downscale_per_scene(coarse, fine, predictors=["NDVI"], model="local_trees", min_samples=20,
                                 model_options={"window": 5, "min_window_samples": 10})

    assert result.attrs["downscaling_model"] == "local_trees"
    assert float(result["coarse_consistency_rmse"].values[0]) < 0.1
    assert result["lst_downscaled"].attrs["units"] == "degC"


def test_linear_leaf_ensemble_does_not_extrapolate_without_limit():
    pytest.importorskip("sklearn")
    from citycube.downscale.local import LinearLeafTreeEnsemble

    X = np.linspace(0, 1, 200)[:, None]
    y = 10 + 5 * X[:, 0]
    model = LinearLeafTreeEnsemble(max_leaf_nodes=4, n_estimators=5).fit(X, y)

    far = model.predict(np.array([[100.0]]))[0]
    assert far <= 15 + 0.25 * 5 + 1e-6  # clipped to the leaf range plus 25 %


def test_reaggregation_accepts_spatially_chunked_stores():
    # Workflow results are opened lazily from Zarr, often tiled in y and x.
    from citycube import reaggregate_to_target

    x = np.arange(40) * 100.0 + 50
    coarse_x = np.arange(4) * 1000.0 + 500
    fine = xr.DataArray(np.random.default_rng(0).random((2, 40, 40)), dims=("time", "y", "x"),
                        coords={"time": [0, 1], "y": x[::-1], "x": x}, attrs={"crs": "EPSG:32633"})
    target = xr.DataArray(np.zeros((2, 4, 4)), dims=("time", "y", "x"),
                          coords={"time": [0, 1], "y": coarse_x[::-1], "x": coarse_x}, attrs={"crs": "EPSG:32633"})
    lazy = reaggregate_to_target(fine.chunk({"time": 1, "y": 16, "x": 16}), target)
    np.testing.assert_allclose(lazy.values, reaggregate_to_target(fine, target).values)


def test_sentinel2_cog_reads_are_cached_per_grid_and_mosaicked_per_date(tmp_path, monkeypatch):
    from citycube.workflow import adapters

    grid = AnalysisGrid.for_aoi(AOI(13.2, 52.4, 13.3, 52.45), resolution_m=500)
    reads: list[str] = []

    def fake_read(reference, target_grid):
        reads.append(reference.name)
        height, width = target_grid.height, target_grid.width
        half = np.full((1, height, width), np.nan, dtype="float32")
        rows = slice(0, height // 2) if reference.name.endswith("a") else slice(height // 2, height)
        half[0, rows] = 0.1
        time = np.array([np.datetime64(reference.start_datetime.rstrip("Z"), "ns")])
        coords = {"time": time, "y": np.arange(height, dtype=float), "x": np.arange(width, dtype=float)}
        bands = {name: (("time", "y", "x"), half.copy()) for name in ("B02", "B03", "B04", "B08", "B11", "B12")}
        return xr.Dataset({**bands, "valid_mask": (("time", "y", "x"), np.isfinite(half))}, coords=coords, attrs={"crs": target_grid.crs})

    monkeypatch.setattr(adapters, "read_s2_l2a_cog", fake_read)
    request = _request(sensors=("sentinel2",), sentinel2_source="stac_cog", predictor_grid=grid)
    # Two tiles of one date and one tile of another.
    references = [_product("T1_a", "2025-06-10T10:00:00Z"), _product("T2_b", "2025-06-10T10:00:00Z"), _product("T1_c", "2025-06-12T10:00:00Z")]

    first = adapters.Sentinel2Adapter().acquire(references, adapters.AcquisitionContext(request=request, output_dir=tmp_path, downloader_factory=lambda: None))
    assert first.sizes["time"] == 2
    assert np.isfinite(first["NDVI"].isel(time=0).values).all()  # both halves of the first date
    assert len(reads) == 3

    again = adapters.Sentinel2Adapter().acquire(references, adapters.AcquisitionContext(request=request, output_dir=tmp_path, downloader_factory=lambda: None))
    assert len(reads) == 3  # served from the stored subsets
    xr.testing.assert_allclose(first["NDVI"], again["NDVI"])

    finer = dataclasses.replace(request, predictor_grid=AnalysisGrid.for_aoi(AOI(13.2, 52.4, 13.3, 52.45), resolution_m=250))
    adapters.Sentinel2Adapter().acquire(references, adapters.AcquisitionContext(request=finer, output_dir=tmp_path, downloader_factory=lambda: None))
    assert len(reads) == 6  # another grid is another read


def test_cloud_probe_results_are_reused_without_downloading_again(tmp_path):
    from citycube.sensors.sentinel3.reader import SLSTR_PROBE_FILES
    from citycube.workflow.adapters import AcquisitionContext, Sentinel3Adapter

    clouded_rows = {"a": 9, "b": 0, "c": 1}
    downloads: list[str] = []

    class CountingDownloader:
        def download_files(self, reference, names, output):
            assert tuple(names) == SLSTR_PROBE_FILES
            downloads.append(reference.name)
            return _probe_files(Path(output) / reference.name, cloudy_rows=clouded_rows[reference.name])

    request = _request(max_products_per_sensor=2, min_clear_fraction=0.5)
    references = [_product(name, f"2026-08-0{i + 1}T09:30:00Z") for i, name in enumerate("abc")]
    first = Sentinel3Adapter()._select_clear(references, AcquisitionContext(request=request, output_dir=tmp_path, downloader_factory=CountingDownloader), tmp_path / "downloads")
    assert [r.name for r in first] == ["b", "c"] and downloads == ["a", "b", "c"]

    again = AcquisitionContext(request=request, output_dir=tmp_path, downloader_factory=CountingDownloader)
    second = Sentinel3Adapter()._select_clear(references, again, tmp_path / "downloads")
    assert [r.name for r in second] == ["b", "c"]
    assert downloads == ["a", "b", "c"]  # nothing probed twice
    assert again.probed_out == [{"product": "a", "aoi_clear_fraction": pytest.approx(0.1)}]

    polygon = dataclasses.replace(request, aoi=AOI.from_geojson({"type": "Polygon", "coordinates": [[[13.2, 52.4], [13.6, 52.4], [13.4, 52.6], [13.2, 52.4]]]}))
    Sentinel3Adapter()._select_clear(references, AcquisitionContext(request=polygon, output_dir=tmp_path, downloader_factory=CountingDownloader), tmp_path / "downloads")
    assert len(downloads) > 3  # another shape is measured again


def test_remote_cog_reads_cannot_hang_forever():
    # A stalled Planetary Computer read once blocked a run for half an hour.
    from citycube.sensors.cog import GDAL_HTTP_OPTIONS

    assert int(GDAL_HTTP_OPTIONS["GDAL_HTTP_TIMEOUT"]) <= 300
    assert int(GDAL_HTTP_OPTIONS["GDAL_HTTP_CONNECTTIMEOUT"]) <= 60
    assert int(GDAL_HTTP_OPTIONS["GDAL_HTTP_LOW_SPEED_TIME"]) > 0
