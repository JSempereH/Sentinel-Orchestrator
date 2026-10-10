from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import xarray as xr

from citycube import AOI, AnalysisGrid, AnalysisRequest, AnalysisWorkflow, AuxiliarySpec, OpenAQInterpolationConfig
from citycube.providers.cams import CAMSProvider
from citycube.providers.era5 import ERA5Provider
from citycube.providers.openaq import OpenAQProvider


class _Response:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _OpenAQSession:
    def __init__(self):
        self.urls: list[str] = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        if url.endswith("/locations"):
            return _Response({"meta": {"found": 1}, "results": [{"id": 7, "coordinates": {"latitude": 52.5, "longitude": 13.3}, "sensors": [{"id": 11, "parameter": {"name": "no2", "units": "ug/m3"}}]}]})
        return _Response({"meta": {"found": 1}, "results": [{"value": 21.0, "parameter": {"name": "no2", "units": "ug/m3"}, "period": {"datetimeFrom": {"utc": "2025-06-01T12:00:00Z"}}}]})


def test_auxiliary_request_round_trip_and_plan():
    request = AnalysisRequest(
        aoi=AOI(13.2, 52.4, 13.5, 52.6),
        start="2025-06-01",
        end="2025-06-02",
        sensors=("sentinel3",),
        auxiliary=(
            AuxiliarySpec("era5", variables=("air_temperature_2m",)),
            AuxiliarySpec("cams", variables=("NO2",)),
            AuxiliarySpec("openaq", variables=("NO2",), options={"max_locations": 5}),
        ),
    )
    restored = AnalysisRequest.from_dict(request.to_dict())
    assert [item.provider for item in restored.auxiliary] == ["era5", "cams", "openaq"]
    assert [step.sensor for step in AnalysisWorkflow(restored).plan.steps] == [
        "sentinel3", "auxiliary:era5", "auxiliary:cams", "auxiliary:openaq"
    ]


def test_cds_requests_are_explicit_and_aoi_order_is_provider_safe():
    aoi = AOI(13.2, 52.4, 13.5, 52.6)
    era5 = ERA5Provider()
    request = era5.build_request(aoi, "2025-06-01", "2025-06-02", variables=("air_temperature_2m",))
    assert request["variable"] == ["2m_temperature"]
    assert request["area"] == [52.6, 13.2, 52.4, 13.5]
    assert request["time"][0] == "00:00"
    cams = CAMSProvider()
    cams_request = cams.build_request(aoi, "2025-06-01", "2025-06-01", variables=("NO2", "PM2.5"))
    assert cams_request["variable"] == ["total_column_nitrogen_dioxide", "particulate_matter_2.5um"]
    assert cams_request["date"] == ["2025-06-01"]
    assert cams_request["area"][0] - cams_request["area"][2] >= 0.75
    assert cams_request["area"][3] - cams_request["area"][1] >= 0.75


def test_gridded_provider_open_normalizes_coordinates_and_metadata(tmp_path: Path):
    source = xr.Dataset(
        {"t2m": (("time", "latitude", "longitude"), np.ones((1, 2, 2)) * 300.0)},
        coords={"time": ["2025-06-01"], "latitude": [52.5, 52.4], "longitude": [13.2, 13.3]},
    )
    path = tmp_path / "era5.nc"
    source.to_netcdf(path)
    result = ERA5Provider().open(path)
    assert "air_temperature_2m" in result
    assert result.attrs["crs"] == "EPSG:4326"
    assert result["air_temperature_2m"].attrs["units"] == "K"
    assert result["air_temperature_2m"].attrs["sensor"] == "era5"


def test_openaq_open_creates_station_table_and_preserves_units(tmp_path: Path):
    payload = {
        "locations": [{"id": 7, "name": "Berlin Mitte", "coordinates": {"latitude": 52.52, "longitude": 13.40}}],
        "measurements": [
            {"location_id": 7, "parameter": {"name": "no2"}, "value": 21.0, "unit": "ug/m3", "datetime": {"utc": "2025-06-01T12:00:00Z"}},
            {"location_id": 7, "parameter": {"name": "no2"}, "value": 23.0, "unit": "ug/m3", "datetime": {"utc": "2025-06-01T13:00:00Z"}},
        ],
    }
    path = tmp_path / "openaq.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = OpenAQProvider().open(path)
    assert result.attrs["analysis_shape"] == "station_table"
    assert result["NO2"].shape == (1, 2)
    assert result["NO2"].attrs["units"] == "ug/m3"
    assert result.latitude.item() == 52.52


def test_openaq_download_follows_sensor_measurement_endpoint(tmp_path: Path):
    session = _OpenAQSession()
    provider = OpenAQProvider(session=cast(Any, session))
    artifact = provider.download(
        AOI(13.2, 52.4, 13.5, 52.6),
        "2025-06-01",
        "2025-06-01",
        AuxiliarySpec("openaq", variables=("NO2",)),
        tmp_path,
    )
    assert any(url.endswith("/sensors/11/measurements") for url in session.urls)
    assert provider.open(artifact.path)["NO2"].item() == 21.0
    assert (tmp_path / "manifest.json").exists()


def test_openaq_leave_one_station_out_reports_coverage_and_error():
    stations = xr.Dataset(
        {"NO2": (("station", "time"), np.full((4, 2), 10.0))},
        coords={
            "station": ["1", "2", "3", "4"],
            "time": np.array(["2025-06-01T12:00", "2025-06-01T13:00"], dtype="datetime64[ns]"),
            "latitude": ("station", [52.49, 52.50, 52.51, 52.52]),
            "longitude": ("station", [13.35, 13.36, 13.37, 13.38]),
        },
        attrs={"analysis_shape": "station_table"},
    )
    metrics = OpenAQProvider.validate_leave_one_station_out(
        stations,
        "NO2",
        config=OpenAQInterpolationConfig(min_neighbors=2, max_distance_m=10000),
    )
    assert metrics["samples"] == 8
    assert metrics["coverage_fraction"] == 1.0
    assert metrics["rmse"] == pytest.approx(0.0)


def test_workflow_merges_gridded_auxiliary_and_keeps_stations_separate():
    request = AnalysisRequest(
        aoi=AOI(13.2, 52.4, 13.5, 52.6),
        start="2025-06-01",
        end="2025-06-01",
        sensors=("sentinel3",),
        auxiliary=(AuxiliarySpec("era5", variables=("air_temperature_2m",)), AuxiliarySpec("openaq", variables=("NO2",))),
    )
    target = xr.Dataset(
        {"lst": (("time", "y", "x"), np.ones((1, 2, 2)) * 300.0)},
        coords={"time": ["2025-06-01"], "y": [52.5, 52.4], "x": [13.3, 13.4]},
        attrs={"crs": "EPSG:4326", "grid_id": "target"},
    )
    auxiliary = xr.Dataset(
        {"air_temperature_2m": (("time", "y", "x"), np.ones((1, 2, 2)) * 290.0, {"units": "K", "sensor": "era5", "product": "reanalysis-era5-land", "aggregation_method": "native"})},
        coords={"time": ["2025-06-01"], "y": [52.5, 52.4], "x": [13.3, 13.4]},
        attrs={"crs": "EPSG:4326", "grid_id": "target", "metadata_contract": "citycube-v1", "analysis_shape": "regular_grid"},
    )
    stations = xr.Dataset(
        {"NO2": (("station", "time"), [[21.0]], {"units": "ug/m3", "sensor": "openaq", "product": "openaq-v3", "aggregation_method": "native"})},
        coords={"station": ["7"], "time": ["2025-06-01"], "latitude": ("station", [52.5]), "longitude": ("station", [13.3])},
        attrs={"analysis_shape": "station_table", "crs": "EPSG:4326"},
    )
    result = AnalysisWorkflow(request).run({"sentinel3": target, "era5": auxiliary, "openaq": stations})
    assert "auxiliary_era5_air_temperature_2m" in result.cube
    assert result.auxiliary is not None and "openaq" in result.auxiliary
    assert "NO2" in result.auxiliary["openaq"]


def test_workflow_interpolates_openaq_with_support_diagnostics():
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    longitudes = np.array([13.35, 13.37, 13.39, 13.36])
    latitudes = np.array([52.49, 52.49, 52.51, 52.52])
    x, y = transformer.transform(longitudes, latitudes)
    grid = AnalysisGrid.from_bounds((float(min(x)) - 1000, float(min(y)) - 1000, float(max(x)) + 1000, float(max(y)) + 1000), crs="EPSG:32633", resolution_m=1000)
    times = np.array(["2025-06-01T12:00:00"], dtype="datetime64[ns]")
    stations = xr.Dataset(
        {"NO2": (("station", "time"), np.array([[20.0], [25.0], [30.0], [22.0]], dtype=float), {"units": "ug/m3"})},
        coords={"station": ["1", "2", "3", "4"], "time": times, "latitude": ("station", latitudes), "longitude": ("station", longitudes)},
        attrs={"analysis_shape": "station_table", "crs": "EPSG:4326", "source": "openaq", "product": "openaq-v3"},
    )
    target = xr.Dataset(
        {"lst": (("time", "y", "x"), np.full((1, grid.height, grid.width), 300.0))},
        coords={"time": times, "y": grid.y, "x": grid.x},
        attrs={"crs": grid.crs, "grid_id": grid.grid_id},
    )
    request = AnalysisRequest(
        aoi=AOI(13.3, 52.45, 13.45, 52.55),
        start="2025-06-01",
        end="2025-06-01",
        sensors=("sentinel3",),
        grid=grid,
        predictor_grid=grid,
        auxiliary=(AuxiliarySpec("openaq", variables=("NO2",), options={"interpolation_max_distance_m": 10000}),),
    )

    result = AnalysisWorkflow(request).run({"sentinel3": target, "openaq": stations})

    assert "auxiliary_openaq_NO2" in result.cube
    assert "auxiliary_openaq_NO2_station_count" in result.cube
    assert result.cube["auxiliary_openaq_NO2_interpolation_valid"].any().item()
    assert result.auxiliary is not None and result.auxiliary["openaq"].attrs["analysis_shape"] == "station_table"
    assert result.provenance["auxiliary_rasters"] == ["openaq"]
    interpolated = OpenAQProvider.interpolate_to_grid(
        stations,
        grid,
        times,
        config=OpenAQInterpolationConfig(min_neighbors=3, max_distance_m=10000),
    )
    assert interpolated["NO2"].notnull().any().item()
