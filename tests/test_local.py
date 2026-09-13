from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from sentinel_analysis import (
    AOI,
    AnalysisGrid,
    GridSpec,
    EcostressCatalog,
    LandsatCatalog,
    LinearDownscaler,
    LSTExceptionFlag,
    ProductQuery,
    ProductRef,
    QualityPolicy,
    read_ecostress_lst,
    read_landsat_lst,
    Sentinel3LST,
    Sentinel1ReadConfig,
    S1ARDProcessingConfig,
    Sentinel5PReadConfig,
    STACCatalog,
    align_features,
    apply_quality_mask,
    cf_flag_mask,
    daily_mean,
    get_city,
    heat_hazard_metrics,
    fit_linear_downscaler,
    fit_coarse_consistent_linear_downscaler,
    read_l2_lst,
    scl_valid_mask,
    sentinel2_indices,
    sentinel1_indices,
    build_s1_graph,
    process_s1_grd,
    process_s1,
    process_s1_rtc,
    read_s1_grd,
    read_s1_ard,
    read_s1_rtc,
    Sentinel1Catalog,
    Sentinel1RTCConfig,
    Sentinel2Catalog,
    SENTINEL2_L1C_PRODUCT_TYPE,
    read_s5p_l2,
    grid_s5p,
    grid_l2_lst,
    compose_s2,
    sentinel2_coverage,
    reaggregate_to_target,
    split_spatiotemporal,
    validate_reaggregation,
    fit_random_forest_downscaler,
    fit_tsharp_downscaler,
    fit_gwr_downscaler,
    fit_coarse_consistent_gwr_downscaler,
    fuse_starfm,
    fuse_estarfm,
    blocked_spatiotemporal_split,
    validate_independent_reference,
    select_product_refs,
    ColumnToSurfaceConfig,
    column_to_surface_estimate,
    collocate_stations,
    compare_to_reference,
    write_cog,
    open_zarr,
    write_zarr,
    validate_cog,
    read_cog,
    AnalysisRequest,
    AnalysisWorkflow,
    statistics,
    tag_cube,
    to_celsius,
    validate_cube,
)
from sentinel_analysis import CubeValidationError
from sentinel_analysis.fusion import TemporalMatch
from sentinel_analysis.sensors.ecostress import _FALLBACK_LST_OFFSET, _FALLBACK_LST_SCALE
from sentinel_analysis.sensors.landsat import _stac_item_to_product_ref
from sentinel_analysis.stac import STACItem
from sentinel_analysis.workflow.runner import _combine_sentinel3, _mosaic_temporal_tiles


def dataset() -> xr.Dataset:
    return xr.Dataset(
        {
            "lst": (("time", "y", "x"), np.array([[[300.0, 301.0], [302.0, np.nan]], [[304.0, 305.0], [306.0, 307.0]]])),
            "lst_uncertainty": (("time", "y", "x"), np.ones((2, 2, 2))),
            "exception_flags": (
                ("time", "y", "x"),
                np.array([[[0, int(LSTExceptionFlag.SATURATION)], [0, 0]], [[0, 0], [0, 0]]]),
            ),
            "cloud_mask": (("time", "y", "x"), np.array([[[False, False], [True, False]], [[False, False], [False, False]]])),
        },
        coords={"time": np.array(["2025-06-10", "2025-06-11"], dtype="datetime64[ns]")},
    )


def test_aoi_and_query_validation():
    aoi = AOI(13.2, 52.4, 13.5, 52.6)
    assert "POLYGON" in aoi.as_wkt()
    start, end = ProductQuery(aoi, "2025-06-01", "2025-06-02").normalized_dates()
    assert start.startswith("2025-06-01")
    assert end.startswith("2025-06-02T23:59:59.999999")


def test_quality_mask_does_not_use_temperature_cutoff():
    masked = apply_quality_mask(dataset())
    assert masked["valid_mask"].sum().item() == 5
    assert np.isnan(masked["lst"].isel(time=0, y=0, x=1).item())
    assert masked["lst"].isel(time=1, y=1, x=1).item() == 307


def test_conversion_daily_aggregation_and_statistics():
    converted = to_celsius(apply_quality_mask(dataset()))
    assert converted["lst"].attrs["units"] == "degC"
    daily = daily_mean(converted, assume_quality_masked=True)
    assert daily.sizes["time"] == 2
    stats = statistics(converted)
    assert stats["pixels_valid"] == 5
    assert np.isclose(stats["min"], 26.85)


def test_reader_discovers_variables_without_fixed_filename(tmp_path: Path):
    product = tmp_path / "S3A_SL_2_LST____20250610T090000_20250610T090300_0000_000_000____LN2_D_NT_005.SEN3"
    product.mkdir()
    xr.Dataset(
        {
            "LST": (("rows", "columns"), np.array([[300, 301]], dtype=np.int16)),
            "LST_uncertainty": (("rows", "columns"), np.array([[1, 2]], dtype=np.int16)),
            "exception_flags": (("rows", "columns"), np.zeros((1, 2), dtype=np.int16)),
        },
        attrs={"units": "K"},
    ).to_netcdf(product / "measurement_l2.nc")
    result = read_l2_lst(product)
    assert result.attrs["product_type"] == "SL_2_LST"
    assert result["lst"].dims == ("time", "y", "x")
    assert result["lst"].attrs["units"] == "K"


def test_facade_combine_requires_products():
    try:
        Sentinel3LST.combine([])
    except ValueError as exc:
        assert "At least one" in str(exc)
    else:
        raise AssertionError("Expected empty combine to fail")


def test_city_grid_quality_and_heat_contracts():
    city = get_city("Guadalajara")
    grid = GridSpec.for_city(city, resolution_m=1000)
    cube = xr.Dataset(
        {"lst": (("time", "y", "x"), np.array([[[30.0]], [[36.0]]]))},
        coords={"time": np.array(["2025-06-12", "2025-06-13"], dtype="datetime64[ns]"), "y": [0], "x": [0]},
        attrs={"crs": grid.crs},
    )
    tagged = tag_cube(cube, source="test", grid=grid)
    validate_cube(tagged, required_variables=("lst",))
    metrics = heat_hazard_metrics(tagged, threshold_celsius=35)
    assert metrics["hot_observation_count"].item() == 1
    assert metrics.attrs["missing_values_filled"] is False


def test_read_radiometric_offsets_resolves_band_id_and_defaults_empty(tmp_path: Path):
    # Regression test: real CDSE products post processing-baseline 04.00
    # (2022-01-25 onward) carry a per-band RADIO_ADD_OFFSET/BOA_ADD_OFFSET
    # (typically -1000) that must be added to the raw DN *before* dividing
    # by QUANTIFICATION_VALUE - confirmed missing (and then fixed) against
    # a real downloaded L1C product in
    # scripts/validate_sentinel2_l1c_real_reference.py.
    from sentinel_analysis.sensors.sentinel2 import _read_radiometric_offsets

    manifest = tmp_path / "MTD_MSIL1C.xml"
    manifest.write_text(
        '<?xml version="1.0"?>\n'
        '<n1:Level-1C_User_Product xmlns:n1="https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-1C.xsd">\n'
        "  <n1:General_Info>\n"
        "    <Product_Image_Characteristics>\n"
        "      <Radiometric_Offset_List>\n"
        '        <RADIO_ADD_OFFSET band_id="0">-1000</RADIO_ADD_OFFSET>\n'
        '        <RADIO_ADD_OFFSET band_id="8">-1000</RADIO_ADD_OFFSET>\n'
        "      </Radiometric_Offset_List>\n"
        "    </Product_Image_Characteristics>\n"
        "  </n1:General_Info>\n"
        "</n1:Level-1C_User_Product>\n",
        encoding="utf-8",
    )

    offsets = _read_radiometric_offsets(manifest, "RADIO_ADD_OFFSET")

    assert offsets == {"B01": -1000, "B8A": -1000}
    assert _read_radiometric_offsets(tmp_path / "missing.xml", "RADIO_ADD_OFFSET") == {}
    assert _read_radiometric_offsets(manifest, "BOA_ADD_OFFSET") == {}


def test_cf_quality_and_sentinel2_indices():
    flags = xr.DataArray(
        np.array([[128, 0]], dtype=np.uint16),
        dims=("y", "x"),
        attrs={"flag_masks": np.array([2, 128]), "flag_meanings": "threshold gross_cloud"},
    )
    assert cf_flag_mask(flags) == 128
    quality = apply_quality_mask(
        xr.Dataset(
            {"lst": (("y", "x"), np.array([[300.0, 301.0]])), "cloud_flags": flags}
        ),
        QualityPolicy(cloud_flag_mask=128),
    )
    assert quality["valid_mask"].sum().item() == 1

    optical = xr.Dataset({
        "B02": (("y", "x"), np.full((1, 1), 0.1)),
        "B03": (("y", "x"), np.full((1, 1), 0.2)),
        "B04": (("y", "x"), np.full((1, 1), 0.2)),
        "B08": (("y", "x"), np.full((1, 1), 0.6)),
        "B11": (("y", "x"), np.full((1, 1), 0.3)),
        "B12": (("y", "x"), np.full((1, 1), 0.3)),
    })
    indices = sentinel2_indices(optical)
    assert np.isclose(indices["NDVI"].item(), 0.5)
    assert scl_valid_mask(xr.DataArray([[4, 9]], dims=("y", "x"))).sum().item() == 1


def test_temporal_feature_alignment():
    times = np.array(["2025-06-12", "2025-06-13"], dtype="datetime64[ns]")
    target = xr.Dataset(
        {"lst": (("time", "y", "x"), np.ones((2, 1, 1)))},
        coords={"time": times, "y": [0], "x": [0]},
        attrs={"crs": "EPSG:32613", "grid_id": "g"},
    )
    features = xr.Dataset(
        {"NDVI": (("time", "y", "x"), np.full((2, 1, 1), 0.5))},
        coords={"time": times + np.timedelta64(1, "D"), "y": [0], "x": [0]},
        attrs={"crs": "EPSG:32613", "grid_id": "g"},
    )
    aligned = align_features(target, features, match=TemporalMatch(np.timedelta64(2, "D")))
    assert "NDVI" in aligned
    assert aligned.sizes["time"] == 2


def test_linear_downscaler_reports_uncertainty():
    time = np.arange(120, dtype="int64").astype("datetime64[D]")
    ndvi = np.linspace(0.1, 0.8, 120).reshape(120, 1, 1)
    ndbi = np.linspace(0.8, 0.1, 120).reshape(120, 1, 1)
    lst = 20 + 10 * ndvi - 4 * ndbi
    training = xr.Dataset(
        {"lst": (("time", "y", "x"), lst), "NDVI": (("time", "y", "x"), ndvi), "NDBI": (("time", "y", "x"), ndbi)},
        coords={"time": time, "y": [0], "x": [0]},
        attrs={"crs": "EPSG:32613", "grid_id": "coarse"},
    )
    model = fit_linear_downscaler(training, predictors=("NDVI", "NDBI"), min_samples=100)
    result = model.predict(training.isel(time=slice(0, 1)))
    assert isinstance(model, LinearDownscaler)
    assert result["downscaled_support"].item()
    assert result["lst_downscaled_uncertainty"].item() < 1e-10


def test_coarse_consistent_downscaler_conserves_target_scale():
    time = np.arange(4, dtype="int64").astype("datetime64[D]")
    coarse = xr.Dataset(
        {
            "lst": (("time", "y", "x"), np.full((4, 2, 2), 300.0)),
            "NDVI": (("time", "y", "x"), np.full((4, 2, 2), 0.5)),
        },
        coords={"time": time, "y": [150.0, 50.0], "x": [50.0, 150.0]},
        attrs={"crs": "EPSG:32613", "grid_id": "coarse"},
    )
    fine = xr.Dataset(
        {"NDVI": (("time", "y", "x"), np.full((4, 4, 4), 0.5))},
        coords={"time": time, "y": [175.0, 125.0, 75.0, 25.0], "x": [25.0, 75.0, 125.0, 175.0]},
        attrs={"crs": "EPSG:32613", "grid_id": "fine"},
    )
    model = fit_coarse_consistent_linear_downscaler(coarse, predictors=("NDVI",), min_samples=4)

    result = model.predict(fine, coarse)

    assert result.attrs["coarse_consistency_within_tolerance"]
    assert np.allclose(reaggregate_to_target(result["lst_downscaled"], coarse["lst"]).values, coarse["lst"].values)


def test_coarse_consistency_uses_matching_reference_time():
    coarse = xr.Dataset(
        {"lst": (("time", "y", "x"), np.array([[[20.0, 21.0], [22.0, 23.0]], [[30.0, 31.0], [32.0, 33.0]]] ))},
        coords={"time": np.array(["2025-06-16", "2025-06-17"], dtype="datetime64[ns]"), "y": [150.0, 50.0], "x": [50.0, 150.0]},
        attrs={"crs": "EPSG:32613", "grid_id": "coarse"},
    )
    fine = xr.Dataset(
        {"NDVI": (("time", "y", "x"), np.full((1, 4, 4), 0.5))},
        coords={"time": np.array(["2025-06-16T12:00:00"], dtype="datetime64[ns]"), "y": [175.0, 125.0, 75.0, 25.0], "x": [25.0, 75.0, 125.0, 175.0]},
        attrs={"crs": "EPSG:32613", "grid_id": "fine"},
    )
    training = coarse.assign(NDVI=(coarse["lst"] * 0 + 0.5))
    model = fit_coarse_consistent_linear_downscaler(training, predictors=("NDVI",), min_samples=4)

    result = model.predict(fine, coarse)

    assert result.attrs["coarse_consistency_within_tolerance"]
    assert result.sizes["time"] == 1


def test_sentinel2_same_time_tiles_are_mosaicked():
    time = np.array(["2025-06-19T10:15:59"], dtype="datetime64[ns]")
    first = xr.Dataset(
        {
            "NDVI": (("time", "y", "x"), np.array([[[0.4, np.nan]]])),
            "valid_mask": (("time", "y", "x"), np.array([[[True, False]]])),
        },
        coords={"time": time, "y": [1.0], "x": [1.0, 2.0]},
        attrs={"crs": "EPSG:32633", "grid_id": "test-grid"},
    )
    second = xr.Dataset(
        {
            "NDVI": (("time", "y", "x"), np.array([[[np.nan, 0.7]]])),
            "valid_mask": (("time", "y", "x"), np.array([[[False, True]]])),
        },
        coords={"time": time, "y": [1.0], "x": [1.0, 2.0]},
        attrs={"crs": "EPSG:32633", "grid_id": "test-grid"},
    )

    result = _mosaic_temporal_tiles([first, second])

    assert len(result) == 1
    assert np.allclose(result[0]["NDVI"].values, [[[0.4, 0.7]]], equal_nan=False)
    assert result[0]["valid_mask"].all().item()


def test_predictor_cube_preserves_union_of_sensor_times():
    request = AnalysisRequest.for_city(
        get_city("Berlin"),
        "2025-06-19",
        "2025-06-22",
        sensors=("sentinel2", "sentinel5p"),
        resolution_m=100,
    )
    workflow = AnalysisWorkflow(request)
    assert request.predictor_grid is not None
    grid = request.predictor_grid
    s2 = xr.Dataset(
        {"NDVI": (("time", "y", "x"), np.ones((1, 2, 2)))},
        coords={"time": np.array(["2025-06-19"], dtype="datetime64[ns]"), "y": [2.0, 1.0], "x": [1.0, 2.0]},
        attrs={"crs": grid.crs, "grid_id": grid.grid_id},
    )
    s5p = xr.Dataset(
        {"NO2": (("time", "y", "x"), np.full((1, 2, 2), 0.001))},
        coords={"time": np.array(["2025-06-20"], dtype="datetime64[ns]"), "y": [2.0, 1.0], "x": [1.0, 2.0]},
        attrs={"crs": grid.crs, "grid_id": grid.grid_id},
    )

    result = workflow._merge_predictors({"sentinel2": s2, "sentinel5p": s5p})

    assert result is not None
    assert result.sizes["time"] == 2
    assert "NDVI" in result and "NO2" in result


def test_stac_search_preserves_pagination_request():
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Session:
        def __init__(self):
            self.calls = []

        def post(self, url, *, json, timeout):
            self.calls.append(json)
            if len(self.calls) == 1:
                return Response({
                    "features": [{"id": "one", "collection": "demo", "properties": {}, "assets": {}}],
                    "links": [{"rel": "next", "href": url, "body": {"collections": ["demo"], "token": "next"}}],
                })
            return Response({"features": [{"id": "two", "collections": ["demo"], "properties": {}, "assets": {}}]})

    session = Session()
    items = STACCatalog("https://example.test/v1", session=session).search(
        collections=("demo",), datetime_range="2025-06-01/2025-06-02", limit=2
    )
    assert [item.item_id for item in items] == ["one", "two"]
    assert session.calls[0]["datetime"] == "2025-06-01T00:00:00Z/2025-06-02T23:59:59Z"
    assert session.calls[1]["token"] == "next"


def test_sentinel1_processing_graph_and_backscatter_reader(tmp_path: Path):
    graph = build_s1_graph("input.SAFE", tmp_path / "output.tif")
    assert "Apply-Orbit-File" in graph
    assert "ThermalNoiseRemoval" in graph
    assert "Remove-GRD-Border-Noise" in graph
    assert "Calibration" in graph
    assert "Terrain-Correction" in graph

    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    raster_path = tmp_path / "sigma0.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=2,
        dtype="float32",
        crs="EPSG:32613",
        transform=from_origin(0, 200, 100, 100),
        nodata=-9999,
    ) as destination:
        destination.write(np.ones((2, 2), dtype=np.float32), 1)
        destination.write(np.full((2, 2), 0.5, dtype=np.float32), 2)
        destination.set_band_description(1, "Sigma0_VV")
        destination.set_band_description(2, "Sigma0_VH")
    dataset = read_s1_grd(raster_path, config=Sentinel1ReadConfig(observation_time="2025-06-13"))
    indices = sentinel1_indices(dataset)
    assert dataset.attrs["backscatter_quantity"] == "sigma0"
    assert np.isclose(indices["VH_VV_ratio_dB"].mean().item(), -3.0103, atol=1e-3)
    assert np.isclose(indices["RVI"].mean().item(), 4 / 3)

    dry_run = process_s1_grd("input.SAFE", tmp_path / "processed", execute=False)
    assert dry_run.graph_path is not None
    assert dry_run.graph_path.exists()
    assert dry_run.command[0] == "gpt"

    ard_run = process_s1(
        "S1A_IW_GRDH_1SDV_20250613T010546_20250613T010611.SAFE",
        tmp_path / "ard",
        backend="s1ard",
        ard_config=S1ARDProcessingConfig(measurement="gamma"),
        execute=False,
    )
    assert ard_run.command[:2] == ("s1rb", "process")
    assert ard_run.backend == "s1ard"
    assert ard_run.graph_path is not None
    assert "measurement = gamma" in ard_run.graph_path.read_text()

    ard_root = tmp_path / "ard_product" / "measurement"
    ard_root.mkdir(parents=True)
    from rasterio.transform import from_origin
    with rasterio.open(
        ard_root / "s1a-iw-nrb-20250613t010546-vv-g-lin.tif",
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:32613",
        transform=from_origin(0, 200, 100, 100),
    ) as destination:
        destination.write(np.ones((2, 2), dtype=np.float32), 1)
    with rasterio.open(
        ard_root / "s1a-iw-nrb-20250613t010546-vh-g-lin.tif",
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:32613",
        transform=from_origin(0, 200, 100, 100),
    ) as destination:
        destination.write(np.full((2, 2), 0.5, dtype=np.float32), 1)
    ard = read_s1_ard(tmp_path / "ard_product")
    assert "gamma0_VV" in ard
    assert ard.attrs["product_type"] == "S1_NRB"


def test_sentinel1_catalog_searches_odata_excluding_cog():
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Session:
        def __init__(self):
            self.calls = []

        def get(self, url, *, params, timeout):
            self.calls.append(params)
            return Response({
                "value": [{
                    "Id": "abc-123",
                    "Name": "S1D_IW_GRDH_1SDV_20260801T165952_20260801T170017_003935_007203_B303.SAFE",
                    "Online": True,
                    "ContentDate": {"Start": "2026-08-01T16:59:52Z", "End": "2026-08-01T17:00:17Z"},
                    "Attributes": {"productType": "IW_GRDH_1S"},
                }]
            })

    session = Session()
    aoi = AOI(west=13.2, south=52.4, east=13.6, north=52.6)
    refs = Sentinel1Catalog(session=session).search(aoi, "2026-08-01", "2026-08-15", limit=5)

    assert len(refs) == 1
    assert refs[0].name.endswith(".SAFE")
    assert refs[0].online is True
    # The two real bugs this rewrite fixes: CDSE's default "COG" GRD
    # packaging (unreadable by pyroSAR/SNAP) and select_product_refs()
    # crashing on a bare STACItem (no .online/.cloud_cover/.start_datetime).
    assert "not contains(Name,'COG')" in session.calls[0]["$filter"]
    assert "IW_GRDH_1S" in session.calls[0]["$filter"]
    selected = select_product_refs(refs, limit=1)
    assert len(selected) == 1


def test_sentinel2_catalog_searches_l1c_product_type_on_request():
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Session:
        def __init__(self):
            self.calls = []

        def get(self, url, *, params, timeout):
            self.calls.append(params)
            return Response({"value": []})

    session = Session()
    aoi = AOI(west=13.2, south=52.4, east=13.6, north=52.6)

    Sentinel2Catalog(session=session).search(aoi, "2026-08-01", "2026-08-15")
    assert "S2MSI2A" in session.calls[0]["$filter"]

    Sentinel2Catalog(session=session, product_type=SENTINEL2_L1C_PRODUCT_TYPE).search(aoi, "2026-08-01", "2026-08-15")
    assert "S2MSI1C" in session.calls[1]["$filter"]

    with pytest.raises(ValueError):
        Sentinel2Catalog(session=session, product_type="bogus")


def test_read_s1_rtc_reads_hyp3_polarization_tifs(tmp_path: Path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    granule = "S1A_IW_GRDH_1SDV_20250613T010546_20250613T010611_012345_012345_ABCD"
    product_dir = tmp_path / granule
    product_dir.mkdir()
    for name, value in (("VV", 1.0), ("VH", 0.5)):
        with rasterio.open(
            product_dir / f"{granule}_{name}.tif",
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype="float32",
            crs="EPSG:32633",
            transform=from_origin(0, 200, 100, 100),
        ) as destination:
            destination.write(np.full((2, 2), value, dtype=np.float32), 1)

    dataset = read_s1_rtc(product_dir)
    assert "gamma0_VV" in dataset and "gamma0_VH" in dataset
    assert dataset.attrs["product_type"] == "S1_RTC"
    assert np.isclose(dataset["gamma0_VV"].values, 1.0).all()


def test_process_s1_rtc_dry_run_and_submit_poll_download(tmp_path: Path, monkeypatch):
    import sys
    import types
    import zipfile

    granule = "S1A_IW_GRDH_1SDV_20250613T010546_20250613T010611_012345_012345_ABCD"

    dry_run = process_s1_rtc(f"{granule}.SAFE", tmp_path / "processed", execute=False)
    assert dry_run.backend == "hyp3_rtc"
    assert dry_run.command[1] == granule

    # hyp3_sdk itself is an optional extra not installed in this test
    # environment - fake it out so the credentials check (the thing this
    # part of the test actually targets) is reachable.
    fake_module = types.ModuleType("hyp3_sdk")
    fake_module.HyP3 = lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not construct without credentials"))
    monkeypatch.setitem(sys.modules, "hyp3_sdk", fake_module)
    monkeypatch.delenv("EARTHDATA_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="Earthdata credentials"):
        process_s1_rtc(granule, tmp_path / "processed")

    monkeypatch.setenv("EARTHDATA_BEARER_TOKEN", "fake-token")

    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for name, value in (("VV", 1.0), ("VH", 0.5)):
        with rasterio.open(
            source_dir / f"{granule}_{name}.tif",
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype="float32",
            crs="EPSG:32633",
            transform=from_origin(0, 200, 100, 100),
        ) as destination:
            destination.write(np.full((2, 2), value, dtype=np.float32), 1)
    zip_path = tmp_path / f"{granule}.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for tif in source_dir.glob("*.tif"):
            archive.write(tif, tif.name)

    class FakeJob:
        def __init__(self):
            self.status_code = "SUCCEEDED"
            self.logs = []

        def succeeded(self):
            return True

        def download_files(self, location):
            location = Path(location)
            location.mkdir(parents=True, exist_ok=True)
            dest = location / zip_path.name
            dest.write_bytes(zip_path.read_bytes())
            return [dest]

    class FakeHyP3:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.submitted = []

        def submit_rtc_job(self, submitted_granule, **kwargs):
            self.submitted.append((submitted_granule, kwargs))
            return [FakeJob()]

        def watch(self, batch, timeout, interval):
            return batch

    fake_module = types.ModuleType("hyp3_sdk")
    fake_module.HyP3 = FakeHyP3
    monkeypatch.setitem(sys.modules, "hyp3_sdk", fake_module)

    result = process_s1_rtc(f"{granule}.SAFE", tmp_path / "processed", config=Sentinel1RTCConfig(resolution=20))
    assert result.backend == "hyp3_rtc"
    assert result.output_path.is_dir()
    dataset = read_s1_rtc(result.output_path)
    assert "gamma0_VV" in dataset


def test_reference_validation_metrics():
    estimate = xr.DataArray([2.0, 4.0, np.nan], dims="sample")
    reference = xr.DataArray([1.0, 5.0, 9.0], dims="sample")
    metrics = compare_to_reference(estimate, reference)
    assert metrics["samples"] == 2
    assert metrics["bias"] == 0.0
    assert np.isclose(metrics["rmse"], 1.0)


def test_write_cog_handles_ascending_y(tmp_path: Path):
    rasterio = pytest.importorskip("rasterio")
    layer = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        dims=("y", "x"),
        coords={"y": [50.0, 150.0], "x": [10.0, 110.0]},
        name="layer",
        attrs={"crs": "EPSG:32613"},
    )
    path = write_cog(layer, tmp_path / "layer.tif")
    window = read_cog(path, bbox=(10.0, 50.0, 110.0, 150.0))
    with rasterio.open(path) as source:
        assert source.crs.to_string() == "EPSG:32613"
        assert source.read(1).tolist() == [[3.0, 4.0], [1.0, 2.0]]
    assert window.shape == (1, 1)
    assert validate_cog(path)["tiled"] is True


def test_zarr_roundtrip_uses_interoperable_format(tmp_path: Path):
    pytest.importorskip("zarr")
    cube = xr.Dataset(
        {"lst": (("time", "y", "x"), np.ones((1, 2, 2), dtype=np.float32))},
        coords={"time": ["2025-01-01"], "y": [1, 0], "x": [0, 1]},
        attrs={"crs": "EPSG:32613"},
    )
    store = tmp_path / "cube.zarr"
    write_zarr(cube, store)
    opened = open_zarr(store)
    assert opened["lst"].shape == (1, 2, 2)
    assert opened.attrs["crs"] == "EPSG:32613"
    opened.close()


def test_sentinel5p_quality_filter_and_gridding(tmp_path: Path):
    source_path = tmp_path / "S5P_OFFL_L2__NO2____20250610T120000.nc"
    source = xr.Dataset(
        {
            "nitrogendioxide_tropospheric_column": (("time", "scanline", "ground_pixel"), np.array([[[1.0, 2.0], [3.0, 4.0]]])),
            "qa_value": (("time", "scanline", "ground_pixel"), np.array([[[0.9, 0.8], [0.2, 0.95]]])),
            "latitude": (("time", "scanline", "ground_pixel"), np.array([[[20.01, 20.01], [20.11, 20.11]]])),
            "longitude": (("time", "scanline", "ground_pixel"), np.array([[[-103.31, -103.21], [-103.31, -103.21]]])),
        },
        coords={"time": [0], "scanline": [0, 1], "ground_pixel": [0, 1]},
    )
    source.to_netcdf(source_path, group="PRODUCT")
    swath = read_s5p_l2(source_path, config=Sentinel5PReadConfig(gas="NO2"))
    assert swath.attrs["analysis_shape"] == "swath"
    assert str(swath.time.values[0]).startswith("2025-06-10")
    assert swath["valid_mask"].sum().item() == 3
    gridded = grid_s5p(swath, resolution_deg=0.1, aoi=AOI(-103.4, 19.9, -103.1, 20.2))
    assert gridded.attrs["crs"] == "EPSG:4326"
    assert 0 < gridded.sizes["y"] <= 4
    assert 0 < gridded.sizes["x"] <= 4
    assert gridded["NO2"].notnull().sum().item() == 3


def test_analysis_workflow_harmonizes_and_fuses_prepared_cubes():
    times = np.array(["2025-06-12", "2025-06-13"], dtype="datetime64[ns]")
    target = xr.Dataset(
        {"lst": (("time", "y", "x"), np.full((2, 2, 2), 30.0))},
        coords={"time": times, "y": [100.0, 0.0], "x": [0.0, 100.0]},
        attrs={"crs": "EPSG:32613", "grid_id": "g"},
    )
    feature = xr.Dataset(
        {"NDVI": (("time", "y", "x"), np.full((2, 3, 3), 0.5))},
        coords={"time": times + np.timedelta64(1, "D"), "y": [100.0, 50.0, 0.0], "x": [0.0, 50.0, 100.0]},
        attrs={"crs": "EPSG:32613", "grid_id": "feature"},
    )
    request = AnalysisRequest(
        aoi=AOI(0, 0, 1, 1),
        start="2025-06-12",
        end="2025-06-13",
        sensors=("sentinel3", "sentinel2"),
        target_sensor="sentinel3",
    )
    result = AnalysisWorkflow(request).run({"sentinel3": target, "sentinel2": feature})
    assert "NDVI" in result.cube
    assert "sentinel2_time_delta" in result.cube
    assert result.cube.attrs["workflow_target_sensor"] == "sentinel3"
    assert result.cube["feature_availability"].sizes["time"] == 2


def test_sentinel3_geolocation_bins_to_real_analysis_grid():
    from pyproj import Transformer

    grid = AnalysisGrid.from_bounds((500000, 2200000, 500200, 2200200), crs="EPSG:32613", resolution_m=100)
    transformer = Transformer.from_crs("EPSG:32613", "EPSG:4326", always_xy=True)
    x = np.array([500050.0, 500150.0])
    y = np.array([2200150.0, 2200050.0])
    lon, lat = transformer.transform(*np.meshgrid(x, y))
    source = xr.Dataset(
        {
            "lst": (("time", "y", "x"), np.array([[[300.0, 301.0], [302.0, 303.0]]])),
            "latitude": (("y", "x"), lat),
            "longitude": (("y", "x"), lon),
        },
        coords={"time": ["2025-06-12"], "y": [0, 1], "x": [0, 1]},
    )
    georeferenced = grid_l2_lst(source, grid)
    assert georeferenced.attrs["crs"] == "EPSG:32613"
    assert georeferenced["lst_observation_count"].sum().item() == 4
    assert georeferenced.sizes["y"] == 2
    converted = to_celsius(georeferenced)
    assert converted["lst"].attrs["units"] == "degC"
    assert np.nanmin(converted["lst"].values) > 20


def test_analysis_grid_chips_partitions_fixed_size_patches():
    # 100x60 px at 20m = 2000x1200m parent, chipped into 500m (25px) chips:
    # 4 whole columns, 2 whole rows, with a half-height row left over.
    grid = AnalysisGrid.from_bounds((0, 0, 2000, 1200), crs="EPSG:32633", resolution_m=20)

    chips = grid.chips(500)

    assert len(chips) == 4 * 2
    assert all(chip.width == 25 and chip.height == 25 for chip in chips)
    assert all(chip.crs == grid.crs and chip.resolution == grid.resolution for chip in chips)
    # Chips tile the parent grid without gaps or overlap.
    lefts = sorted({chip.bounds[0] for chip in chips})
    tops = sorted({chip.bounds[3] for chip in chips})
    assert lefts == [0.0, 500.0, 1000.0, 1500.0]
    assert tops == [700.0, 1200.0]
    assert len({chip.grid_id for chip in chips}) == len(chips)

    kept = grid.chips(500, drop_partial=False)
    assert len(kept) == 4 * 3
    assert any(chip.height < 25 for chip in kept)


def test_analysis_grid_chips_rejects_non_multiple_chip_size():
    grid = AnalysisGrid.from_bounds((0, 0, 1000, 1000), crs="EPSG:32633", resolution_m=20)
    with pytest.raises(ValueError):
        grid.chips(15)


def _synthetic_l2_lst_product(*, time: str, x: np.ndarray, y: np.ndarray, lst_value: float) -> xr.Dataset:
    """A raw (ungridded) Sentinel-3 L2 LST product with its own native pixel
    shape and geolocation, mirroring what Sentinel3LST.read() returns for one
    real acquisition."""

    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:32613", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(*np.meshgrid(x, y))
    shape = lat.shape
    return xr.Dataset(
        {
            "lst": (("time", "y", "x"), np.full((1, *shape), lst_value)),
            "latitude": (("y", "x"), lat),
            "longitude": (("y", "x"), lon),
        },
        coords={"time": [time], "y": np.arange(shape[0]), "x": np.arange(shape[1])},
    )


def test_combine_sentinel3_regrids_each_acquisition_before_concatenating():
    """Regression test for a real crash: two real Sentinel-3 acquisitions
    from different orbits have different native pixel-array shapes (e.g.
    1200 vs 1202 rows), so concatenating them raw - before regridding -
    fails with "cannot reindex or align along dimension 'y' because of
    conflicting dimension sizes". It also silently mis-georeferenced every
    slice but the first on the rare occasion native shapes did match,
    because both binning methods reused acquisition 1's geolocation for
    every time step (see _require_single_geolocation). _combine_sentinel3
    must regrid each acquisition individually - onto a shared, common-size
    grid - before concatenating, sidestepping both failure modes.
    """

    grid = AnalysisGrid.from_bounds((500000, 2200000, 500400, 2200400), crs="EPSG:32613", resolution_m=200)

    # Product A: 2x2 raw pixels, all falling in one grid cell (empirically
    # cell [y=0, x=0] for this geometry - see the assertions below).
    product_a = _synthetic_l2_lst_product(
        time="2025-06-10", x=np.array([500050.0, 500150.0]), y=np.array([2200150.0, 2200050.0]), lst_value=300.0,
    )
    # Product B: a *different* native shape (2x3, not 2x2 - the real-world
    # crash trigger), all falling in a different grid cell ([y=1, x=1]).
    product_b = _synthetic_l2_lst_product(
        time="2025-06-11",
        x=np.array([500250.0, 500300.0, 500350.0]),
        y=np.array([2200350.0, 2200250.0]),
        lst_value=500.0,
    )

    combined = _combine_sentinel3([product_a, product_b], grid)

    assert combined.sizes["time"] == 2
    lst = combined["lst"]

    # Time 0 (product A): its cell has A's value; B's cell is untouched.
    assert lst.isel(time=0, y=0, x=0).item() == pytest.approx(300.0)
    assert np.isnan(lst.isel(time=0, y=1, x=1).item())

    # Time 1 (product B): its cell has B's value, *not* a stale copy of A's
    # - this is what the pre-fix "reuse time=0 geolocation" bug got wrong
    # even when it didn't crash outright - and A's cell is untouched.
    assert lst.isel(time=1, y=1, x=1).item() == pytest.approx(500.0)
    assert np.isnan(lst.isel(time=1, y=0, x=0).item())


def test_combine_sentinel3_single_product_unchanged():
    grid = AnalysisGrid.from_bounds((500000, 2200000, 500200, 2200200), crs="EPSG:32613", resolution_m=100)
    product = _synthetic_l2_lst_product(
        time="2025-06-10", x=np.array([500050.0, 500150.0]), y=np.array([2200150.0, 2200050.0]), lst_value=300.0,
    )
    combined = _combine_sentinel3([product], grid)
    assert combined.sizes["time"] == 1
    assert np.nanmean(combined["lst"].isel(time=0).values) == pytest.approx(300.0)


def test_combine_sentinel3_requires_at_least_one_product():
    grid = AnalysisGrid.from_bounds((500000, 2200000, 500200, 2200200), crs="EPSG:32613", resolution_m=100)
    with pytest.raises(ValueError):
        _combine_sentinel3([], grid)


def test_grid_l2_lst_rejects_shared_geolocation_across_multiple_times():
    """Concatenating raw acquisitions with xr.concat(..., data_vars="minimal")
    drops the per-acquisition geolocation entirely, leaving one shared 2-D
    lat/lon reused for every time step - exactly what Sentinel3LST.combine()
    used to hand to grid_l2_lst. That must be a loud, explicit error, not a
    silently wrong regrid."""

    from pyproj import Transformer

    grid = AnalysisGrid.from_bounds((500000, 2200000, 500200, 2200200), crs="EPSG:32613", resolution_m=100)
    transformer = Transformer.from_crs("EPSG:32613", "EPSG:4326", always_xy=True)
    x = np.array([500050.0, 500150.0])
    y = np.array([2200150.0, 2200050.0])
    lon, lat = transformer.transform(*np.meshgrid(x, y))
    ambiguous = xr.Dataset(
        {
            "lst": (("time", "y", "x"), np.array([[[300.0, 301.0], [302.0, 303.0]], [[400.0, 401.0], [402.0, 403.0]]])),
            "latitude": (("y", "x"), lat),
            "longitude": (("y", "x"), lon),
        },
        coords={"time": ["2025-06-12", "2025-06-13"], "y": [0, 1], "x": [0, 1]},
    )
    with pytest.raises(CubeValidationError, match="georeference each Sentinel-3 acquisition individually"):
        grid_l2_lst(ambiguous, grid)
    with pytest.raises(CubeValidationError, match="georeference each Sentinel-3 acquisition individually"):
        grid_l2_lst(ambiguous, grid, method="point")


def test_grid_l2_lst_rejects_differing_per_time_geolocation():
    """Even when latitude/longitude do carry an explicit time dimension,
    slices that actually differ between acquisitions must not be silently
    collapsed to time=0's geolocation."""

    from pyproj import Transformer

    grid = AnalysisGrid.from_bounds((500000, 2200000, 500200, 2200200), crs="EPSG:32613", resolution_m=100)
    transformer = Transformer.from_crs("EPSG:32613", "EPSG:4326", always_xy=True)
    lon_a, lat_a = transformer.transform(*np.meshgrid([500050.0, 500150.0], [2200150.0, 2200050.0]))
    lon_b, lat_b = transformer.transform(*np.meshgrid([500060.0, 500160.0], [2200160.0, 2200060.0]))
    mismatched = xr.Dataset(
        {
            "lst": (("time", "y", "x"), np.array([[[300.0, 301.0], [302.0, 303.0]], [[400.0, 401.0], [402.0, 403.0]]])),
            "latitude": (("time", "y", "x"), np.stack([lat_a, lat_b])),
            "longitude": (("time", "y", "x"), np.stack([lon_a, lon_b])),
        },
        coords={"time": ["2025-06-12", "2025-06-13"], "y": [0, 1], "x": [0, 1]},
    )
    with pytest.raises(CubeValidationError, match="differing"):
        grid_l2_lst(mismatched, grid)


def test_grid_l2_lst_allows_multi_time_when_geolocation_is_truly_identical():
    """The guard must not reject the (rare but valid) case where several
    time steps genuinely share identical geolocation."""

    from pyproj import Transformer

    grid = AnalysisGrid.from_bounds((500000, 2200000, 500200, 2200200), crs="EPSG:32613", resolution_m=100)
    transformer = Transformer.from_crs("EPSG:32613", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(*np.meshgrid([500050.0, 500150.0], [2200150.0, 2200050.0]))
    identical = xr.Dataset(
        {
            "lst": (("time", "y", "x"), np.array([[[300.0, 301.0], [302.0, 303.0]], [[400.0, 401.0], [402.0, 403.0]]])),
            "latitude": (("time", "y", "x"), np.stack([lat, lat])),
            "longitude": (("time", "y", "x"), np.stack([lon, lon])),
        },
        coords={"time": ["2025-06-12", "2025-06-13"], "y": [0, 1], "x": [0, 1]},
    )
    georeferenced = grid_l2_lst(identical, grid)
    assert georeferenced.sizes["time"] == 2


def test_grid_l2_lst_area_method_bins_float_flag_variables():
    """Regression test for a real crash surfaced by real multi-product CDSE
    data: cloud_flags is float32 on disk (CF conventions promote integer
    bitmasks to float to represent missing values as NaN), and numpy's
    bitwise-or ufunc outright rejects float operands - "ufunc 'bitwise_or'
    not supported for the input types". exception_flags (int16) never
    triggered this; cloud_flags did, once real multi-orbit data actually
    reached the flag-binning step (see test_combine_sentinel3_* above for
    the separate bug that used to crash before ever getting this far)."""

    from pyproj import Transformer

    grid = AnalysisGrid.from_bounds((500000, 2200000, 500200, 2200200), crs="EPSG:32613", resolution_m=100)
    transformer = Transformer.from_crs("EPSG:32613", "EPSG:4326", always_xy=True)
    x = np.array([500050.0, 500150.0])
    y = np.array([2200150.0, 2200050.0])
    lon, lat = transformer.transform(*np.meshgrid(x, y))
    source = xr.Dataset(
        {
            "lst": (("time", "y", "x"), np.array([[[300.0, 301.0], [302.0, 303.0]]])),
            # int16, like the real exception_flags variable.
            "exception_flags": (("time", "y", "x"), np.array([[[1, 0], [2, 0]]], dtype=np.int16)),
            # float32 with a real NaN, like the real cloud_flags variable.
            "cloud_flags": (("time", "y", "x"), np.array([[[4.0, np.nan], [8.0, 0.0]]], dtype=np.float32)),
            "latitude": (("y", "x"), lat),
            "longitude": (("y", "x"), lon),
        },
        coords={"time": ["2025-06-12"], "y": [0, 1], "x": [0, 1]},
    )
    georeferenced = grid_l2_lst(source, grid)  # must not raise
    assert georeferenced["cloud_flags"].dtype.kind in "iu"
    assert int(georeferenced["cloud_flags"].sum().item()) > 0
    assert georeferenced["exception_flags"].dtype.kind in "iu"


def test_sentinel2_composite_reports_clear_coverage():
    optical = xr.Dataset(
        {"NDVI": (("time", "y", "x"), np.array([[[0.2, 0.4]], [[0.6, np.nan]]])),
         "valid_mask": (("time", "y", "x"), np.array([[[True, True]], [[True, False]]]))},
        coords={"time": ["2025-06-10", "2025-06-15"], "y": [0], "x": [0, 1]},
    )
    coverage = sentinel2_coverage(optical)
    composite = compose_s2(optical, min_observations=2)
    assert coverage["clear_pixel_count"].values.tolist() == [2, 1]
    assert composite["clear_observation_count"].item(0, 0) == 2
    assert composite["valid_mask"].item(0, 0)


def test_downscaling_split_and_reaggregation_consistency():
    coarse = xr.DataArray(
        np.full((1, 2, 2), 30.0),
        dims=("time", "y", "x"),
        coords={"time": ["2025-06-12"], "y": [150.0, 50.0], "x": [50.0, 150.0]},
        attrs={"crs": "EPSG:32613"},
    )
    fine = xr.DataArray(
        np.full((1, 4, 4), 30.0),
        dims=("time", "y", "x"),
        coords={"time": ["2025-06-12"], "y": [175.0, 125.0, 75.0, 25.0], "x": [25.0, 75.0, 125.0, 175.0]},
        attrs={"crs": "EPSG:32613"},
    )
    aggregated = reaggregate_to_target(fine, coarse)
    metrics = validate_reaggregation(coarse, fine, tolerance=0.01)
    assert np.allclose(aggregated.values, coarse.values)
    assert metrics["within_tolerance"]

    training = xr.Dataset(
        {"lst": (("time", "y", "x"), np.ones((5, 2, 2))), "NDVI": (("time", "y", "x"), np.ones((5, 2, 2)))},
        coords={"time": np.arange(5).astype("datetime64[D]"), "y": [0, 1], "x": [0, 1]},
        attrs={"crs": "EPSG:32613"},
    )
    train, validation = split_spatiotemporal(training, validation_fraction=0.4)
    assert np.isfinite(train["lst"]).sum().item() > 0
    assert np.isfinite(validation["lst"]).sum().item() > 0


def test_random_forest_downscaler_is_optional():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(42)
    ndvi = rng.random((120, 2, 2))
    training = xr.Dataset(
        {"lst": (("time", "y", "x"), 20 + 5 * ndvi), "NDVI": (("time", "y", "x"), ndvi)},
        coords={"time": np.arange(120).astype("datetime64[D]"), "y": [0, 1], "x": [0, 1]},
        attrs={"crs": "EPSG:32613"},
    )
    model = fit_random_forest_downscaler(training, predictors=("NDVI",), n_estimators=10, min_samples=100)
    result = model.predict(training.isel(time=slice(0, 1)))
    assert result["downscaled_support"].sum().item() == 4


def test_tsharp_downscaler_is_a_single_predictor_coarse_relation():
    values = np.arange(16, dtype=float).reshape(4, 2, 2)
    training = xr.Dataset(
        {"lst": (("time", "y", "x"), 280 + values), "NDVI": (("time", "y", "x"), values / 20)},
        coords={"time": np.arange(4), "y": [0, 1], "x": [0, 1]},
        attrs={"crs": "EPSG:32633"},
    )
    model = fit_tsharp_downscaler(training, predictor="NDVI", min_samples=4)
    result = model.predict(training[["NDVI"]])
    assert result.attrs["downscaling_method"] == "ts_harp_dis_trad"
    assert result["lst_downscaled"].notnull().all().item()


def test_gwr_downscaler_adapts_locally_where_a_global_linear_model_cannot():
    """Two well-separated locations with opposite NDVI-LST slopes - the
    local non-stationarity a geographically weighted baseline exists for
    (docs/downscaling.md cites MFGWML for exactly this). A single global
    linear relation cannot fit both at once and should land near a
    canceled-out ~0 slope; GWR, with a bandwidth spanning only one cluster,
    should recover each cluster's own slope instead."""

    pytest.importorskip("scipy")
    n_time = 60
    rng = np.random.default_rng(0)
    ndvi_a = rng.uniform(0, 1, n_time)
    ndvi_b = rng.uniform(0, 1, n_time)
    lst_a = 300 + 5.0 * ndvi_a  # positive slope
    lst_b = 300 - 5.0 * ndvi_b  # negative slope

    # x=[0, 100] is cluster A; x=[100000, 100100] is cluster B - far enough
    # apart that a bandwidth of 200 includes only one cluster's own points.
    x_coords = [0.0, 100.0, 100000.0, 100100.0]
    lst = np.stack([lst_a, lst_a, lst_b, lst_b], axis=-1)[:, None, :]
    ndvi = np.stack([ndvi_a, ndvi_a, ndvi_b, ndvi_b], axis=-1)[:, None, :]

    training = xr.Dataset(
        {"lst": (("time", "y", "x"), lst), "NDVI": (("time", "y", "x"), ndvi)},
        coords={"time": np.arange(n_time), "y": [0.0], "x": x_coords},
        attrs={"crs": "EPSG:32633"},
    )
    model = fit_gwr_downscaler(
        training, predictors=("NDVI",), bandwidth=200.0, min_samples=100, min_local_samples=20, max_local_samples=200,
    )
    assert np.isfinite(model.residual_std)

    query_a = xr.Dataset({"NDVI": (("y", "x"), [[0.9]])}, coords={"y": [0.0], "x": [0.0]}, attrs={"crs": "EPSG:32633"})
    query_b = xr.Dataset({"NDVI": (("y", "x"), [[0.9]])}, coords={"y": [0.0], "x": [100000.0]}, attrs={"crs": "EPSG:32633"})

    predicted_a = float(model.predict(query_a)["lst_downscaled"].item())
    predicted_b = float(model.predict(query_b)["lst_downscaled"].item())
    assert predicted_a == pytest.approx(300 + 5.0 * 0.9, abs=1.0)
    assert predicted_b == pytest.approx(300 - 5.0 * 0.9, abs=1.0)

    # The point of the test: a single global relation cannot do this - its
    # one slope averages the two opposing +-5 local slopes toward something
    # much smaller in magnitude (not exactly 0: the two clusters' NDVI
    # values are independently random, so the cancellation isn't perfect).
    global_model = fit_linear_downscaler(training, predictors=("NDVI",), min_samples=100)
    assert abs(global_model.coefficients[0]) < 2.5


def test_coarse_consistent_gwr_downscaler_conserves_target_scale():
    pytest.importorskip("scipy")
    time = np.arange(4, dtype="int64").astype("datetime64[D]")
    coarse = xr.Dataset(
        {
            "lst": (("time", "y", "x"), np.full((4, 2, 2), 300.0)),
            "NDVI": (("time", "y", "x"), np.full((4, 2, 2), 0.5)),
        },
        coords={"time": time, "y": [150.0, 50.0], "x": [50.0, 150.0]},
        attrs={"crs": "EPSG:32613", "grid_id": "coarse"},
    )
    fine = xr.Dataset(
        {"NDVI": (("time", "y", "x"), np.full((4, 4, 4), 0.5))},
        coords={"time": time, "y": [175.0, 125.0, 75.0, 25.0], "x": [25.0, 75.0, 125.0, 175.0]},
        attrs={"crs": "EPSG:32613", "grid_id": "fine"},
    )
    model = fit_coarse_consistent_gwr_downscaler(
        coarse, predictors=("NDVI",), bandwidth=200.0, min_samples=4, min_local_samples=4, max_local_samples=16,
    )

    result = model.predict(fine, coarse)

    assert result.attrs["coarse_consistency_within_tolerance"]
    assert np.allclose(reaggregate_to_target(result["lst_downscaled"], coarse["lst"]).values, coarse["lst"].values)


def test_starfm_fuses_fine_texture_with_coarse_change_and_respects_similarity():
    """Left two columns and right two columns are different surfaces at t0
    (10 vs 30) - only the right surface actually changes (+5) between t0
    and t1. STARFM's similarity filter must keep that change from bleeding
    into the left surface's own prediction, even though the window (radius
    2 on a width-4 grid) spans both surfaces."""

    fine_t0 = xr.DataArray(
        np.array([[10.0, 10.0, 30.0, 30.0]] * 4),
        dims=("y", "x"),
        coords={"y": np.arange(4) * 100.0, "x": np.arange(4) * 100.0},
        attrs={"crs": "EPSG:32633"},
    )
    coarse_t0 = xr.DataArray(np.zeros((4, 4)), dims=("y", "x"), coords=fine_t0.coords)
    coarse_t1 = xr.DataArray(
        np.array([[0.0, 0.0, 5.0, 5.0]] * 4), dims=("y", "x"), coords=fine_t0.coords,
    )

    result = fuse_starfm(fine_t0, coarse_t0, coarse_t1, window_radius=2, similarity_threshold_multiplier=0.5)
    predicted = result["lst_downscaled"].values

    assert predicted[:, :2] == pytest.approx(10.0, abs=0.5)
    assert predicted[:, 2:] == pytest.approx(35.0, abs=0.5)
    assert result["downscaled_support"].values.all()
    assert result.attrs["downscaling_method"] == "starfm"


def test_starfm_reindexes_mismatched_coarse_grids_onto_the_fine_grid():
    """coarse_t0/coarse_t1 need not already be on fine_t0's grid - a real
    caller has a genuinely coarser sensor's pixels, not a pre-resampled
    copy - fuse_starfm reindexes them (nearest-neighbor) itself."""

    fine_t0 = xr.DataArray(
        np.full((4, 4), 20.0),
        dims=("y", "x"),
        coords={"y": np.arange(4) * 100.0, "x": np.arange(4) * 100.0},
        attrs={"crs": "EPSG:32633"},
    )
    # A 2x2 coarse grid - coarser resolution, on different coordinates.
    coarse_t0 = xr.DataArray(
        np.zeros((2, 2)), dims=("y", "x"), coords={"y": [50.0, 250.0], "x": [50.0, 250.0]},
    )
    coarse_t1 = xr.DataArray(
        np.full((2, 2), 3.0), dims=("y", "x"), coords={"y": [50.0, 250.0], "x": [50.0, 250.0]},
    )

    result = fuse_starfm(fine_t0, coarse_t0, coarse_t1, window_radius=1)
    assert result["lst_downscaled"].values == pytest.approx(23.0, abs=0.5)


def test_estarfm_local_regression_beats_starfms_raw_difference_on_a_slope_change():
    """Two surfaces (left/right) where fine responds to coarse change at a
    *different rate* (slope), not just a different offset - exactly the
    case docs/downscaling.md flags single-pair STARFM's raw additive
    difference as unable to represent. Each surface has real within-surface
    spatial variation in coarse/fine values across rows so the per-window
    regression has something to fit (a perfectly uniform surface has no
    coarse variance to regress against - see the identity-fallback test
    below for that case instead).

    Surface A (columns 0-1): fine = 100 + 2*coarse at both t0 and t2 (slope 2).
    Surface B (columns 2-3): fine = 300 + 0.5*coarse at both t0 and t2 (slope 0.5).
    coarse_t1 sits exactly halfway between coarse_t0 and coarse_t2 everywhere.

    A correct per-surface regression recovers each true slope and predicts
    the exact noise-free t1 value from *either* input pair - a naive
    STARFM-style slope=1 raw difference would not (it would use `fine0 +
    (c1 - c0)` = fine0 + 0.5 uniformly, off by a surface-dependent amount).
    """

    rows = np.arange(4.0)[:, None]  # (4, 1) - broadcasts across each surface's 2 columns
    ones = np.ones((4, 2))

    coarse_t0 = np.concatenate([rows * ones, rows * ones], axis=1)  # 0,1,2,3 per row, both surfaces
    coarse_t2 = coarse_t0 + 1.0
    coarse_t1 = coarse_t0 + 0.5

    fine_a_t0, fine_a_t2 = 100 + 2 * coarse_t0[:, :2], 100 + 2 * coarse_t2[:, :2]
    fine_b_t0, fine_b_t2 = 300 + 0.5 * coarse_t0[:, 2:], 300 + 0.5 * coarse_t2[:, 2:]
    fine_t0_values = np.concatenate([fine_a_t0, fine_b_t0], axis=1)
    fine_t2_values = np.concatenate([fine_a_t2, fine_b_t2], axis=1)

    coords = {"y": np.arange(4) * 100.0, "x": np.arange(4) * 100.0}
    fine_t0 = xr.DataArray(fine_t0_values, dims=("y", "x"), coords=coords, attrs={"crs": "EPSG:32633"})
    coarse_t0_da = xr.DataArray(coarse_t0, dims=("y", "x"), coords=coords)
    fine_t2 = xr.DataArray(fine_t2_values, dims=("y", "x"), coords=coords)
    coarse_t2_da = xr.DataArray(coarse_t2, dims=("y", "x"), coords=coords)
    coarse_t1_da = xr.DataArray(coarse_t1, dims=("y", "x"), coords=coords)

    result = fuse_estarfm(
        fine_t0, coarse_t0_da, fine_t2, coarse_t2_da, coarse_t1_da,
        window_radius=2, similarity_threshold_multiplier=0.5,
    )
    predicted = result["lst_downscaled"].values

    expected_a = 100 + 2 * coarse_t1[:, :2]
    expected_b = 300 + 0.5 * coarse_t1[:, 2:]
    assert predicted[:, :2] == pytest.approx(expected_a, abs=0.1)
    assert predicted[:, 2:] == pytest.approx(expected_b, abs=0.1)

    # The point of the test: a naive raw-difference (slope=1) prediction
    # would have been off by a surface-dependent amount - confirm this
    # synthetic case actually distinguishes the two approaches, not just
    # that fuse_estarfm ran without crashing.
    naive_a = fine_a_t0 + (coarse_t1[:, :2] - coarse_t0[:, :2])
    assert not np.allclose(naive_a, expected_a)
    assert result.attrs["downscaling_method"] == "estarfm"


def test_estarfm_falls_back_to_identity_and_reindexes_mismatched_grids():
    """A uniform region has no local coarse variance to regress against -
    both pairs fall back to the identity slope (fuse_starfm's own
    behavior), and must still be reported as supported (not silently
    dropped just because no real regression fit). Also exercises
    reindexing coarse/fine_t2 inputs from a different, coarser grid onto
    fine_t0's, like fuse_starfm's own equivalent test."""

    fine_t0 = xr.DataArray(
        np.full((4, 4), 20.0),
        dims=("y", "x"),
        coords={"y": np.arange(4) * 100.0, "x": np.arange(4) * 100.0},
        attrs={"crs": "EPSG:32633"},
    )
    coarse_coords = {"y": [50.0, 250.0], "x": [50.0, 250.0]}
    coarse_t0 = xr.DataArray(np.zeros((2, 2)), dims=("y", "x"), coords=coarse_coords)
    fine_t2 = xr.DataArray(
        np.full((4, 4), 22.0), dims=("y", "x"), coords={"y": np.arange(4) * 100.0, "x": np.arange(4) * 100.0},
    )
    coarse_t2 = xr.DataArray(np.full((2, 2), 2.0), dims=("y", "x"), coords=coarse_coords)
    coarse_t1 = xr.DataArray(np.full((2, 2), 1.0), dims=("y", "x"), coords=coarse_coords)

    result = fuse_estarfm(fine_t0, coarse_t0, fine_t2, coarse_t2, coarse_t1, window_radius=1)

    # Both pairs fall back to slope=1: fine0 + (1-0) = 21, fine2 + (1-2) = 21.
    assert result["lst_downscaled"].values == pytest.approx(21.0, abs=0.5)
    assert result["downscaled_support"].values.all()


def test_product_selection_prefers_clear_online_products():
    def product(identifier: str, cloud: float, online: bool) -> ProductRef:
        return ProductRef(
            product_id=identifier,
            name=identifier,
            product_type="test",
            start_datetime=None,
            end_datetime=None,
            timeliness=None,
            online=online,
            download_url="https://example.test",
            metadata={"attributes": {"cloudCover": cloud}},
        )

    selected = select_product_refs([product("cloudy", 80, True), product("clear", 5, True), product("offline", 0, False)], limit=2)
    assert [item.product_id for item in selected] == ["clear", "cloudy"]


def test_blocked_validation_and_independent_reference_metrics():
    values = np.arange(32, dtype=float).reshape(8, 2, 2)
    cube = xr.Dataset(
        {"lst": (("time", "y", "x"), values), "feature": (("time", "y", "x"), values / 10)},
        coords={"time": np.arange(8), "y": [0, 1], "x": [0, 1]},
        attrs={"crs": "EPSG:32633"},
    )
    train, validation = blocked_spatiotemporal_split(cube, validation_fraction=0.25, spatial_block_period=2)
    assert np.isfinite(train["lst"].values[:6]).any()
    assert np.isfinite(validation["lst"].values[6:]).any()
    metrics = validate_independent_reference(cube["lst"], cube["lst"], reference_name="landsat_lst")
    assert metrics["rmse"] == 0.0
    assert metrics["independent_reference"] is True


def test_atmospheric_column_conversion_is_explicitly_modelled():
    dataset = xr.Dataset({"NO2": ("time", [1e-5]), "boundary_layer_height": ("time", [1000.0])}, coords={"time": ["2025-06-12"]})
    result = column_to_surface_estimate(dataset, config=ColumnToSurfaceConfig(gas="NO2"))
    assert result["NO2_surface_estimated"].attrs["modelled"] is True
    assert result["NO2_surface_estimated"].attrs["units"] == "ug m-3"


def test_fusion_quality_excludes_targets_and_diagnostics():
    from sentinel_analysis import fusion_quality

    dataset = xr.Dataset(
        {
            "lst": (("time", "y", "x"), np.ones((1, 1, 2))),
            "NDVI": (("time", "y", "x"), np.array([[[0.5, np.nan]]]), {"variable_role": "feature"}),
            "lst_observation_count": (("time", "y", "x"), np.zeros((1, 1, 2), dtype=np.int32)),
        },
        coords={"time": ["2025-06-12"], "y": [0], "x": [0, 1]},
        attrs={"crs": "EPSG:32613"},
    )
    result = fusion_quality(dataset)
    assert result["feature_availability"].item() == 0.5


def test_station_collocation_reports_metrics():
    cube = xr.Dataset(
        {"lst": (("time", "y", "x"), np.array([[[300.0]], [[301.0]]]))},
        coords={"time": ["2025-06-12", "2025-06-13"], "y": [20.0], "x": [-103.0]},
        attrs={"crs": "EPSG:4326"},
    )
    stations = xr.Dataset(
        {"observed": (("station", "time"), np.array([[300.0, 302.0]]))},
        coords={"station": ["A"], "time": ["2025-06-12", "2025-06-13"], "latitude": ("station", [20.0]), "longitude": ("station", [-103.0])},
    )
    report = collocate_stations(cube, stations, variable="lst", station_variable="observed")
    assert report["A"]["samples"] == 2


def _write_single_band_geotiff(path, values, *, transform, crs, nodata=None):
    import rasterio

    height, width = values.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype=values.dtype, crs=crs, transform=transform, nodata=nodata,
    ) as destination:
        destination.write(values, 1)


def test_read_landsat_lst_reprojects_masks_and_converts_units(tmp_path):
    """Landsat Collection 2 Level-2 is already an ortho-rectified GeoTIFF (unlike
    Sentinel-3's raw swath), so reading it is one rasterio reproject per band. This
    exercises that path with a synthetic scene: one QA-flagged (cloud) pixel, one
    nodata pixel, and known scale/offset - the real values confirmed against a live
    Planetary Computer STAC search for the `landsat-c2-l2` collection."""

    from affine import Affine

    # Bounds chosen as exact multiples of the 30 m resolution so
    # AnalysisGrid.from_bounds's floor/ceil snapping is a no-op and the
    # grid comes out exactly 4x4, matching the synthetic source raster below.
    grid = AnalysisGrid.from_bounds((499980, 2199900, 500100, 2200020), crs="EPSG:32633", resolution_m=30)
    transform = Affine(30, 0, 499980, 0, -30, 2200020)
    assert (grid.width, grid.height) == (4, 4)

    scale, offset, lwir_nodata = 0.00341802, 149.0, 0
    raw_dn = 44177  # -> ~300.0 K -> ~26.9 degC, chosen to be physically unremarkable
    lwir_raw = np.full((4, 4), raw_dn, dtype=np.uint16)
    lwir_raw[3, :] = lwir_nodata  # bottom row: no data at all

    qa_raw = np.zeros((4, 4), dtype=np.uint16)
    qa_raw[0, 0] = 0b1000  # "cloud" bit (bit 3) set - must be masked out

    lwir_path = tmp_path / "LC09_ST_B10.TIF"
    qa_path = tmp_path / "LC09_QA_PIXEL.TIF"
    _write_single_band_geotiff(lwir_path, lwir_raw, transform=transform, crs="EPSG:32633")
    _write_single_band_geotiff(qa_path, qa_raw, transform=transform, crs="EPSG:32633")

    product = ProductRef(
        product_id="LC09_L2SP_192024_20240825_20240826_02_T2",
        name="LC09_L2SP_192024_20240825_20240826_02_T2",
        product_type="L2SP",
        start_datetime="2024-08-25T10:00:00Z",
        end_datetime="2024-08-25T10:00:00Z",
        timeliness=None,
        online=True,
        download_url=str(lwir_path),
        metadata={
            "attributes": {"cloudCover": 12.3},
            "assets": {
                "lwir11": {"href": str(lwir_path), "raster:bands": [{"scale": scale, "offset": offset, "nodata": lwir_nodata}]},
                "qa_pixel": {"href": str(qa_path), "raster:bands": [{"nodata": None}]},
            },
        },
    )

    dataset = read_landsat_lst(product, grid)

    expected_celsius = raw_dn * scale + offset - 273.15
    lst = dataset["landsat_lst"].isel(time=0).values
    valid = dataset["valid_mask"].isel(time=0).values

    assert dataset["landsat_lst"].attrs["units"] == "degC"
    assert dataset["landsat_lst"].attrs["sensor"] == "Landsat-8/9"
    assert dataset["landsat_lst"].attrs["product"] == "L2SP"
    assert dataset.attrs["crs"] == "EPSG:32633"

    # Row 0, col 1: clear and has data -> converted value, not masked.
    assert valid[0, 1]
    assert lst[0, 1] == pytest.approx(expected_celsius, abs=1e-6)

    # Row 0, col 0: cloud-flagged -> masked out despite having a finite raw value.
    assert not valid[0, 0]
    assert np.isnan(lst[0, 0])

    # Row 3: nodata in the source band -> masked out.
    assert not valid[3, :].any()
    assert np.isnan(lst[3, :]).all()


def test_landsat_catalog_search_maps_stac_items_to_product_refs():
    fake_item = STACItem(
        item_id="LC09_L2SP_192024_20240825_20240826_02_T2",
        collections=("landsat-c2-l2",),
        datetime="2024-08-25T10:00:00Z",
        bbox=(13.2, 52.4, 13.55, 52.58),
        assets={
            "lwir11": {"href": "https://example.blob.core.windows.net/LC09_ST_B10.TIF", "raster:bands": [{"scale": 0.00341802, "offset": 149.0, "nodata": 0}]},
            "qa_pixel": {"href": "https://example.blob.core.windows.net/LC09_QA_PIXEL.TIF"},
        },
        properties={"eo:cloud_cover": 12.3, "platform": "landsat-9", "datetime": "2024-08-25T10:00:00Z"},
        raw={},
    )

    class _FakeSTACCatalog:
        def search(self, **kwargs):
            self.last_call = kwargs
            return [fake_item]

    fake_catalog = _FakeSTACCatalog()
    catalog = LandsatCatalog(catalog=fake_catalog)
    refs = catalog.search(AOI(13.20, 52.40, 13.55, 52.58), "2024-08-01", "2024-08-31", cloud_cover_max=30)

    assert len(refs) == 1
    ref = refs[0]
    assert ref.product_id == "LC09_L2SP_192024_20240825_20240826_02_T2"
    assert ref.download_url == "https://example.blob.core.windows.net/LC09_ST_B10.TIF"
    assert ref.cloud_cover == pytest.approx(12.3)
    assert fake_catalog.last_call["collections"] == ["landsat-c2-l2"]
    assert fake_catalog.last_call["query"]["eo:cloud_cover"] == {"lt": 30}


def test_stac_item_to_product_ref_requires_thermal_asset():
    item = STACItem(
        item_id="no-thermal-band",
        collections=("landsat-c2-l2",),
        datetime="2024-08-25T10:00:00Z",
        bbox=None,
        assets={"red": {"href": "https://example.com/red.tif"}},
        properties={},
        raw={},
    )
    with pytest.raises(ValueError, match="lwir11"):
        _stac_item_to_product_ref(item)


def test_read_ecostress_lst_requires_bearer_token(monkeypatch):
    monkeypatch.delenv("EARTHDATA_BEARER_TOKEN", raising=False)
    grid = AnalysisGrid.from_bounds((499980, 2199900, 500100, 2200020), crs="EPSG:32633", resolution_m=30)
    product = ProductRef(
        product_id="ECOv003_L2T_LSTE_test", name="ECOv003_L2T_LSTE_test", product_type="L2T_LSTE",
        start_datetime="2025-10-06T00:42:00Z", end_datetime="2025-10-06T00:42:52Z",
        timeliness=None, online=True, download_url="https://example.com/_LST.tif",
        metadata={"assets": {"https://example.com/_LST.tif": {"href": "https://example.com/_LST.tif"}}},
    )
    with pytest.raises(RuntimeError, match="EARTHDATA_BEARER_TOKEN"):
        read_ecostress_lst(product, grid)


def test_read_ecostress_lst_reprojects_masks_and_converts_units(tmp_path, monkeypatch):
    """ECOSTRESS's own CMR-STAC bridge exposes no raster:bands scale/offset
    (unlike Landsat via Planetary Computer) - read_ecostress_lst falls back
    to _FALLBACK_LST_SCALE/_FALLBACK_LST_OFFSET (identity) when a GeoTIFF
    doesn't embed its own, since a live read (2026-09) confirmed real L2T
    LSTE tiles store LST directly in Kelvin, not ATBD digital counts (an
    earlier, unverified 0.02-scale assumption produced ~5 K nonsense against
    real data). Masking uses the dedicated `cloud` band, not a QC bitmask,
    per the module's documented (unverified-live at the time) choice."""

    monkeypatch.setenv("EARTHDATA_BEARER_TOKEN", "test-token")
    from affine import Affine

    grid = AnalysisGrid.from_bounds((499980, 2199900, 500100, 2200020), crs="EPSG:32633", resolution_m=30)
    transform = Affine(30, 0, 499980, 0, -30, 2200020)
    assert (grid.width, grid.height) == (4, 4)

    dn = 300  # already Kelvin - fallback scale/offset are identity, not a rescale
    lst_raw = np.full((4, 4), dn, dtype=np.uint16)
    cloud_raw = np.zeros((4, 4), dtype=np.uint8)
    cloud_raw[0, 0] = 1  # cloud-flagged - must be masked out

    lst_path = tmp_path / "ECOv003_L2T_LSTE_test_LST.tif"
    cloud_path = tmp_path / "ECOv003_L2T_LSTE_test_cloud.tif"
    _write_single_band_geotiff(lst_path, lst_raw, transform=transform, crs="EPSG:32633")
    _write_single_band_geotiff(cloud_path, cloud_raw, transform=transform, crs="EPSG:32633")

    # Keys mirror the real (unusable) CMR-STAC bridge shape - a caller must
    # never rely on the dict key itself, only on _find_asset's href match.
    product = ProductRef(
        product_id="ECOv003_L2T_LSTE_41112_011_01UBU_20251006T004200_03",
        name="ECOv003_L2T_LSTE_41112_011_01UBU_20251006T004200_03",
        product_type="L2T_LSTE",
        start_datetime="2025-10-06T00:42:00.100Z",
        end_datetime="2025-10-06T00:42:52.069Z",
        timeliness=None,
        online=True,
        download_url=str(lst_path),
        metadata={
            "assets": {
                "003/.../..._LST": {"href": str(lst_path)},
                "003/.../..._cloud": {"href": str(cloud_path)},
            },
        },
    )

    dataset = read_ecostress_lst(product, grid)

    expected_celsius = dn * _FALLBACK_LST_SCALE + _FALLBACK_LST_OFFSET - 273.15
    lst = dataset["ecostress_lst"].isel(time=0).values
    valid = dataset["valid_mask"].isel(time=0).values

    assert dataset["ecostress_lst"].attrs["units"] == "degC"
    assert dataset["ecostress_lst"].attrs["sensor"] == "ECOSTRESS"
    assert dataset["ecostress_lst"].attrs["product"] == "L2T_LSTE"

    assert valid[0, 1]
    # abs=1e-3, not 1e-6 like the Landsat test: the dataset stores lst_kelvin
    # as float32 (~7 significant digits), and a ~300 K value's absolute
    # float32 rounding error already exceeds 1e-6 - a real precision limit,
    # not a logic bug.
    assert lst[0, 1] == pytest.approx(expected_celsius, abs=1e-3)
    assert not valid[0, 0]
    assert np.isnan(lst[0, 0])


def test_ecostress_catalog_search_maps_stac_items_and_skips_non_data_items():
    good_item = STACItem(
        item_id="ECOv003_L2T_LSTE_good",
        collections=("ECO_L2T_LSTE_003",),
        datetime="2025-10-06T00:42:00.100Z",
        bbox=(13.2, 52.4, 13.55, 52.58),
        assets={
            "003/.../..._LST": {"href": "https://data.lpdaac.earthdatacloud.nasa.gov/.../..._LST.tif"},
            "003/.../..._cloud": {"href": "https://data.lpdaac.earthdatacloud.nasa.gov/.../..._cloud.tif"},
        },
        properties={"start_datetime": "2025-10-06T00:42:00.100Z", "end_datetime": "2025-10-06T00:42:52.069Z"},
        raw={},
    )
    non_data_item = STACItem(
        item_id="ECOv003_L2T_LSTE_no_lst",
        collections=("ECO_L2T_LSTE_003",),
        datetime="2025-10-06T00:42:00.100Z",
        bbox=None,
        assets={"metadata": {"href": "https://example.com/metadata.xml"}},
        properties={},
        raw={},
    )

    class _FakeSTACCatalog:
        def search(self, **kwargs):
            self.last_call = kwargs
            return [good_item, non_data_item]

    fake_catalog = _FakeSTACCatalog()
    catalog = EcostressCatalog(catalog=fake_catalog)
    refs = catalog.search(AOI(13.20, 52.40, 13.55, 52.58), "2025-10-01", "2025-10-31")

    assert len(refs) == 1  # non_data_item silently skipped, not raised
    assert refs[0].product_id == "ECOv003_L2T_LSTE_good"
    assert refs[0].download_url.endswith("_LST.tif")
    assert fake_catalog.last_call["collections"] == ["ECO_L2T_LSTE_003"]
