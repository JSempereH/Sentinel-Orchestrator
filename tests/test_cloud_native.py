"""Cloud-native (STAC/COG) acquisition paths and the sensor adapter registry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from citycube import AOI, AnalysisGrid, AnalysisRequest, AnalysisWorkflow, ProductRef
from citycube.providers import AUXILIARY_PROVIDER_FACTORIES, AUXILIARY_PROVIDERS
from citycube.sensors.sentinel1 import Sentinel1RTCSTACCatalog, read_s1_rtc_cog
from citycube.sensors.sentinel2 import Sentinel2STACCatalog, read_s2_l2a_cog
from citycube.stac import STACItem
from citycube.workflow import adapters
from citycube.workflow.request import SUPPORTED_SENSORS

pytest.importorskip("rasterio")

CRS = "EPSG:32633"


def _write_tif(path: Path, values: np.ndarray, *, nodata=None) -> str:
    import rasterio
    from affine import Affine

    height, width = values.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1, dtype=values.dtype,
        crs=CRS, transform=Affine(10, 0, 500000, 0, -10, 4000040), nodata=nodata,
    ) as destination:
        destination.write(values, 1)
    return str(path)


def _grid() -> AnalysisGrid:
    # 40x40 m at 20 m: each target cell averages a 2x2 block of the 10 m source.
    return AnalysisGrid.from_bounds((500000, 4000000, 500040, 4000040), crs=CRS, resolution_m=20)


def _product(assets: dict, properties: dict | None = None, *, product_type: str = "S2MSI2A") -> ProductRef:
    return ProductRef(
        product_id="item-1", name="item-1", product_type=product_type,
        start_datetime="2025-06-10T10:30:00Z", end_datetime="2025-06-10T10:30:00Z",
        timeliness=None, online=True, download_url="",
        metadata={"attributes": {"cloudCover": 3.0}, "properties": properties or {}, "assets": assets},
    )


def test_read_s2_l2a_cog_applies_baseline_offset_scl_mask_and_footprint(tmp_path):
    dn = np.full((4, 4), 2000, dtype=np.uint16)
    dn[0, 0] = 0  # nodata pixel inside the first 2x2 block
    scl = np.full((4, 4), 4, dtype=np.uint8)  # vegetation (clear)
    scl[2:, 2:] = 9  # high-probability cloud in the bottom-right block
    bands = ("B02", "B03", "B04", "B08", "B11", "B12")
    assets = {band: {"href": _write_tif(tmp_path / f"{band}.tif", dn, nodata=0)} for band in bands}
    assets["SCL"] = {"href": _write_tif(tmp_path / "SCL.tif", scl, nodata=0)}

    cube = read_s2_l2a_cog(_product(assets, {"s2:processing_baseline": "05.11"}), _grid())

    # Baseline >= 04.00: reflectance = (DN - 1000) / 10000; nodata excluded from the average.
    assert cube["B04"].isel(time=0).values == pytest.approx(np.full((2, 2), 0.1), abs=1e-6)
    assert cube["valid_mask"].isel(time=0).values.tolist() == [[True, True], [True, False]]
    assert cube.attrs["crs"] == CRS
    assert cube.time.values[0] == np.datetime64("2025-06-10T10:30:00", "ns")


def test_read_s2_l2a_cog_trusts_published_scale_offset_and_marks_outside_footprint_invalid(tmp_path):
    dn = np.full((2, 2), 2000, dtype=np.uint16)  # covers only the top-left target cell
    bands = ("B02", "B03", "B04", "B08", "B11", "B12")
    published = {"raster:bands": [{"scale": 0.0001, "offset": -0.1, "nodata": 0}]}
    # Earth Search style common-name assets.
    names = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir", "B11": "swir16", "B12": "swir22"}
    assets = {names[band]: {"href": _write_tif(tmp_path / f"{band}.tif", dn, nodata=0), **published} for band in bands}
    assets["scl"] = {"href": _write_tif(tmp_path / "SCL.tif", np.full((2, 2), 4, dtype=np.uint8), nodata=0)}

    cube = read_s2_l2a_cog(_product(assets), _grid())

    assert cube["B04"].isel(time=0, y=0, x=0).item() == pytest.approx(0.1, abs=1e-6)
    assert np.isnan(cube["B04"].isel(time=0, y=1, x=1).item())
    assert not bool(cube["valid_mask"].isel(time=0, y=1, x=1).item())


def test_read_s1_rtc_cog_averages_linear_power_and_masks_nodata(tmp_path):
    vv = np.full((4, 4), 0.2, dtype=np.float32)
    vv[0, :2] = -32768.0
    vv[1, :2] = -32768.0  # whole first block is nodata
    vh = np.full((4, 4), 0.05, dtype=np.float32)
    assets = {
        "vv": {"href": _write_tif(tmp_path / "vv.tif", vv), "raster:bands": [{"nodata": -32768.0}]},
        "vh": {"href": _write_tif(tmp_path / "vh.tif", vh)},
    }

    cube = read_s1_rtc_cog(_product(assets, product_type="S1_RTC"), _grid())

    assert np.isnan(cube["gamma0_VV"].isel(time=0, y=0, x=0).item())
    assert cube["gamma0_VV"].isel(time=0, y=1, x=1).item() == pytest.approx(0.2, abs=1e-6)
    assert not bool(cube["valid_mask"].isel(time=0, y=0, x=0).item())
    assert bool(cube["valid_mask"].isel(time=0, y=1, x=1).item())

    db = read_s1_rtc_cog(_product(assets, product_type="S1_RTC"), _grid(), output_units="dB")
    assert db["gamma0_VH"].isel(time=0, y=1, x=1).item() == pytest.approx(10 * np.log10(0.05), abs=1e-4)


class _FakeSTACCatalog:
    def __init__(self, items):
        self.items = items
        self.calls: list[dict] = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.items


def _item(item_id: str, cloud: float | None) -> STACItem:
    properties = {"datetime": "2025-06-10T10:30:00Z", "platform": "sentinel-2a"}
    if cloud is not None:
        properties["eo:cloud_cover"] = cloud
    return STACItem(item_id, ("c",), properties["datetime"], None, {"B04": {"href": "x"}}, properties, {})


def test_stac_catalogs_map_items_and_filter_cloud_cover():
    s2 = _FakeSTACCatalog([_item("a", 12.0)])
    refs = Sentinel2STACCatalog(catalog=s2).search(AOI(13.2, 52.4, 13.6, 52.6), "2025-06-01", "2025-06-30", cloud_cover_max=20, limit=5)
    assert refs[0].cloud_cover == pytest.approx(12.0)
    assert refs[0].metadata["assets"]["B04"]["href"] == "x"
    assert s2.calls[0]["collections"] == ["sentinel-2-l2a"]
    assert s2.calls[0]["query"] == {"eo:cloud_cover": {"lte": 20}}

    s1 = _FakeSTACCatalog([_item("b", None)])
    refs = Sentinel1RTCSTACCatalog(catalog=s1).search(AOI(13.2, 52.4, 13.6, 52.6), "2025-06-01", "2025-06-30")
    assert refs[0].product_type == "S1_RTC"
    assert s1.calls[0]["collections"] == ["sentinel-1-rtc"]


def test_every_supported_sensor_and_provider_has_a_registered_implementation():
    assert set(adapters.SENSOR_ADAPTERS) == set(SUPPORTED_SENSORS)
    assert set(AUXILIARY_PROVIDER_FACTORIES) == set(AUXILIARY_PROVIDERS)


def test_request_round_trips_cloud_native_options_and_rejects_unknown_ones():
    request = AnalysisRequest(aoi=AOI(0, 0, 1, 1), start="2025-06-01", end="2025-06-02", sensors=("sentinel1", "sentinel2"), sentinel1_backend="pc_rtc", sentinel2_source="stac_cog")
    restored = AnalysisRequest.from_dict(request.to_dict())
    assert (restored.sentinel1_backend, restored.sentinel2_source) == ("pc_rtc", "stac_cog")
    with pytest.raises(ValueError, match="sentinel2_source"):
        AnalysisRequest(aoi=AOI(0, 0, 1, 1), start="2025-06-01", end="2025-06-02", sentinel2_source="ftp")


def test_execute_drives_registered_adapters_and_fuses_their_cubes(tmp_path, monkeypatch):
    times = np.array(["2025-06-11", "2025-06-10"], dtype="datetime64[ns]")  # deliberately unsorted

    def cube(name: str, value: float) -> xr.Dataset:
        return xr.Dataset(
            {name: (("time", "y", "x"), np.full((2, 2, 2), value))},
            coords={"time": times, "y": [100.0, 0.0], "x": [0.0, 100.0]},
            attrs={"crs": CRS, "grid_id": "g"},
        )

    class FakeAdapter:
        def __init__(self, name: str, variable: str, value: float):
            self.name, self.variable, self.value = name, variable, value
            self.acquired: list[ProductRef] = []

        def search(self, request, *, limit):
            return [_product({}, product_type=self.name)] * 2  # duplicate id must be deduplicated

        def acquire(self, references, context):
            self.acquired = references
            return cube(self.variable, self.value)

    thermal, optical = FakeAdapter("sentinel3", "lst", 30.0), FakeAdapter("sentinel2", "NDVI", 0.5)
    monkeypatch.setattr("citycube.workflow.runner.SENSOR_ADAPTERS", {"sentinel3": thermal, "sentinel2": optical})
    request = AnalysisRequest(aoi=AOI(0, 0, 1, 1), start="2025-06-10", end="2025-06-11", sensors=("sentinel3", "sentinel2"))

    result = AnalysisWorkflow(request).execute(tmp_path)

    assert len(thermal.acquired) == 1
    assert {"lst", "NDVI"}.issubset(result.cube.data_vars)
    assert list(result.cube.time.values) == sorted(times)


def test_analysis_grid_for_aoi_picks_utm_and_encloses_the_whole_aoi():
    from pyproj import Transformer

    aoi = AOI(13.20, 52.40, 13.55, 52.58)
    grid = AnalysisGrid.for_aoi(aoi, resolution_m=100)

    assert grid.crs == "EPSG:32633"
    to_utm = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    left, bottom, right, top = grid.bounds
    # Densified edges: the mid-point of the north edge bows outwards in UTM.
    for lon in np.linspace(aoi.west, aoi.east, 11):
        for lat in (aoi.south, aoi.north):
            x, y = to_utm.transform(lon, lat)
            assert left <= x <= right and bottom <= y <= top
    assert AnalysisGrid.for_aoi(AOI(-58.5, -34.7, -58.3, -34.5), resolution_m=100).crs == "EPSG:32721"


def test_cli_run_executes_and_saves_the_result(tmp_path, monkeypatch, capsys):
    pytest.importorskip("zarr")
    import json

    from citycube.cli import main
    from citycube.workflow.plan import build_plan
    from citycube.workflow.result import AnalysisResult

    request = AnalysisRequest(aoi=AOI(0, 0, 1, 1), start="2025-06-10", end="2025-06-11")
    request_path = request.save_json(tmp_path / "request.json")
    cube = xr.Dataset(
        {"lst": (("time", "y", "x"), np.ones((1, 2, 2), dtype=np.float32))},
        coords={"time": np.array(["2025-06-10"], dtype="datetime64[ns]"), "y": [1.0, 0.0], "x": [0.0, 1.0]},
        attrs={"crs": CRS},
    )
    calls = {}

    def fake_execute(self, output_dir, *, max_workers, gpt, progress):
        calls.update(output_dir=output_dir, max_workers=max_workers)
        return AnalysisResult(cube=cube, plan=build_plan(self.request), provenance={"sensors": ["sentinel3"]}, thermal_cube=cube)

    monkeypatch.setattr(AnalysisWorkflow, "execute", fake_execute)

    assert main(["run", str(request_path), str(tmp_path / "out"), "--max-workers", "2"]) == 0

    result_dir = tmp_path / "out" / "result"
    assert calls == {"output_dir": tmp_path / "out" / "work", "max_workers": 2}
    assert (result_dir / "cube.zarr").is_dir() and (result_dir / "thermal_cube.zarr").is_dir()
    assert AnalysisRequest.from_json(result_dir / "request.json").to_dict() == request.to_dict()
    assert json.loads((result_dir / "provenance.json").read_text()) == {"sensors": ["sentinel3"]}
    assert json.loads(capsys.readouterr().out)["variables"] == ["lst"]


def test_analysis_grid_round_trips_through_an_odc_geobox():
    pytest.importorskip("odc.geo")

    grid = AnalysisGrid.from_bounds((500000, 4000000, 500200, 4000100), crs=CRS, resolution_m=20)
    geobox = grid.to_geobox()

    assert geobox.shape == (grid.height, grid.width)
    restored = AnalysisGrid.from_geobox(geobox, grid_id=grid.grid_id)
    assert restored == grid
    np.testing.assert_allclose(restored.x, grid.x)


def test_remote_cog_reads_are_retried_after_a_transient_failure(tmp_path, monkeypatch):
    import rasterio
    from rasterio.errors import RasterioIOError

    from citycube.sensors import cog

    local = _write_tif(tmp_path / "band.tif", np.full((4, 4), 7, dtype=np.uint16), nodata=0)
    real_open = rasterio.open
    calls = {"count": 0}

    def flaky_open(href, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RasterioIOError("Read failed. See previous exception for details.")
        return real_open(local, *args, **kwargs)

    monkeypatch.setattr(rasterio, "open", flaky_open)
    values = cog.read_cog_to_grid("https://example.blob.core.windows.net/band.tif", _grid(), retry_delay_s=0)
    assert calls["count"] == 2
    np.testing.assert_allclose(values, 7.0)
