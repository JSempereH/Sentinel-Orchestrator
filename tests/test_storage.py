"""AOI subsets, raw-data retention and Sentinel-5P product handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from citycube import AOI, AnalysisRequest, ProductRef, grid_s5p, read_s5p_l2
from citycube.sensors.sentinel5p import Sentinel5PReadConfig, deduplicate_orbits
from citycube.workflow.adapters import AcquisitionContext


def _ref(name: str, product_id: str = "id-1") -> ProductRef:
    return ProductRef(product_id, name, "t", None, None, None, True, "https://example/odata/v1/Products(x)/$value", {})


def _context(tmp_path: Path, retention: str) -> AcquisitionContext:
    request = AnalysisRequest(aoi=AOI(13.2, 52.4, 13.6, 52.6), start="2025-06-10", end="2025-06-11", raw_retention=retention)
    return AcquisitionContext(request=request, output_dir=tmp_path, downloader_factory=lambda: None)  # type: ignore[arg-type,return-value]


@pytest.mark.parametrize("retention, raw_survives", [("aoi_subset", False), ("keep", True)])
def test_subset_is_stored_reused_and_raw_data_follows_the_retention_policy(tmp_path, retention, raw_survives):
    context = _context(tmp_path, retention)
    raw_dir = tmp_path / "downloads" / "PRODUCT.SEN3"
    raw_dir.mkdir(parents=True)
    (raw_dir / "LST_in.nc").write_bytes(b"x" * 1000)
    calls = []

    def build():
        calls.append(1)
        dataset = xr.Dataset({"lst": (("y", "x"), np.ones((2, 2)))}, attrs={"flag": True, "nothing": None, "nested": {"a": 1}})
        return dataset, [raw_dir]

    first = context.subset("sentinel3", _ref("PRODUCT.SEN3"), build)
    second = context.subset("sentinel3", _ref("PRODUCT.SEN3"), build)

    assert len(calls) == 1  # the second call re-used the stored subset
    xr.testing.assert_equal(first["lst"], second["lst"])
    assert second.attrs["source_product_id"] == "id-1"
    assert raw_dir.exists() is raw_survives
    assert len(list((tmp_path / "subsets" / "sentinel3").glob("*.nc"))) == 1


def test_sentinel5p_keeps_one_product_per_orbit_preferring_offline_processing():
    products = [
        _ref("S5P_NRTI_L2__NO2____20260815T114649_20260815T115149_45798_03_020901_20260815T122509.nc", "nrti"),
        _ref("S5P_OFFL_L2__NO2____20260815T104614_20260815T122744_45798_03_020901_20260817T031428.nc", "offl"),
        _ref("S5P_NRTI_L2__NO2____20260817T110649_20260817T111149_45826_03_020901_20260817T114929.nc", "other-orbit"),
    ]
    kept = {product.product_id for product in deduplicate_orbits(products)}
    assert kept == {"offl", "other-orbit"}


def test_sentinel5p_aoi_crop_reads_the_requested_gas_and_is_storable(tmp_path):
    path = tmp_path / "S5P_OFFL_L2__CO_____20250610T120000.nc"
    lat = np.array([[[20.01, 20.01, 40.0], [20.11, 20.11, 40.0]]])
    lon = np.array([[[-103.31, -103.21, 10.0], [-103.31, -103.21, 10.0]]])
    xr.Dataset(
        {
            "carbonmonoxide_total_column": (("time", "scanline", "ground_pixel"), np.ones((1, 2, 3))),
            "qa_value": (("time", "scanline", "ground_pixel"), np.ones((1, 2, 3))),
            "latitude": (("time", "scanline", "ground_pixel"), lat),
            "longitude": (("time", "scanline", "ground_pixel"), lon),
        },
        coords={"time": [0], "scanline": [0, 1], "ground_pixel": [0, 1, 2]},
    ).to_netcdf(path, group="PRODUCT")
    aoi = AOI(-103.4, 19.9, -103.1, 20.2)

    swath = read_s5p_l2(path, config=Sentinel5PReadConfig(gas="CO"), aoi=aoi)

    assert swath.sizes["observation"] == 4  # the far-away column is dropped
    swath.to_netcdf(tmp_path / "subset.nc")  # no MultiIndex left
    with xr.open_dataset(tmp_path / "subset.nc") as stored:
        assert grid_s5p(stored.load(), resolution_deg=0.1, aoi=aoi)["CO"].notnull().sum().item() == 4


def test_sentinel3_adapter_downloads_partially_grids_and_reuses_the_subset(tmp_path):
    import shutil

    from citycube import AnalysisGrid
    from citycube.workflow.adapters import Sentinel3Adapter
    from test_sentinel3_reader import NAME, _write_product

    source = _write_product(tmp_path / "source", rows=8, columns=32)

    class FakeDownloader:
        def __init__(self):
            self.requested: list[list[str]] = []

        def download_files(self, reference, names, output_dir):
            names = list(names)
            self.requested.append(names)
            root = Path(output_dir) / reference.name
            root.mkdir(parents=True, exist_ok=True)
            for name in names:
                if (source / name).exists():
                    shutil.copy(source / name, root / name)
            return root

    downloader = FakeDownloader()
    aoi = AOI(13.1, 52.3, 13.7, 52.6)
    grid = AnalysisGrid.for_aoi(aoi, resolution_m=1000)
    request = AnalysisRequest(aoi=aoi, start="2025-06-10", end="2025-06-10", thermal_grid=grid, grid=grid)
    context = AcquisitionContext(request=request, output_dir=tmp_path / "run", downloader_factory=lambda: downloader)  # type: ignore[arg-type,return-value]
    reference = _ref(NAME)

    cube = Sentinel3Adapter().acquire([reference], context)

    assert "met_tx.nc" not in downloader.requested[0]
    assert not (tmp_path / "run" / "downloads" / "sentinel3" / NAME).exists()  # raw files deleted
    assert cube is not None and (cube.sizes["y"], cube.sizes["x"]) == (grid.height, grid.width)
    assert np.isfinite(cube["lst"].values).any()

    again = Sentinel3Adapter().acquire([reference], context)
    assert len(downloader.requested) == 1  # served from the stored subset
    xr.testing.assert_allclose(again["lst"], cube["lst"])
