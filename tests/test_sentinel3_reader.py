"""Sentinel-3 LST reader: selective extraction, AOI crop, geometry, quality."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from sentinel_analysis import AOI, LSTConfidenceFlag, QualityPolicy, apply_quality_mask, read_l2_lst

NAME = "S3A_SL_2_LST____20250610T090000_20250610T090300_20250611T000000_0179_000_000_0000_PS1_O_NT_005.SEN3"


def _write_product(root: Path, *, rows: int = 4, columns: int = 32) -> Path:
    product = root / NAME
    product.mkdir(parents=True)
    lat = np.linspace(52.6, 52.3, rows)[:, None] * np.ones((1, columns))
    lon = np.linspace(13.0, 13.8, columns)[None, :] * np.ones((rows, 1))
    # Real SLSTR files store "resolution" as a string, e.g. "[ 1000 1000 ]".
    image_attrs = {"track_offset": 16, "resolution": "[ 1000 1000 ]"}
    xr.Dataset({"LST": (("rows", "columns"), np.full((rows, columns), 300.0, dtype=np.float32), {"units": "K"})}).to_netcdf(product / "LST_in.nc")
    xr.Dataset({"latitude_in": (("rows", "columns"), lat), "longitude_in": (("rows", "columns"), lon)}, attrs=image_attrs).to_netcdf(product / "geodetic_in.nc")
    confidence = np.zeros((rows, columns), dtype=np.uint16)
    confidence[0, 5] = int(LSTConfidenceFlag.DUPLICATE)
    confidence[0, 6] = int(LSTConfidenceFlag.SNOW)  # not rejected by default
    flag_meanings = "coastline ocean tidal land inland_water unfilled spare spare cosmetic duplicate day twilight sun_glint snow summary_cloud summary_pointing"
    xr.Dataset({
        "confidence_in": (("rows", "columns"), confidence, {"flag_masks": (1 << np.arange(16)).astype(np.uint16), "flag_meanings": flag_meanings}),
        "bayes_in": (("rows", "columns"), np.zeros((rows, columns), dtype=np.uint8), {"flag_masks": np.array([1, 2, 4, 8], dtype=np.uint8), "flag_meanings": "single_low single_moderate dual_low dual_moderate"}),
    }).to_netcdf(product / "flags_in.nc")
    # Tie points every 16 km; image column i sits at tie column 64 + (i - 16) / 16,
    # so the 32 image columns span tie columns 63..65 around nadir (column 64).
    tie_columns = np.arange(130)
    zenith = np.abs(tie_columns - 64) * 60.0  # steep on purpose: 0 deg at nadir, 60 deg one tie point away
    xr.Dataset(
        {"sat_zenith_tn": (("rows", "columns"), np.tile(zenith, (rows, 1)))},
        attrs={"track_offset": 64, "resolution": "[ 16000 1000 ]"},
    ).to_netcdf(product / "geometry_tn.nc")
    xr.Dataset({"temperature": (("x",), np.zeros(10))}).to_netcdf(product / "met_tx.nc")
    return product


def _zip(product: Path, target: Path) -> Path:
    with zipfile.ZipFile(target, "w") as archive:
        for file in product.iterdir():
            archive.write(file, f"{product.name}/{file.name}")
    return target


def test_zip_read_extracts_only_needed_files_and_leaves_no_temporary_directory(tmp_path, monkeypatch):
    product = _write_product(tmp_path / "src")
    archive = _zip(product, tmp_path / "product.zip")
    temp_root = tmp_path / "tmp"
    temp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))
    extracted: list[str] = []
    original = zipfile.ZipFile.extract

    def recording_extract(self, member, path=None, pwd=None):
        extracted.append(Path(getattr(member, "filename", member)).name)
        return original(self, member, path, pwd)

    monkeypatch.setattr(zipfile.ZipFile, "extract", recording_extract)

    dataset = read_l2_lst(archive)

    assert "met_tx.nc" not in extracted
    assert list(temp_root.iterdir()) == []  # the temporary extraction is gone
    assert float(dataset["lst"].isel(time=0).values[0, 0]) == pytest.approx(300.0)  # data loaded into memory


def test_view_zenith_is_interpolated_from_tie_points():
    with tempfile.TemporaryDirectory() as directory:
        dataset = read_l2_lst(_write_product(Path(directory)))
        zenith = dataset["sat_zenith"].squeeze().values[0]
    expected = np.abs((np.arange(32) - 16) / 16) * 60.0
    np.testing.assert_allclose(zenith, expected, rtol=1e-5)


def test_aoi_crop_keeps_only_the_covering_window():
    with tempfile.TemporaryDirectory() as directory:
        full = read_l2_lst(_write_product(Path(directory)))
        cropped = read_l2_lst(Path(directory) / NAME, aoi=AOI(13.30, 52.40, 13.40, 52.50), aoi_margin_deg=0.0)
    assert cropped.sizes["x"] < full.sizes["x"]
    lon = cropped["longitude"].squeeze().values
    assert lon.min() >= 13.30 - 1e-9 and lon.max() <= 13.40 + 1e-9


def test_quality_rejects_duplicates_wide_views_and_buffers_clouds():
    with tempfile.TemporaryDirectory() as directory:
        dataset = read_l2_lst(_write_product(Path(directory)))
    valid = apply_quality_mask(dataset)["valid_mask"].squeeze().values
    zenith = dataset["sat_zenith"].squeeze().values
    assert not valid[0, 5]  # duplicate
    assert valid[0, 6]  # snow is kept by default
    assert not valid[zenith > 45].any() and valid[(zenith <= 45)].sum() > 0

    cloudy = dataset.assign(cloud_mask=dataset["lst"].copy(data=np.zeros(dataset["lst"].shape, dtype=bool)))
    cloudy["cloud_mask"].values[0, 2, 2] = True
    buffered = apply_quality_mask(cloudy, QualityPolicy(max_view_zenith=None, cloud_buffer_pixels=1))["valid_mask"].squeeze().values
    assert not buffered[1:4, 1:4].any() and buffered[0, 0]
    assert apply_quality_mask(cloudy, QualityPolicy.permissive())["valid_mask"].squeeze().values[0, 5]


def test_aoi_crop_owns_its_memory_instead_of_viewing_the_full_swath(tmp_path):
    archive = _zip(_write_product(tmp_path / "src"), tmp_path / "product.zip")
    cropped = read_l2_lst(archive, aoi=AOI(13.30, 52.40, 13.40, 52.50), aoi_margin_deg=0.0)
    # A NumPy view of the full swath would keep the whole array alive
    # (~80 MB per real product before this was fixed).
    for name, variable in cropped.data_vars.items():
        assert variable.values.base is None or variable.values.base.size == variable.values.size, name
