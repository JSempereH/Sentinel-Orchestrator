from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from citycube import block_aggregate, sharpen_landsat

CRS = "EPSG:32633"
N = 180  # 30 m cells: 5.4 km
X = np.arange(N) * 30.0 + 15
Y = X[::-1].copy()
LANDSAT_TIME = np.array(["2026-08-15T10:00"], dtype="datetime64[ns]")


def _smooth_noise(rng: np.random.Generator, size: int, passes: int = 3) -> np.ndarray:
    field = rng.random((size, size))
    for _ in range(passes):  # cheap blur: average with the four neighbours
        field = (field + np.roll(field, 1, 0) + np.roll(field, -1, 0) + np.roll(field, 1, 1) + np.roll(field, -1, 1)) / 5
    return (field - field.min()) / (field.max() - field.min())


def _scene(s2_offset_days: float = 1.0):
    rng = np.random.default_rng(4)
    ndvi = _smooth_noise(rng, N)
    xx = np.broadcast_to(X[None, :], (N, N))
    truth = 40 - 15 * ndvi + 2 * np.sin(xx / 1500)
    # Collection 2 delivers 30 m cells resampled from ~100 m: here 90 m block means repeated.
    product = np.kron(truth.reshape(N // 3, 3, N // 3, 3).mean(axis=(1, 3)), np.ones((3, 3)))
    landsat = xr.Dataset({"landsat_lst": (("time", "y", "x"), product[None].astype("float32"), {"units": "degC"})},
                         coords={"time": LANDSAT_TIME, "y": Y, "x": X}, attrs={"crs": CRS})
    s2_time = LANDSAT_TIME + np.timedelta64(int(s2_offset_days * 24), "h")
    sentinel2 = xr.Dataset({"NDVI": (("time", "y", "x"), ndvi[None]), "NDBI": (("time", "y", "x"), (0.3 - ndvi)[None])},
                           coords={"time": s2_time, "y": Y, "x": X}, attrs={"crs": CRS})
    return landsat, sentinel2, truth, product


def test_block_aggregate_means_blocks_and_needs_two_thirds_coverage():
    values = np.arange(36, dtype=float).reshape(1, 6, 6)
    values[0, 0, :2] = np.nan  # 2 of 9 cells missing in the first block: still observed
    values[0, 3:, 3:5] = np.nan  # 6 of 9 missing in the last block: not observed
    data = xr.Dataset({"v": (("time", "y", "x"), values, {"units": "K"})},
                      coords={"time": LANDSAT_TIME, "y": Y[:6], "x": X[:6]}, attrs={"crs": CRS})
    out = block_aggregate(data, 3)
    assert out.sizes == {"time": 1, "y": 2, "x": 2}
    assert out["v"].values[0, 0, 0] == pytest.approx(np.nanmean(values[0, :3, :3]))
    assert np.isnan(out["v"].values[0, 1, 1])
    assert out["v"].attrs["units"] == "K" and out.attrs["crs"] == CRS
    assert out.x.values[0] == pytest.approx(X[1])  # block centre


def test_sharpening_recovers_detail_lost_by_resampling_and_conserves_blocks():
    landsat, sentinel2, truth, product = _scene()
    out = sharpen_landsat(landsat, sentinel2, model="linear", min_samples=20)
    sharpened = out["lst_downscaled"].isel(time=0).values

    product_rmse = np.sqrt(np.mean((product - truth) ** 2))
    sharpened_rmse = np.sqrt(np.nanmean((sharpened - truth) ** 2))
    assert sharpened_rmse < 0.5 * product_rmse
    blocks = sharpened.reshape(N // 3, 3, N // 3, 3).mean(axis=(1, 3))
    np.testing.assert_allclose(blocks, truth.reshape(N // 3, 3, N // 3, 3).mean(axis=(1, 3)), atol=1e-4)
    assert out.attrs["downscaling_source"] == "landsat"
    assert out.attrs["downscaling_coarse_resolution_m"] == pytest.approx(90.0)
    assert out.attrs["downscaling_correction"] == "atpk"


def test_landsat_scene_without_a_close_sentinel2_image_is_skipped():
    landsat, sentinel2, _, _ = _scene(s2_offset_days=6)
    with pytest.raises(ValueError, match="No scene could be downscaled"):
        sharpen_landsat(landsat, sentinel2, model="linear", min_samples=20)
    out = sharpen_landsat(landsat, sentinel2, model="linear", min_samples=20, max_gap=np.timedelta64(7, "D"))
    assert out.sizes["time"] == 1


def test_sharpening_needs_matching_grids():
    landsat, sentinel2, _, _ = _scene()
    with pytest.raises(ValueError, match="same grid"):
        sharpen_landsat(landsat, sentinel2.isel(x=slice(0, -3)))


def test_tiles_of_one_landsat_pass_become_one_map():
    from citycube.workflow.adapters import LANDSAT_PASS_TOLERANCE, mosaic_temporal_tiles

    def tile(seconds, rows):
        values = np.full((1, 6, 6), np.nan)
        values[0, rows] = 30.0 + seconds
        valid = np.isfinite(values)
        time = np.array([np.datetime64("2026-08-15T09:56:00", "ns") + np.timedelta64(seconds, "s")])
        return xr.Dataset({"landsat_lst": (("time", "y", "x"), values), "valid_mask": (("time", "y", "x"), valid)},
                          coords={"time": time, "y": Y[:6], "x": X[:6]}, attrs={"crs": CRS})

    other_pass = tile(16 * 86400, slice(0, 6))
    mosaics = mosaic_temporal_tiles([tile(24, slice(3, 6)), tile(0, slice(0, 3)), other_pass], tolerance=LANDSAT_PASS_TOLERANCE)
    assert len(mosaics) == 2
    first = mosaics[0]
    assert first.time.values[0] == np.datetime64("2026-08-15T09:56:00", "ns")
    assert np.isfinite(first["landsat_lst"].values).all() and first["valid_mask"].values.all()
