"""Sentinel-3 area-weighted gridding: orientation, equivalence and speed."""

from __future__ import annotations

import time

import numpy as np
import pytest
import xarray as xr

from sentinel_analysis import AnalysisGrid
from sentinel_analysis.sensors.sentinel3.georeference import grid_l2_lst

pytest.importorskip("shapely")
pyproj = pytest.importorskip("pyproj")

CRS = "EPSG:32613"


def _swath(n: int, *, spacing: float, origin=(500010.0, 2200390.0), angle_deg: float = 0.0, n_times: int = 1, seed: int = 0, nan_fraction: float = 0.1) -> xr.Dataset:
    """A synthetic swath whose LST is its northing in km (north is warmer)."""

    rng = np.random.default_rng(seed)
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    theta = np.deg2rad(angle_deg)
    x = origin[0] + spacing * (j * np.cos(theta) + i * np.sin(theta))
    y = origin[1] - spacing * (i * np.cos(theta) - j * np.sin(theta))
    lon, lat = pyproj.Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(x, y)
    lst = np.stack([y / 1000.0 + 0.1 * t for t in range(n_times)])
    lst[:, rng.random((n, n)) < nan_fraction] = np.nan
    cloud = rng.integers(0, 4, size=(n_times, n, n)).astype(np.float32)
    cloud[:, rng.random((n, n)) < 0.1] = np.nan
    return xr.Dataset(
        {
            "lst": (("time", "y", "x"), lst),
            "lst_uncertainty": (("time", "y", "x"), rng.random((n_times, n, n))),
            "cloud_flags": (("time", "y", "x"), cloud),
            "exception_flags": (("time", "y", "x"), rng.integers(0, 8, size=(n_times, n, n)).astype(np.int16)),
            "latitude": (("y", "x"), lat),
            "longitude": (("y", "x"), lon),
        },
        coords={"time": np.array([f"2025-06-1{t}" for t in range(n_times)], dtype="datetime64[ns]"), "y": np.arange(n), "x": np.arange(n)},
    )


def test_area_gridding_keeps_north_up_orientation():
    grid = AnalysisGrid.from_bounds((500000, 2200000, 500400, 2200400), crs=CRS, resolution_m=100)
    gridded = grid_l2_lst(_swath(8, spacing=50.0, origin=(500025.0, 2200375.0), nan_fraction=0.0), grid)
    column = gridded["lst"].isel(time=0, x=0).values
    # grid.y runs north to south, so a north-is-warmer field must decrease.
    assert np.all(np.diff(column) < 0)
    np.testing.assert_allclose(column, gridded.y.values / 1000.0, atol=0.03)


def test_area_gridding_matches_the_original_per_pixel_implementation():
    from _reference_s3_area import reference_grid_area_l2_lst

    grid = AnalysisGrid.from_bounds((500000, 2200000, 500400, 2200300), crs=CRS, resolution_m=100)
    source = _swath(12, spacing=37.0, angle_deg=13.0, seed=3)

    new = grid_l2_lst(source, grid)
    # The reference numbered target rows from the bottom; flip it to compare.
    old = reference_grid_area_l2_lst(source, grid).isel(y=slice(None, None, -1)).assign_coords(y=new.y)

    assert set(new.data_vars) == set(old.data_vars)
    for name in new.data_vars:
        np.testing.assert_allclose(np.asarray(new[name].values, dtype=float), np.asarray(old[name].values, dtype=float), rtol=1e-5, equal_nan=True, err_msg=name)


def test_area_gridding_only_builds_footprints_near_the_grid():
    # A 600x600 swath (360k pixels) over a 4x3 km grid: the old per-pixel
    # version built a shapely polygon for every swath pixel.
    grid = AnalysisGrid.from_bounds((500000, 2200000, 504000, 2203000), crs=CRS, resolution_m=1000)
    source = _swath(600, spacing=1000.0, origin=(200000.0, 2500000.0), seed=5)
    started = time.perf_counter()
    gridded = grid_l2_lst(source, grid)
    assert time.perf_counter() - started < 10
    assert np.isfinite(gridded["lst"].values).mean() > 0.5


def test_reader_masks_clouds_with_the_bayesian_flags_when_present(tmp_path):
    from sentinel_analysis import apply_quality_mask, read_l2_lst

    product = tmp_path / "S3A_SL_2_LST____20250610T090000_20250610T090300_0000_000_000____LN2_D_NT_005.SEN3"
    product.mkdir()
    xr.Dataset({"LST": (("rows", "columns"), np.array([[300.0, 290.0, 301.0]]))}, attrs={"units": "K"}).to_netcdf(product / "LST_in.nc")
    meanings = "visible 1.37_threshold gross_cloud"
    xr.Dataset(
        {
            # cloud_in: only the "visible" test fired on pixel 1, a bit the
            # threshold-test subset does not treat as cloud.
            "cloud_in": (("rows", "columns"), np.array([[0, 1, 0]], dtype=np.uint16), {"flag_masks": np.array([1, 2, 4], dtype=np.uint16), "flag_meanings": meanings}),
            "bayes_in": (("rows", "columns"), np.array([[0, 2, 1]], dtype=np.uint8), {"flag_masks": np.array([1, 2, 4, 8], dtype=np.uint8), "flag_meanings": "single_low single_moderate dual_low dual_moderate"}),
        }
    ).to_netcdf(product / "flags_in.nc")

    dataset = read_l2_lst(product)
    assert dataset["cloud_mask"].isel(time=0).values.tolist() == [[False, True, False]]
    masked = apply_quality_mask(dataset)
    assert np.isnan(masked["lst"].isel(time=0).values[0, 1])
    assert np.isfinite(masked["lst"].isel(time=0).values[0, [0, 2]]).all()

    # The new flag variable must survive gridding and the metadata contract
    # that AnalysisWorkflow.run() enforces.
    from sentinel_analysis.metadata import validate_variable_contract

    lat, lon = pyproj.Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(np.array([[500050.0, 500150.0, 500250.0]]), np.array([[2200050.0] * 3]))
    located = masked.assign(latitude=(("y", "x"), lat), longitude=(("y", "x"), lon))
    validate_variable_contract(grid_l2_lst(located, AnalysisGrid.from_bounds((500000, 2200000, 500300, 2200100), crs=CRS, resolution_m=100)))


def test_area_gridding_agrees_with_an_independent_resampler():
    """Cross-check against pyresample (an independent swath-to-grid
    implementation): this alone would have caught the north-south flip."""

    pytest.importorskip("pyresample")
    from pyresample.geometry import AreaDefinition, SwathDefinition
    from pyresample.kd_tree import resample_nearest

    grid = AnalysisGrid.from_bounds((500000, 2200000, 501200, 2200800), crs=CRS, resolution_m=100)
    # Fine (25 m), rotated swath with a field varying in both x and y.
    n = 64
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    theta = np.deg2rad(20.0)
    x = 499900 + 25.0 * (j * np.cos(theta) + i * np.sin(theta))
    y = 2200900 - 25.0 * (i * np.cos(theta) - j * np.sin(theta))
    lon, lat = pyproj.Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(x, y)
    field = (y - 2200000) / 100.0 + 0.3 * (x - 500000) / 100.0
    source = xr.Dataset(
        {"lst": (("time", "y", "x"), field[None]), "latitude": (("y", "x"), lat), "longitude": (("y", "x"), lon)},
        coords={"time": np.array(["2025-06-10"], dtype="datetime64[ns]"), "y": np.arange(n), "x": np.arange(n)},
    )

    ours = grid_l2_lst(source, grid)["lst"].isel(time=0).values
    area = AreaDefinition("grid", "grid", "grid", CRS, grid.width, grid.height, grid.bounds)
    theirs = resample_nearest(SwathDefinition(lons=lon, lats=lat), field, area, radius_of_influence=50, fill_value=np.nan)

    both = np.isfinite(ours) & np.isfinite(theirs)
    assert both.mean() > 0.8
    # Area averaging vs one nearest sample: same field, same orientation.
    assert np.corrcoef(ours[both], theirs[both])[0, 1] > 0.99
    assert np.abs(ours[both] - theirs[both]).max() < 0.5


def test_area_gridding_of_an_empty_aoi_crop_is_all_missing_not_a_crash():
    from sentinel_analysis import apply_quality_mask

    grid = AnalysisGrid.from_bounds((500000, 2200000, 500400, 2200400), crs=CRS, resolution_m=100)
    empty = _swath(8, spacing=50.0, nan_fraction=0.0).isel(y=slice(0, 0), x=slice(0, 0))
    gridded = grid_l2_lst(apply_quality_mask(empty), grid)
    assert gridded["lst"].shape == (1, grid.height, grid.width)
    assert gridded["lst"].isnull().all()
    assert not gridded["valid_mask"].any()
