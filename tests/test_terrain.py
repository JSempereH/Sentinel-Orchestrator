"""Terrain predictors: DEM loading, slope/aspect, solar position, illumination."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from sentinel_analysis import AnalysisGrid
from sentinel_analysis.sensors.terrain import load_dem, slope_aspect, solar_position, terrain_predictors
from sentinel_analysis.stac import STACItem

pytest.importorskip("pyproj")
CRS = "EPSG:32633"


def _plane(gradient_east: float, gradient_north: float) -> xr.DataArray:
    x = 400050.0 + 100.0 * np.arange(20)
    y = 5800950.0 - 100.0 * np.arange(10)  # north-up: y decreasing
    xx, yy = np.meshgrid(x, y)
    return xr.DataArray(gradient_east * (xx - x[0]) + gradient_north * (yy - y[-1]), dims=("y", "x"), coords={"y": y, "x": x})


@pytest.mark.parametrize("east, north, aspect", [(0.1, 0.0, 270.0), (-0.1, 0.0, 90.0), (0.0, 0.1, 180.0), (0.0, -0.1, 0.0)])
def test_slope_and_aspect_of_tilted_planes(east, north, aspect):
    slope, facing = slope_aspect(_plane(east, north))
    np.testing.assert_allclose(slope.values, np.degrees(np.arctan(0.1)), rtol=1e-5)
    assert np.allclose((facing.values - aspect + 180) % 360 - 180, 0, atol=1e-4)


def test_solar_position_matches_textbook_geometry():
    zenith, _ = solar_position(np.datetime64("2025-03-20T12:07"), np.array([0.0]), np.array([0.0]))
    assert zenith[0] < 1.5  # equinox noon at the equator: sun overhead
    # Berlin (52.52 N, 13.4 E), summer solstice, local solar noon (~11:05 UTC).
    zenith, azimuth = solar_position(np.datetime64("2025-06-21T11:05"), np.array([52.52]), np.array([13.4]))
    assert zenith[0] == pytest.approx(52.52 - 23.44, abs=1.0)
    assert azimuth[0] == pytest.approx(180.0, abs=5.0)  # due south
    _, morning = solar_position(np.datetime64("2025-06-21T05:00"), np.array([52.52]), np.array([13.4]))
    assert 40 < morning[0] < 90  # north-east to east


def test_illumination_is_cos_zenith_on_flat_ground_and_peaks_on_slopes_facing_the_sun():
    time = np.datetime64("2025-06-21T11:05")
    flat = _plane(0.0, 0.0)
    slope, aspect = slope_aspect(flat)
    static = xr.Dataset({"elevation": flat, "slope": slope, "aspect": aspect})
    result = terrain_predictors(static, [time], crs=CRS)
    zenith = 52.52 - 23.44
    np.testing.assert_allclose(result["cos_incidence"].values, np.cos(np.radians(zenith)), atol=0.03)
    assert result["cos_incidence"].dims == ("time", "y", "x")

    # A south-facing slope tilted by the solar zenith faces the noon sun head-on.
    south_facing = _plane(0.0, np.tan(np.radians(zenith)))  # rises to the north, so it faces south
    slope, aspect = slope_aspect(south_facing)
    static = xr.Dataset({"elevation": south_facing, "slope": slope, "aspect": aspect})
    assert np.median(terrain_predictors(static, [time], crs=CRS)["cos_incidence"].values) == pytest.approx(1.0, abs=0.02)


def test_load_dem_mosaics_tiles_onto_the_grid(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from affine import Affine

    grid = AnalysisGrid.from_bounds((400000, 5800000, 402000, 5801000), crs=CRS, resolution_m=100)
    path = tmp_path / "dem.tif"
    with rasterio.open(path, "w", driver="GTiff", width=40, height=20, count=1, dtype="float32", crs=CRS, transform=Affine(50, 0, 400000, 0, -50, 5801000)) as dst:
        dst.write(np.full((20, 40), 35.0, dtype=np.float32), 1)

    class Catalog:
        def search(self, **kwargs):
            self.kwargs = kwargs
            return [STACItem("tile", ("cop-dem-glo-30",), None, None, {"data": {"href": str(path)}}, {}, {})]

    catalog = Catalog()
    elevation = load_dem(grid, catalog=catalog)
    assert catalog.kwargs["collections"] == ["cop-dem-glo-30"]
    west, south, east, north = catalog.kwargs["bbox"]
    assert 12 < west < east < 14 and 52 < south < north < 53  # lon/lat bounds of the UTM grid
    assert elevation.shape == (grid.height, grid.width)
    np.testing.assert_allclose(elevation.values, 35.0)


def test_workflow_run_adds_aggregated_terrain_and_illumination_predictors():
    from sentinel_analysis import AOI, AnalysisRequest, AnalysisWorkflow

    times = np.array(["2025-06-21T10:00", "2025-06-22T10:00"], dtype="datetime64[ns]")
    coarse = xr.Dataset(
        {"lst": (("time", "y", "x"), np.full((2, 2, 2), 30.0))},
        coords={"time": times, "y": [5801500.0, 5800500.0], "x": [400500.0, 401500.0]},
        attrs={"crs": CRS, "grid_id": "coarse"},
    )
    fx = 400050.0 + 100.0 * np.arange(20)
    fy = 5801950.0 - 100.0 * np.arange(20)
    fine = xr.Dataset(
        {"NDVI": (("time", "y", "x"), np.full((2, 20, 20), 0.4))},
        coords={"time": times, "y": fy, "x": fx},
        attrs={"crs": CRS, "grid_id": "fine"},
    )
    elevation = xr.DataArray(np.where(fx[None, :] < 401000, 100.0, 300.0) * np.ones((20, 1)), dims=("y", "x"), coords={"y": fy, "x": fx})
    slope, aspect = slope_aspect(elevation)
    terrain = xr.Dataset({"elevation": elevation, "slope": slope, "aspect": aspect}, attrs={"crs": CRS})
    request = AnalysisRequest(aoi=AOI(13.5, 52.3, 13.6, 52.4), start="2025-06-21", end="2025-06-22", sensors=("sentinel3", "sentinel2"))

    result = AnalysisWorkflow(request).run({"sentinel3": coarse, "sentinel2": fine}, terrain=terrain)

    np.testing.assert_allclose(result.cube["elevation"].isel(time=0).values, [[100.0, 300.0], [100.0, 300.0]])
    assert result.cube["cos_incidence"].dims == ("time", "y", "x") and np.isfinite(result.cube["cos_incidence"].values).all()
    assert result.predictor_cube["elevation"].sizes["time"] == 2
    assert result.terrain is terrain
