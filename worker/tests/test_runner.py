"""
Validates the request-building glue in app.runner: turning a simple
{aoi, resolution_m} payload (what this worker's own frontend sends for
an arbitrary drawn AOI) into a correct AnalysisRequest with a real projected
AnalysisGrid. This is pure math (UTM projection via pyproj) - no network,
no credentials - so it is checked directly against known reference values
rather than mocked.
"""

import pytest

from app.runner import build_request
from citycube.config import AOI
from citycube.cube import projected_bounds as _projected_bounds
from citycube.cube import utm_crs as _utm_epsg


def test_utm_epsg_northern_hemisphere_berlin():
    # Berlin (~13.4E, 52.5N) is UTM zone 33N.
    assert _utm_epsg(13.4, 52.5) == "EPSG:32633"


def test_utm_epsg_southern_hemisphere_buenos_aires():
    # Buenos Aires (~-58.4E, -34.6N) is UTM zone 21S.
    assert _utm_epsg(-58.4, -34.6) == "EPSG:32721"


def test_utm_epsg_zone_boundaries_wrap_correctly():
    # Longitude 180 must not overflow past zone 60.
    assert _utm_epsg(179.9, 10.0) == "EPSG:32660"
    assert _utm_epsg(-179.9, 10.0) == "EPSG:32601"


def test_projected_bounds_are_larger_than_a_degenerate_point():
    aoi = AOI(west=13.3, south=52.4, east=13.5, north=52.6)
    west, south, east, north = _projected_bounds(aoi, "EPSG:32633")
    assert west < east
    assert south < north
    # A ~0.2deg x 0.2deg box near Berlin should span roughly 10-20 km in UTM metres.
    assert 5_000 < (east - west) < 30_000
    assert 5_000 < (north - south) < 30_000


def test_build_request_from_simple_aoi_payload():
    payload = {
        "aoi": {"west": 13.3, "south": 52.4, "east": 13.5, "north": 52.6},
        "start": "2024-06-01",
        "end": "2024-06-05",
        "sensors": ["sentinel3"],
        "resolution_m": 200,
    }
    request = build_request(payload)

    assert request.aoi == AOI(west=13.3, south=52.4, east=13.5, north=52.6)
    assert request.sensors == ("sentinel3",)
    assert request.grid is not None and request.predictor_grid is not None
    assert request.grid.crs == "EPSG:32633"
    assert request.grid.resolution_m == 200
    assert request.grid.width > 0 and request.grid.height > 0
    # Thermal grid defaults to max(1000, resolution_m) when not given explicitly.
    assert request.thermal_grid is not None
    assert request.thermal_grid.resolution_m == 1000


def test_build_request_keeps_a_drawn_polygon():
    triangle = {"type": "Polygon", "coordinates": [[[13.3, 52.4], [13.5, 52.4], [13.4, 52.6], [13.3, 52.4]]]}
    payload = {
        "aoi": {"west": 13.3, "south": 52.4, "east": 13.5, "north": 52.6, "geometry": triangle},
        "start": "2024-06-01",
        "end": "2024-06-05",
        "sensors": ["sentinel3"],
    }
    request = build_request(payload)
    assert request.aoi.has_geometry
    assert (request.aoi.west, request.aoi.north) == pytest.approx((13.3, 52.6))
    assert request.to_dict()["aoi"]["geometry"]["type"] == "Polygon"


def test_build_request_rejects_an_unusable_polygon():
    line = {"type": "LineString", "coordinates": [[13.3, 52.4], [13.5, 52.6]]}
    with pytest.raises(ValueError):
        build_request({"aoi": {"west": 13.3, "south": 52.4, "east": 13.5, "north": 52.6, "geometry": line},
                       "start": "2024-06-01", "end": "2024-06-05", "sensors": ["sentinel3"]})


def test_build_request_honours_explicit_thermal_resolution():
    payload = {
        "aoi": {"west": 13.3, "south": 52.4, "east": 13.5, "north": 52.6},
        "start": "2024-06-01",
        "end": "2024-06-05",
        "sensors": ["sentinel3"],
        "resolution_m": 100,
        "thermal_resolution_m": 300,
    }
    request = build_request(payload)
    assert request.thermal_grid.resolution_m == 300


def test_build_request_passes_through_full_grid_shape_unchanged():
    # When grid/thermal_grid/predictor_grid are already present (e.g. a
    # request re-submitted via AnalysisRequest.to_dict()), build_request
    # must defer to AnalysisRequest.from_dict rather than recompute anything.
    grid_dict = {
        "grid_id": "custom",
        "crs": "EPSG:32633",
        "bounds": [0, 0, 1000, 1000],
        "resolution_m": 100,
        "city_id": None,
    }
    payload = {
        "aoi": {"west": 13.3, "south": 52.4, "east": 13.5, "north": 52.6},
        "start": "2024-06-01",
        "end": "2024-06-05",
        "sensors": ["sentinel3"],
        "grid": grid_dict,
        "thermal_grid": grid_dict,
    }
    request = build_request(payload)
    assert request.grid.grid_id == "custom"
    assert request.grid.bounds == (0, 0, 1000, 1000)
