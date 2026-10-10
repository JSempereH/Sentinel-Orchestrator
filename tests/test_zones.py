from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import xarray as xr

from citycube import AOI, AnalysisRequest, ZoneSet, aoi_mask, clip_to_aoi, downscale_per_scene, zonal_statistics
from citycube.cities import get_city
from citycube.catalog import ProductRef, annotate_aoi_coverage
from citycube.cube import projected_bounds
from citycube.config import MAX_AOI_VERTICES

TRIANGLE = {"type": "Polygon", "coordinates": [[[13.3, 52.4], [13.5, 52.4], [13.4, 52.6], [13.3, 52.4]]]}
SQUARE = {"type": "Polygon", "coordinates": [[[13.3, 52.4], [13.4, 52.4], [13.4, 52.5], [13.3, 52.5], [13.3, 52.4]]]}
CRS = "EPSG:32633"


def grid(values=None, *, step=100.0, times=2) -> xr.DataArray:
    west, south, east, north = projected_bounds(AOI(13.3, 52.4, 13.5, 52.6), CRS)
    x = np.arange(west, east, step) + step / 2
    y = np.arange(north, south, -step) - step / 2
    if values is None:
        values = np.random.default_rng(0).normal(30, 2, (times, y.size, x.size))
    time = np.array(["2026-08-01T10:00", "2026-08-02T10:00"][:times], dtype="datetime64[ns]")
    return xr.DataArray(values, dims=("time", "y", "x"), coords={"time": time, "y": y, "x": x}, attrs={"crs": CRS, "units": "degC"}, name="lst")


def test_polygon_aoi_bounds_and_round_trip():
    aoi = AOI.from_geojson(TRIANGLE)
    assert (aoi.west, aoi.south, aoi.east, aoi.north) == pytest.approx((13.3, 52.4, 13.5, 52.6))
    assert aoi.has_geometry
    assert AOI.from_dict(json.loads(json.dumps(aoi.to_dict()))) == aoi
    assert AOI.from_geometry(aoi.geometry_wkt) == aoi
    feature = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": TRIANGLE}]}
    assert AOI.from_geojson(json.dumps(feature)) == aoi
    assert hash(aoi) == hash(AOI.from_geojson(TRIANGLE))


def test_plain_aoi_round_trip_has_no_geometry():
    aoi = AOI(13.3, 52.4, 13.5, 52.6)
    assert aoi.to_dict() == {"west": 13.3, "south": 52.4, "east": 13.5, "north": 52.6}
    assert AOI.from_dict(aoi.to_dict()) == aoi
    with pytest.raises(ValueError, match="Missing AOI keys"):
        AOI.from_dict({"west": 1, "south": 2})


def test_polygon_aoi_rejects_bad_geometry():
    with pytest.raises(ValueError, match="antimeridian"):
        AOI.from_geojson({"type": "Polygon", "coordinates": [[[-179, 0], [179, 0], [179, 1], [-179, 1], [-179, 0]]]})
    with pytest.raises(ValueError, match="Polygon"):
        AOI(0, 0, 1, 1, geometry_wkt="POINT (0.5 0.5)")
    with pytest.raises(ValueError, match="beyond"):
        AOI(0, 0, 1, 1, geometry_wkt="POLYGON ((0 0, 2 0, 2 2, 0 0))")
    ring = [[13 + 0.1 * np.cos(a), 52 + 0.1 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, MAX_AOI_VERTICES + 2)]
    ring[-1] = ring[0]
    with pytest.raises(ValueError, match="vertices"):
        AOI.from_geojson({"type": "Polygon", "coordinates": [ring]})


def test_request_serialises_polygon_aoi():
    request = dataclasses.replace(AnalysisRequest.for_city(get_city("berlin"), "2026-08-01", "2026-08-02"), aoi=AOI.from_geojson(TRIANGLE))
    assert AnalysisRequest.from_dict(json.loads(json.dumps(request.to_dict()))).aoi == request.aoi


def test_aoi_mask_follows_the_polygon():
    data = grid()
    mask = aoi_mask(data, AOI.from_geojson(TRIANGLE))
    assert mask.dtype == bool and mask.dims == ("y", "x")
    assert 0.4 < float(mask.mean()) < 0.55  # a triangle is half its box
    assert bool(aoi_mask(data, AOI(13.3, 52.4, 13.5, 52.6)).mean() > 0.9)  # a lon/lat box bows in UTM
    clipped = clip_to_aoi(data, AOI.from_geojson(TRIANGLE))
    assert np.isnan(clipped.where(~mask)).all()
    assert np.isfinite(clipped.where(mask, 0)).all()


def test_aoi_mask_needs_crs():
    with pytest.raises(ValueError, match="crs"):
        aoi_mask(grid().assign_attrs(crs=None), AOI.from_geojson(TRIANGLE))


def test_zonal_statistics_match_numpy():
    data = grid()
    zones = ZoneSet.from_geojson({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": "square"}, "geometry": SQUARE},
        {"type": "Feature", "properties": {"name": "triangle"}, "geometry": TRIANGLE},
    ]}, id_field="name")
    data[0, :5] = np.nan
    table = zonal_statistics(data, zones, statistics=("count", "mean", "std", "min", "max", "median", "p90"))
    assert list(table.columns) == ["zone", "time", "count", "mean", "std", "min", "max", "median", "p90"]
    assert len(table) == 4
    labels = zones.labels(data).values
    for _, row in table.iterrows():
        values = data.sel(time=row["time"]).values[labels == zones.ids.index(row["zone"])]
        values = values[np.isfinite(values)]
        assert row["count"] == values.size
        assert row["mean"] == pytest.approx(values.mean())
        assert row["std"] == pytest.approx(values.std())
        assert row["p90"] == pytest.approx(np.quantile(values, 0.9))
    # Overlap goes to the first zone, so the triangle loses the shared corner.
    assert (labels == 0).sum() > 0 and (labels == 1).sum() < aoi_mask(data, AOI.from_geojson(TRIANGLE)).sum()


def test_zonal_statistics_on_2d_and_empty_zone(tmp_path):
    data = grid(times=1).isel(time=0, drop=True)
    far = {"type": "Polygon", "coordinates": [[[0, 0], [0.1, 0], [0.1, 0.1], [0, 0]]]}
    zones = ZoneSet.from_geojson({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": TRIANGLE},
        {"type": "Feature", "properties": {}, "geometry": far},
    ]})
    table = zonal_statistics(data, zones)
    assert "time" not in table and list(table["zone"]) == ["0", "1"]
    assert table.loc[1, "count"] == 0 and np.isnan(table.loc[1, "mean"])
    written = json.loads(zones.to_geojson(table, tmp_path / "zones.geojson").read_text())
    assert written["features"][0]["properties"]["mean"] == pytest.approx(table.loc[0, "mean"])
    assert written["features"][1]["properties"]["mean"] is None


def test_zone_set_validation():
    with pytest.raises(ValueError, match="no 'name'"):
        ZoneSet.from_geojson({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": SQUARE}]}, id_field="name")
    with pytest.raises(ValueError, match="unique"):
        ZoneSet(ids=("a", "a"), geometries_wkt=("POLYGON ((0 0, 1 0, 1 1, 0 0))",) * 2)
    with pytest.raises(ValueError, match="Unknown"):
        zonal_statistics(grid(), ZoneSet(ids=("a",), geometries_wkt=("POLYGON ((0 0, 1 0, 1 1, 0 0))",)), statistics=("mode",))


def test_coverage_uses_the_polygon():
    # A footprint over the box's top-left corner, which the triangle leaves out.
    corner = {"type": "Polygon", "coordinates": [[[13.3, 52.55], [13.33, 52.55], [13.33, 52.6], [13.3, 52.6], [13.3, 52.55]]]}
    product = ProductRef(product_id="p", name="p", product_type="x", start_datetime="2026-08-01T10:00:00Z", end_datetime=None,
                         timeliness=None, online=True, download_url="", metadata={"GeoFootprint": corner})
    assert annotate_aoi_coverage([product], AOI.from_geojson(TRIANGLE))[0].coverage == pytest.approx(0.0, abs=1e-9)
    assert annotate_aoi_coverage([product], AOI(13.3, 52.4, 13.5, 52.6))[0].coverage > 0.01


def test_per_scene_domain_blanks_outside(monkeypatch):
    coarse_x = np.arange(5) * 1000.0 + 500
    coarse = xr.Dataset(
        {
            "lst": (("time", "y", "x"), np.random.default_rng(1).normal(30, 2, (1, 5, 5))),
            "ndvi": (("time", "y", "x"), np.random.default_rng(2).random((1, 5, 5))),
            "sentinel2_matched_time": (("time",), np.array(["2026-08-01T10:00"], dtype="datetime64[ns]")),
        },
        coords={"time": np.array(["2026-08-01T10:00"], dtype="datetime64[ns]"), "y": coarse_x[::-1], "x": coarse_x},
        attrs={"crs": CRS},
    )
    fine_x = np.arange(50) * 100.0 + 50
    fine = xr.Dataset(
        {"ndvi": (("time", "y", "x"), np.random.default_rng(3).random((1, 50, 50)))},
        coords={"time": coarse.time.values, "y": fine_x[::-1], "x": fine_x},
        attrs={"crs": CRS},
    )
    domain = xr.DataArray(np.tri(50, dtype=bool), dims=("y", "x"), coords={"y": fine.y, "x": fine.x})
    result = downscale_per_scene(coarse, fine, predictors=["ndvi"], model="linear", min_samples=5, domain=domain)
    values = result["lst_downscaled"].isel(time=0)
    assert np.isnan(values.where(~domain)).all()
    assert np.isfinite(values.where(domain, 0)).all()
    assert values.attrs["crs"] == CRS  # each map carries the CRS, so it can be masked on its own
