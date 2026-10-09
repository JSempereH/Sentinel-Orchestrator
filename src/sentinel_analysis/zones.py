"""Polygon masks and zonal statistics on analysis grids.

Grids are projected (``attrs["crs"]``, cell-centre ``y``/``x`` coordinates);
polygons are WGS84 and are projected to the grid once. A cell belongs to a
polygon when its centre lies inside it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from .config import AOI

ZONAL_STATISTICS = ("count", "mean", "std", "min", "max", "median", "p10", "p90")
DEFAULT_ZONAL_STATISTICS = ("count", "mean", "std", "min", "max")
# Rows of cell centres tested at once; bounds memory on very fine grids.
_ROWS_PER_BLOCK = 256


def _projected(geometry, crs: str):
    import shapely
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return shapely.transform(geometry, lambda xy: np.column_stack(transformer.transform(xy[:, 0], xy[:, 1])))


def _grid_crs(grid: xr.Dataset | xr.DataArray) -> str:
    crs = grid.attrs.get("crs")
    if not crs:
        raise ValueError("The grid has no 'crs' attribute; masks need to know its projection")
    return str(crs)


def _contains(geometry, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Boolean (y, x) array: cell centres inside a projected geometry."""

    import shapely

    shapely.prepare(geometry)
    inside = np.zeros((y.size, x.size), dtype=bool)
    west, south, east, north = geometry.bounds
    columns = np.flatnonzero((x >= west) & (x <= east))
    if columns.size == 0:
        return inside
    for start in range(0, y.size, _ROWS_PER_BLOCK):
        rows = np.arange(start, min(start + _ROWS_PER_BLOCK, y.size))
        rows = rows[(y[rows] >= south) & (y[rows] <= north)]
        if rows.size == 0:
            continue
        xx, yy = np.meshgrid(x[columns], y[rows])
        inside[np.ix_(rows, columns)] = shapely.contains_xy(geometry, xx, yy)
    return inside


def aoi_mask(grid: xr.Dataset | xr.DataArray, aoi: AOI) -> xr.DataArray:
    """True where a grid cell's centre is inside the AOI (its polygon, or its box)."""

    inside = _contains(_projected(aoi.shape(), _grid_crs(grid)), np.asarray(grid.x.values), np.asarray(grid.y.values))
    return xr.DataArray(inside, dims=("y", "x"), coords={"y": grid.y, "x": grid.x}, name="in_aoi")


def clip_to_aoi(data: xr.Dataset | xr.DataArray, aoi: AOI) -> xr.Dataset | xr.DataArray:
    """Blank (NaN) every value outside the AOI polygon; boolean variables become False there."""

    mask = aoi_mask(data, aoi)
    if isinstance(data, xr.DataArray):
        return data.where(mask, False) if data.dtype == bool else data.where(mask)
    clipped = data.copy()
    for name, variable in data.data_vars.items():
        if {"y", "x"} <= set(variable.dims):
            clipped[name] = variable.where(mask, False) if variable.dtype == bool else variable.where(mask)
    return clipped


@dataclass(frozen=True)
class ZoneSet:
    """Named WGS84 polygons (districts, neighbourhoods, parks) to summarise a grid by."""

    ids: tuple[str, ...]
    geometries_wkt: tuple[str, ...]
    properties: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if len(self.ids) != len(self.geometries_wkt):
            raise ValueError("A ZoneSet needs exactly one geometry per id")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Zone ids must be unique")
        if self.properties and len(self.properties) != len(self.ids):
            raise ValueError("A ZoneSet needs one properties mapping per zone, or none")
        if not self.ids:
            raise ValueError("A ZoneSet needs at least one zone")

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def from_geojson(cls, source: str | Path | Mapping[str, Any], id_field: str | None = None) -> "ZoneSet":
        """Zones from a GeoJSON FeatureCollection (path, JSON text or parsed object).

        ``id_field`` names the property used as zone id; without it features
        are numbered from 0. Only polygon features are kept.
        """

        import shapely
        from shapely.geometry import shape

        if isinstance(source, Mapping):
            collection = source
        else:
            text = str(source)
            collection = json.loads(text if text.lstrip().startswith("{") else Path(source).read_text(encoding="utf-8"))
        features = collection.get("features") if collection.get("type") == "FeatureCollection" else [collection]
        ids: list[str] = []
        geometries: list[str] = []
        properties: list[Mapping[str, Any]] = []
        for index, feature in enumerate(features or []):
            if not feature.get("geometry"):
                continue
            geometry = shapely.make_valid(shape(feature["geometry"]))
            if geometry.geom_type == "GeometryCollection":
                geometry = shapely.unary_union([part for part in geometry.geoms if part.geom_type in ("Polygon", "MultiPolygon")])
            if geometry.is_empty or geometry.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            feature_properties = dict(feature.get("properties") or {})
            if id_field is not None:
                if id_field not in feature_properties:
                    raise ValueError(f"Feature {index} has no '{id_field}' property")
                ids.append(str(feature_properties[id_field]))
            else:
                ids.append(str(index))
            geometries.append(geometry.wkt)
            properties.append(feature_properties)
        return cls(ids=tuple(ids), geometries_wkt=tuple(geometries), properties=tuple(properties))

    def shapes(self) -> list:
        from shapely import wkt

        return [wkt.loads(text) for text in self.geometries_wkt]

    def labels(self, grid: xr.Dataset | xr.DataArray) -> xr.DataArray:
        """Zone index of each grid cell (-1 outside every zone; the first zone wins overlaps)."""

        crs = _grid_crs(grid)
        x, y = np.asarray(grid.x.values), np.asarray(grid.y.values)
        labels = np.full((y.size, x.size), -1, dtype=np.int32)
        for index, geometry in enumerate(self.shapes()):
            inside = _contains(_projected(geometry, crs), x, y)
            labels[inside & (labels < 0)] = index
        return xr.DataArray(labels, dims=("y", "x"), coords={"y": grid.y, "x": grid.x}, name="zone")

    def to_geojson(self, statistics: "Any", path: str | Path, *, value: str = "mean") -> Path:
        """Write the zones as GeoJSON with one property per time step of ``value``.

        ``statistics`` is a ``zonal_statistics`` table; properties are named
        ``<value>_<ISO time>`` (or just ``<value>`` without a time column).
        """

        from shapely.geometry import mapping

        path = Path(path)
        by_zone: dict[str, dict[str, Any]] = {zone: {} for zone in self.ids}
        for row in statistics.to_dict("records"):
            key = value if "time" not in row else f"{value}_{np.datetime_as_string(np.datetime64(row['time']), unit='m')}"
            entry = row.get(value)
            by_zone[str(row["zone"])][key] = None if entry is None or (isinstance(entry, float) and not np.isfinite(entry)) else entry
        features = []
        for index, (zone, geometry) in enumerate(zip(self.ids, self.shapes())):
            properties = {**(self.properties[index] if self.properties else {}), "zone": zone, **by_zone[zone]}
            features.append({"type": "Feature", "properties": _jsonable(properties), "geometry": mapping(geometry)})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
        return path


def _jsonable(values: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, entry in values.items():
        if isinstance(entry, np.generic):
            entry = entry.item()
        result[key] = entry
    return result


def zonal_statistics(data: xr.DataArray, zones: ZoneSet, *, statistics: Sequence[str] = DEFAULT_ZONAL_STATISTICS):
    """Per-zone, per-time statistics of a (time, y, x) or (y, x) grid, ignoring NaN.

    Returns a pandas DataFrame with columns ``zone``, ``time`` (when the data
    has a time axis) and one column per statistic; ``count`` is the number of
    valid cells. One time step is loaded at a time, so a lazily opened Zarr
    store is never read whole.
    """

    import pandas as pd

    unknown = [name for name in statistics if name not in ZONAL_STATISTICS]
    if unknown:
        raise ValueError(f"Unknown zonal statistics {unknown}; choose from {ZONAL_STATISTICS}")
    if not {"y", "x"} <= set(data.dims) or not set(data.dims) <= {"time", "y", "x"}:
        raise ValueError("zonal_statistics expects a (y, x) or (time, y, x) DataArray")
    labels = zones.labels(data).values.ravel()
    inside = labels >= 0
    zone_labels = labels[inside]
    steps = list(data.time.values) if "time" in data.dims else [None]
    frames = []
    quantiles = {"median": 0.5, "p10": 0.1, "p90": 0.9}
    for step in steps:
        values = (data.sel(time=step) if step is not None else data).transpose("y", "x").values.ravel()[inside]
        valid = np.isfinite(values)
        frame = pd.DataFrame({"zone": zone_labels[valid], "value": values[valid].astype(np.float64)})
        grouped = frame.groupby("zone")["value"]
        table = pd.DataFrame(index=pd.RangeIndex(len(zones), name="zone"))
        for name in statistics:
            if name in quantiles:
                table[name] = grouped.quantile(quantiles[name])
            elif name == "std":
                table[name] = grouped.std(ddof=0)
            else:
                table[name] = getattr(grouped, name)()
        if "count" in table:
            table["count"] = table["count"].fillna(0).astype(np.int64)
        table = table.reset_index()
        table["zone"] = [zones.ids[index] for index in table["zone"]]
        if step is not None:
            table.insert(1, "time", pd.Timestamp(step))
        frames.append(table)
    result = pd.concat(frames, ignore_index=True)
    result.attrs["variable"] = data.name
    result.attrs["units"] = data.attrs.get("units")
    return result
