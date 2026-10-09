# Workflows

## Declarative Request

```python
from sentinel_analysis import AnalysisRequest, AuxiliarySpec, get_city

request = AnalysisRequest.for_city(
    get_city("Berlin"),
    "2025-06-19",
    "2025-06-19",
    sensors=("sentinel3",),
    auxiliary=(
        AuxiliarySpec("era5", variables=("air_temperature_2m",)),
        AuxiliarySpec("cams", variables=("NO2",)),
        AuxiliarySpec("openaq", variables=("NO2",), options={"max_locations": 2}),
    ),
)
```

For an arbitrary area, `AnalysisGrid.for_aoi` snaps a grid in the AOI's
UTM zone (edges densified, so the whole AOI is enclosed):

```python
from sentinel_analysis import AOI, AnalysisGrid, AnalysisRequest

aoi = AOI(west=-3.80, south=40.35, east=-3.60, north=40.50)
predictor_grid = AnalysisGrid.for_aoi(aoi, resolution_m=100)
thermal_grid = AnalysisGrid.for_aoi(aoi, resolution_m=1000, crs=predictor_grid.crs)
request = AnalysisRequest(
    aoi=aoi, start="2025-07-01", end="2025-07-07",
    sensors=("sentinel3", "sentinel2"),
    grid=predictor_grid, predictor_grid=predictor_grid, thermal_grid=thermal_grid,
    sentinel2_source="stac_cog",  # read COGs in place instead of full SAFE downloads
)
```

`sentinel1_backend` accepts `"snap"` (default), `"hyp3_rtc"`, `"pc_rtc"`
and `"s1ard"`; `sentinel2_source` accepts `"cdse_safe"` (default) and
`"stac_cog"`. The cloud-native options (`"pc_rtc"`, `"stac_cog"`) read only
the blocks covering the grid and need the `landsat` (Planetary Computer
signing) and `optical` extras. Both are validated against real scenes
(`scripts/validate_cloud_native_real_reference.py`) but stay opt-in because
they change the data source.

### Day or night thermal passes

`thermal_overpass` (`"any"` by default, `"day"` or `"night"`) keeps only
Sentinel-3 and ECOSTRESS acquisitions whose local mean solar time at the AOI
centre falls inside (or outside) 06:00-18:00. It is applied during
discovery, before ranking, so night passes are never downloaded and do not
take the slots of daytime ones. Use `"day"` for downscaling: the relation
between temperature and optical predictors only holds while the sun heats
the surface.

### Failed products

With `on_product_error="skip"` (the default) a product that fails to
download, extract or read is logged, left out and recorded in
`result.provenance["failed_products"]` (`sensor`, `product`, `error`); one
corrupt archive no longer aborts a run of a hundred products. If *every*
product of a sensor fails the run still stops with
`ProductAcquisitionError`, since continuing without that sensor would
silently change the result (for example another sensor becoming the fusion
target). `on_product_error="raise"` stops at the first failure.

### Product selection: AOI coverage and clear sky

Every catalogue product carries its footprint (CDSE `GeoFootprint`, STAC
item geometry), so before anything is downloaded each product gets
`metadata["aoi_coverage"]`: the fraction of the AOI it covers, with tiles
of one pass (same start time) counted together. Products below
`min_aoi_coverage` (default 0.3) are dropped, and the remaining ones are
ranked by coverage and cloud cover. A Sentinel-3 granule that only grazes
the AOI edge no longer takes a slot from one that covers it.

Sentinel-3 cloud cover in the catalogue is for the whole 1500 km granule,
which says little about one city. With `min_clear_fraction > 0` the
workflow takes three times as many Sentinel-3 candidates, downloads only
their geolocation, cloud mask and view geometry (`geodetic_in.nc`,
`flags_in.nc`, `geometry_tn.nc`, ~14 MB, `SLSTR_PROBE_FILES`), and keeps
the products whose usable fraction of the AOI reaches the threshold. Usable
means what the default quality screening will keep: clear for the Bayesian
cloud mask *and* seen at no more than 45 degrees (a clear scene at the
swath edge is otherwise downloaded only to be screened out entirely).
Rejected products are listed in `provenance["probed_out"]`. Products already
stored as AOI subsets are judged from the subset itself, with no download.

### Polygon AOIs and zonal statistics

An `AOI` is a bounding box and, optionally, the exact polygon of the area:

```python
from sentinel_analysis import AOI, ZoneSet, zonal_statistics, clip_to_aoi

aoi = AOI.from_geojson("berlin_boundary.geojson")  # file, JSON text or dict; features are merged
request = dataclasses.replace(request, aoi=aoi)
```

Catalogue queries and the analysis grids still use the bounding box (a grid
is rectangular, and a detailed boundary does not fit in a query URL). The
polygon is used where the shape matters:

- `aoi_coverage` and the Sentinel-3 clear-sky probe measure the polygon,
  not its box;
- downscaled values outside the polygon are blanked (`aoi_mask`);
- `clip_to_aoi(dataset, aoi)` blanks any other cube outside it.

`AOI.to_dict()` adds a GeoJSON `geometry` and `AnalysisRequest.to_dict()`
carries it, so a polygon request survives `request.json` and the worker
API, whose map sends the drawn polygon when it is not a plain rectangle.
Polygons crossing the antimeridian and polygons with more than
`MAX_AOI_VERTICES` (10 000) vertices are rejected: simplify a detailed
boundary first.

Zonal statistics summarise a grid by districts or any other polygons:

```python
zones = ZoneSet.from_geojson("districts.geojson", id_field="name")
table = zonal_statistics(result.downscaled["lst_downscaled"], zones,
                         statistics=("count", "mean", "p90"))
table.to_csv("district_lst.csv", index=False)
zones.to_geojson(table, "district_lst.geojson", value="mean")
```

A cell belongs to a zone when its centre lies inside it (the first zone
wins where zones overlap). Statistics ignore NaN; `count` is the number of
valid cells, so a district hidden by clouds shows `count == 0` rather than
a misleading mean. One time step is read at a time, so a lazily opened
Zarr store is never loaded whole. Available statistics:
`ZONAL_STATISTICS` (`count`, `mean`, `std`, `min`, `max`, `median`, `p10`,
`p90`).

### Storage: AOI subsets instead of raw archives

`raw_retention` (default `"aoi_subset"`) controls what stays on disk. Each
Sentinel-3, Sentinel-5P and Sentinel-2 (SAFE path) product is read once,
cropped to the AOI plus a small margin, and stored as NetCDF under
`<output>/subsets/<sensor>/`, keyed by product, AOI and subset format; the
downloaded original is then deleted. A later run over the same AOI reads
the subsets and downloads nothing. Bump `SUBSET_FORMAT_VERSION`
(`workflow/adapters.py`) whenever what a reader stores changes, or old
subsets are reused as they are. The trade-off: a larger AOI or another
variable later means downloading those products again (the product id is
kept in each subset's attributes). `raw_retention="keep"` keeps the
originals too.

Sentinel-3 products are not even downloaded whole: only the files an LST
analysis reads (`LST_in.nc`, `geodetic_in.nc`, `flags_in.nc`,
`geometry_tn.nc`, `xfdumanifest.xml`, ~17 MB) are fetched through CDSE's
OData `Nodes` endpoint instead of the ~70 MB archive (more than half of
which is meteorological profiles, `met_tx.nc`). Sentinel-5P keeps one
product per orbit (reprocessed > offline > near-real-time).

### Terrain predictors

`terrain_predictors=True` loads Copernicus GLO-30 elevation (Planetary
Computer COGs, `landsat` + `optical` extras) on the predictor grid and adds
`elevation`, `slope` and `cos_incidence` (cosine of the solar incidence
angle on the tilted terrain at each Sentinel-3 acquisition time) to the
fused cube and the fine static terrain as `result.terrain`. For a fine-resolution prediction at a
given time, `sensors.terrain.terrain_predictors(result.terrain, [time],
crs=...)` returns the matching `cos_incidence`. These are the predictors
ESA's Sen-ET sharpening uses besides Sentinel-2 reflectance.

### Downscaling

`downscale=DownscaleSpec(...)` adds a final stage that sharpens every
fused thermal scene onto the predictor grid and returns it as
`result.downscaled` (`downscaled.zarr` when saved):

```python
import dataclasses

from sentinel_analysis import DownscaleSpec

request = AnalysisRequest.for_city(
    get_city("Berlin"), "2026-08-01", "2026-08-14",
    sensors=("sentinel3", "sentinel2"), resolution_m=100,
)
request = dataclasses.replace(
    request,
    sentinel2_source="stac_cog",
    terrain_predictors=True,
    thermal_overpass="day",
    downscale=DownscaleSpec()  # model="local_trees" by default,
)
```

It uses the per-scene protocol the multi-city benchmark found most accurate
(`docs/downscaling.md`): for each coarse scene a model (`"linear"`,
`"random_forest"`, `"xgboost"` or `"local_trees"`) is fitted on that scene's own 1 km cells
and aggregated predictors, predicts its matched Sentinel-2 scene at fine
resolution, and (with `coarse_consistent=True`, the default) is corrected
so it averages back to the observed coarse scene. Predictors default to
the Sentinel-2 indices plus whatever terrain predictors the run produced;
`cos_incidence` is recomputed at each thermal acquisition time. Scenes with
no Sentinel-2 match within the temporal tolerance or fewer than
`min_samples` clear cells are skipped and listed in
`provenance["downscaling"]["skipped_scenes"]`. Outside a workflow, call
`downscale_per_scene(result.cube, result.predictors["sentinel2"], terrain=result.terrain)`
directly.

## Execute

```python
from sentinel_analysis import AnalysisWorkflow, ClientConfig

result = AnalysisWorkflow(request).execute(
    "output/berlin",
    config=ClientConfig.from_env(),
)
result.save("output/berlin/result")  # Zarr cubes + request.json + provenance.json
```

Discovery scans up to 1000 catalogue candidates per sensor and only then
keeps the best `max_products_per_sensor` (online first, least cloud), so a
long period is not truncated to its earliest scenes. Every prepared cube is
sorted chronologically before fusion.

### Size limits

All prepared cubes are held in memory, so before anything is downloaded
`execute()` estimates their size (grid cells x `max_products_per_sensor` x
variables per sensor) and raises `RequestTooLargeError` above
`limits=RequestLimits(max_estimated_gb=16)` by default. Peak memory during
processing is typically up to about twice the estimate. Pass your own
`RequestLimits(max_aoi_km2=..., max_products_per_sensor=..., max_estimated_gb=...)`,
or `limits=None` to disable the check; `estimate_request(request)` and
`sentinel-analysis plan request.json` show the estimate without running
anything. The worker applies its own limits at submission (see
`worker/.env.example`). The estimate does not cover external processors:
a local SNAP run of one Sentinel-1 scene needs ~11 GB on its own.

## Adding a sensor

Write one adapter with `search(request, *, limit)` and
`acquire(references, context)` in `workflow/adapters.py`, register it in
`SENSOR_ADAPTERS`, and add its name to `request.SUPPORTED_SENSORS`. Auxiliary
providers register in `providers.AUXILIARY_PROVIDER_FACTORIES` the same way.

## CLI

```bash
sentinel-analysis plan request.json
sentinel-analysis discover request.json
sentinel-analysis auxiliary request.json output/auxiliary
sentinel-analysis run request.json output/run [--max-workers 1] [--gpt gpt]
```

`plan` does not access the network and prints the size estimate. `discover` queries Sentinel catalogs.
`auxiliary` downloads only the configured auxiliary sources. `run` executes
the whole request (work files under `output/run/work`, cubes under
`output/run/result`); keep `--max-workers` at 1 on machines with limited
RAM, since each product can need several GB.
