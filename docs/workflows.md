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
signing) and `optical` extras; they have not yet been validated against real
scenes, see `docs/roadmap.md`.

### Storage: AOI subsets instead of raw archives

`raw_retention` (default `"aoi_subset"`) controls what stays on disk. Each
Sentinel-3, Sentinel-5P and Sentinel-2 (SAFE path) product is read once,
cropped to the AOI plus a small margin, and stored as NetCDF under
`<output>/subsets/<sensor>/`, keyed by product, AOI and subset format; the
downloaded original is then deleted. A later run over the same AOI reads
the subsets and downloads nothing. The trade-off: a larger AOI or another
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
fused cube, static `elevation`/`slope` to the predictor cube, and the fine
static terrain as `result.terrain`. For a fine-resolution prediction at a
given time, `sensors.terrain.terrain_predictors(result.terrain, [time],
crs=...)` returns the matching `cos_incidence`. These are the predictors
ESA's Sen-ET sharpening uses besides Sentinel-2 reflectance.

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

`plan` does not access the network. `discover` queries Sentinel catalogs.
`auxiliary` downloads only the configured auxiliary sources. `run` executes
the whole request (work files under `output/run/work`, cubes under
`output/run/result`); keep `--max-workers` at 1 on machines with limited
RAM, since each product can need several GB.
