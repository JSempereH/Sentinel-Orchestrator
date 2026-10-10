# Change history

Chronological record of each development pass: what was found, fixed,
validated and decided, with the evidence. It is kept as written at the
time, so figures and statuses in older entries may be superseded by later
ones; the current state is in [`roadmap.md`](roadmap.md).

The package was named `sentinel_analysis` until 2026-10-10. Entries below
use the current name, `citycube`, for paths and imports.

## Done

- **Sensors**: Sentinel-1/2/3/5P, Landsat 8/9, and ECOSTRESS - all fully
  verified against real data, including a real authenticated ECOSTRESS
  pixel read (Berlin, 2026-09) that returned physically correct values.
  Detail: `docs/downscaling.md` roadmap item 3.
- **Worker API**: submit/poll/download (`POST /jobs`, `GET /jobs/{id}`,
  `GET /jobs/{id}/result`), job durability across restarts (SQLite-backed
  `JobState`, `_migrate_schema` for schema evolution without Alembic), a
  `/usage` snapshot for CDSE/CDS-ADS/OpenAQ. `GET /jobs` (list) and
  `request_params`/`name`/`created_at` on `JobState` were added in this
  pass specifically so a UI could show history, not just current status.
- **Worker UI**: `worker/frontend/` (React 18 + Vite + TS) - submit/monitor/
  download/settings, served by the worker itself with no CORS anywhere.
  Detail: `docs/platform.md`'s "Worker UI" section.
- **Containers**: `worker/Dockerfile` is a working multi-stage build (Node
  stage for the frontend, Python stage for the API); verified end to end in
  this pass with `docker compose build worker` + a real container smoke
  test (`GET /`, `/health`, `/jobs` all correct; a real job submitted
  against real CDSE data reached `SUCCEEDED` and downloaded a valid result
  zip).
- **Geographically weighted regression downscaler**: `fit_gwr_downscaler`/
  `fit_coarse_consistent_gwr_downscaler` - local coefficients refit per fine
  pixel from nearest coarse training points, unlike every other downscaler
  here's single global relation. Unit-tested (two opposite-slope spatial
  clusters, confirming it recovers each cluster's own relation where a
  global linear model cancels them toward ~0) and now also benchmarked
  against real data - see "Benchmarked against real data" below. Detail:
  `docs/downscaling.md` roadmap item 3.
- **STARFM classical spatiotemporal fusion**: `fuse_starfm` - the
  single-pair form (not the full multi-pair ensemble; FSDAF remains open).
  Unit-tested against synthetic textures (confirms its similarity filter
  keeps a localized coarse-scale change from bleeding into an unrelated
  adjacent surface's own prediction) and sanity-checked against real data -
  see "Benchmarked against real data" below. Detail: `docs/downscaling.md`
  roadmap item 4.
- **ESTARFM spatiotemporal fusion**: `fuse_estarfm` - unlike single-pair
  STARFM (which adds the raw coarse difference directly, assuming a 1:1
  fine/coarse relationship), this fits a local linear regression per pixel's
  window and uses each bracketing pair's own slope, weighting the two
  pairs' predictions by regression reliability. Unit-tested (a local
  regression beats STARFM's raw-difference approach on a slope change;
  falls back to identity and reindexes correctly on mismatched grids), not
  yet run against real data. Detail: `docs/downscaling.md`, "Implemented
  Model APIs".
- **Downscaler benchmark**: `scripts/benchmark_downscalers.py` fetches a
  real Sentinel-3 + Sentinel-2 Berlin cube and evaluates OLS, TsHARP,
  Random Forest, XGBoost and GWR on one real `blocked_spatiotemporal_split`
  holdout, plus a STARFM coarse-consistency sanity check - the benchmark
  `docs/downscaling.md` roadmap item 2 asked for, now runnable on demand.
  Detail: `docs/downscaling.md`, "Benchmark Results".
- **Tests**: `citycube` 56 passed / 5 skipped (4 need real local
  products, 1 needs scikit-learn - up from 52/5 with the new GWR/STARFM
  tests), `worker` 22 passed (was 14 before the worker-UI pass - added
  `test_main.py`'s endpoint-level coverage, including a test that only
  stops being skipped once the frontend is actually built).

## New from this pass

- **`sentinel1_backend="snap"` (the new default) timed end to end against a
  real Sentinel-1 IW GRD scene**: `process_s1_grd` (orbit file, thermal/
  border noise removal, calibration, terrain correction at 20 m) took
  **153.1s (2.6 min)**, producing a 2.0 GB terrain-corrected sigma0
  GeoTIFF (10638x24360 px) that `read_s1_grd` then read correctly
  (`sigma0_VV`/`sigma0_VH`/`valid_mask`). Peak RAM was ~10.9 GB (67%) with
  swap nearly exhausted - the machine did not freeze, but this is a real
  data point for anyone running this alongside other memory-heavy work on
  a similar machine: don't.
- **GWR's bandwidth-insensitive coverage (11.6% unchanged at 3x bandwidth,
  logged previously) is fixed, root cause confirmed empirically before
  touching it**: `GWRDownscaler`'s k-NN search ran over raw training rows,
  but training data is stacked over (time, y, x) - a pixel observed at T
  valid times appears T times at the same (x, y). A synthetic reproduction
  showed `max_local_samples=200` capped at just 8-10 *distinct* locations
  ~2 px away once T reached 20-28 (matching the 4-week benchmark) -
  bandwidth was never the limiting factor because the neighbor set itself
  never reached that far. Fixed by adding `_distinct_locations()`
  (`downscale.py`) and using it in both `GWRDownscaler.predict()` and
  `fit_gwr_downscaler`'s leave-one-out residual loop - the latter fix also
  closes a related leak (a different time step at the exact same pixel sat
  at distance 0 and wasn't excluded by row-only `indices[sample] != sample`).
  A synthetic before/after check (24 time steps, checkerboard holdout)
  confirmed coverage no longer plateaus.
- **The real 4-week benchmark was rerun with the GWR fix above ("Run 3" in
  `docs/downscaling.md`) - the fix is real but was not the dominant cause
  of low real-data coverage.** rmse shifted for both GWR configurations
  (proof the dedup fix changes which training points a local fit uses),
  but coverage stayed at exactly the same 11.6% it was at before the fix,
  identical between the 5x and 15x bandwidths again. Working hypothesis
  (not yet confirmed with real numbers): `fit_gwr_downscaler` only keeps
  training rows where every requested variable is finite
  (`np.isfinite(table).all(axis=1)`), so real Sentinel-3 cloud/quality
  gaps can leave whole regions of the training half of
  `blocked_spatiotemporal_split`'s checkerboard with zero qualifying
  samples - a gap no bandwidth increase closes. A live diagnostic script
  to confirm this (count distinct training locations after the finite
  filter, and how many fall within each bandwidth for held-out cells) was
  started but had to be aborted mid-run: this machine was under real
  memory pressure from unrelated concurrent processes at the time (not
  part of this benchmark - `scripts/run_loe_mission2.py` and
  `scripts/run_loe_l2_walkforward.py`, ~2.8 GB and ~0.8 GB RSS, apparently
  someone else's work on this same machine), and swap was fully exhausted.
  Re-run it once the machine is free, before touching `min_local_samples`
  or the split geometry.
- **`sentinel1_backend="hyp3_rtc"` implemented**: submits the granule name
  to ASF HyP3 for on-demand RTC processing in the cloud instead of a local
  GRD download+process (`Sentinel1RTCConfig`, `process_s1_rtc`,
  `read_s1_rtc`, the new `hyp3` extra). Architecturally different from
  `"snap"`/`"s1ard"` - `workflow/runner.py`'s sentinel1 handling branches to
  it *before* any CDSE download happens. Unit-tested against a mocked
  `hyp3_sdk` (no real HyP3 credits needed to test); a real submission has
  not been run yet.
- **A second, previously-undiscovered Sentinel-1 bug, unrelated to
  s1ard/SNAP**: `Sentinel1Catalog.search()` searched CDSE's STAC endpoint
  and returned raw `STACItem`s, which lack every field
  `select_product_refs()` needs (`online`, `cloud_cover`,
  `start_datetime`) - `AnalysisWorkflow.execute()` for `sentinel1` crashed
  with `AttributeError: 'STACItem' object has no attribute 'online'` on
  discovery alone, before any backend ran. Zero test coverage had ever
  exercised this path. Found while wiring up `hyp3_rtc` (which needs a
  real `ProductRef.name` to submit). Rewritten to search CDSE's OData
  catalogue instead - the same one `Sentinel2Catalog` already uses -
  returning real `ProductRef`s and excluding the "COG" GRD packaging
  noted above (`not contains(Name,'COG')`). Verified against a real query.
- **`sentinel1_backend="s1ard"` is confirmed broken end to end against a
  real Sentinel-1 GRD scene**, not just heavy to install. Chased 5 real
  bugs in sequence: (1) `gdal`'s Python bindings silently missing
  `osgeo._gdal_array` because numpy isn't visible at build time under
  `uv`'s isolated build - fixed generally, see `Makefile`'s
  `install-worker`; (2) CDSE now serves Sentinel-1 GRD in a "COG" packaging
  by default (`..._COG.SAFE`, e.g. `s1d-iw-grd-vh-...-cog.tiff`) that
  `pyroSAR.identify()`'s naming regex does not recognize at all - CDSE
  still serves the classic non-COG SAFE for the same acquisitions, just not
  through `Sentinel1Catalog`'s STAC search by default; (3) s1ard 2.13.1
  rejects `mode = sar, nrb` when `scene` is set directly (`if argument
  'scene' is set, the processing mode must be 'sar'`); (4) s1ard's `db_file`
  scene-inventory only gets populated when `scene_dir` is *also* set
  (`s1ard.processor.main` only calls `archive.insert()` in that branch) -
  a bare `db_file` path passes config validation but leaves the archive
  empty, and `check_acquisition_completeness()` then crashes
  (`ValueError: min() iterable argument is empty`) trying to find even the
  scene itself. (1)-(4) are fixed in `sensors/sentinel1.py`'s
  `_build_s1ard_config`. (5) is **not fixable here**: s1ard's own scene
  registration (`pyroSAR.Archive.insert()` -> `spatialist.vector.Vector.
  reproject()`) passes an `ogr.DataSource` to `gdal.VectorTranslate()`,
  which expects a `gdal.Dataset` - confirmed via a real, isolated
  reproduction (fails identically with a bare in-memory `ogr` datasource,
  no pyroSAR involved) and a real `gdal-dev` mailing list thread describing
  the same SWIG-typemap incompatibility. This is a `spatialist` bug, not
  ours or a build issue - rebuilding `gdal` cleanly (without the `--no-deps`
  used to fix (1)) did not change the outcome.
  **Decision**: `sentinel1_backend` now defaults to `"snap"`
  (`process_s1_grd`, which never touches `pyroSAR`/`spatialist`) everywhere
  it appears (`workflow/request.py` x2, `worker/app/runner.py`,
  `sensors/sentinel1.py`'s `process_s1`). `process_s1_ard` now raises a
  `UserWarning` pointing at this entry instead of failing silently.
  See `worker/README.md` for the user-facing version of this note. A
  cloud-based alternative (ASF HyP3's on-demand RTC product, reusing the
  `hyp3_sdk` pattern `InSAR-Orchestrator/packages/insar_core` already uses
  for InSAR jobs) is planned as a new `sentinel1_backend="hyp3_rtc"` -
  not yet implemented.
- **Downscaler model priority decided, not just benchmarked**: given run
  1 vs run 2's ranking flip (below), no new downscaling model work should
  build on top of the current benchmark's ranking. OLS/TsHARP stay the
  default recommendation (cheapest, never far off even when "losing");
  Random Forest/XGBoost stay available but untuned further until the
  benchmark is broadened (more AOIs/seasons) rather than tuned on one
  noisy comparison; GWR's next step is diagnosing *why* 11.6% coverage did
  not move when bandwidth tripled (`min_local_samples` vs. holdout block
  geometry - not another bandwidth change); FSDAF stays unimplemented until
  ESTARFM gets a real-data validation run (STARFM already has one via
  `scripts/validate_starfm_real_reference.py`) - adding a third
  spatiotemporal-fusion method before validating the second is premature.
- **`worker/Dockerfile` was missing the `cloud` extra** (zarr/dask) even
  though `execute_and_persist()` unconditionally calls `write_zarr()` on
  every run - every containerized job would have reached `SUCCEEDED` for
  discovery/processing and then failed at the final write. Caught by a real
  end-to-end run (submit → real CDSE download → process → write), not
  inferred from reading the code; fixed by adding `cloud` to both the
  Dockerfile's `pip install` and confirming `worker/.venv` locally. `s1ard`
  remains deliberately excluded (SNAP has no pip-installable path).
- **ECOSTRESS live pixel-read validation is done**, and it surfaced two real
  bugs neither could have been caught without a real token and real data:
  (1) GDAL's `vsicurl` does not work cleanly against this specific
  Earthdata Cloud host - curl auto-attaches a `~/.netrc` entry for
  `urs.earthdata.nasa.gov` mid-OAuth-redirect and loops forever instead of
  using the Bearer header; fixed by fetching the asset with a plain
  `requests.get()` into a temp file instead of opening the remote URL
  through rasterio/GDAL directly. (2) The scale/offset fallback assumed the
  published ECOSTRESS ATBD digital-number formula (`* 0.02`), which
  produced ~5 K ("-267 degC") on real data - that formula turns out to
  apply to the older Swath LSTE product; the real L2T tiles served here
  already store LST directly in Kelvin (identity scale/offset). Both fixed
  in `sensors/ecostress.py`; see its module docstring for the full
  explanation. **Incident**: debugging this also leaked the user's
  Earthdata bearer token and Basic Auth credentials into the terminal/chat
  transcript via a `CPL_CURL_VERBOSE`/dict-printing mistake - both were
  rotated. Verbose curl/header dumps must never be used when a real token
  is in the environment; check success/failure only.
- **Worker UI auth is a single shared bearer token**, matching the worker
  API it drives - a documented tradeoff of the platform's single-
  trusted-user model, not a bug. If this worker is ever exposed beyond a
  LAN/VPN, that assumption needs revisiting before anything else here.
- **The benchmark ran twice - a one-week pass and a four-week, tuned
  rerun - and the ranking changed.** Run 1 (one week) found OLS/TsHARP
  (5.4-5.5 K rmse) clearly beating Random Forest/XGBoost (9.0-9.2 K). Run 2
  (four weeks, `max_workers=1` for the memory-safety reason below, two GWR
  bandwidths, two named RF/XGBoost configurations) found every regression
  model within 0.1 K of every other (7.26-7.35 K) - XGBoost's plain
  defaults are now marginally *best*. Run 1's "simple models win" reading
  was a small-sample artifact, not a settled finding - the real lesson is
  that one AOI over days-to-weeks moves this ranking too much to trust
  either run alone. Correlation also collapsed to near zero (even
  negative) in run 2 despite a lower rmse than run 1's "worse" models -
  weak real predictive skill over a longer, more varied window, not a bug.
  Widening GWR's bandwidth 3x (5x -> 15x grid spacing) left its coverage
  completely unchanged (11.6% both times) - a real negative result
  disproving the natural assumption that bandwidth alone was the coverage
  bottleneck; `min_local_samples` and the holdout's block geometry are the
  more likely next things to check, not a wider bandwidth. STARFM's sanity
  check improved (5.9 K, still outside its own 1 K tolerance) but was
  seeded from XGBoost's own modelled map, not an independent observation -
  see `scripts/validate_starfm_real_reference.py` for the check that uses
  a real one. Full writeup with both runs' tables:
  `docs/downscaling.md`, "Benchmark Results".
- **This machine froze twice** when two live-data scripts
  (`benchmark_downscalers.py` and `validate_starfm_real_reference.py`)
  each ran concurrently with `max_workers=2` - no kernel OOM-kill was
  found in `dmesg`/`journalctl`, but the entire desktop session restarted
  both times immediately after, which is the strongest signal available.
  Fixed by never running more than one such script at a time and dropping
  `max_workers` to 1 in both (serial, not parallel, product processing) -
  confirmed safe on the rerun (memory returned to baseline after
  completion, no freeze). Treat this as the standing rule for any future
  live multi-product fetch on this machine, not just these two scripts.
- **The root `.venv`'s optional extras (`landsat`, `ml`, ...) were observed
  disappearing mid-session more than once** during this pass, unrelated to
  any command run here - something else (a concurrent `uv sync`-like
  operation, given other concurrent edits to `docs/platform.md` and
  `worker/README.md` observed the same session) appears to be resetting it.
  Not a code issue, but worth knowing if `pytest` unexpectedly reports
  Landsat/ECOSTRESS/GWR tests failing on `ModuleNotFoundError` again - the
  fix is `uv pip install --python .venv -e ".[cdse,auxiliary,optical,landsat,ecostress,ml]"`,
  not a code change.

## New from this pass (methane point-source detection data plumbing)

Added for parity with the *data* side of Varon et al. 2024 (Nat. Commun.
s41467-024-47754-y) - a methane point-source detection paper unrelated to
this platform's existing LST-downscaling focus. The deep-learning model
itself is explicitly out of scope; these are the four data-acquisition
gaps identified against that paper.

- **Sentinel-2 L1C (top-of-atmosphere) reading**: `read_s2_l1c`,
  `Sentinel2L1CReadConfig`, `SENTINEL2_L1C_PRODUCT_TYPE`, and
  `Sentinel2Catalog(product_type=...)` to search either product level
  through the same CDSE OData catalogue `Sentinel2Catalog` already used
  for L2A. L1C matters here because L2A's atmospheric correction assumes a
  methane-free atmosphere and would attenuate the SWIR absorption feature
  detection relies on. L1C's SAFE layout differs from L2A's (no SCL, no
  R10m/R20m/R60m resolution triplicates - one native-resolution file per
  band), so this is a new reader (`_band_file_l1c`), not a flag on the
  existing one. **Verified against a real product**
  (`scripts/validate_sentinel2_l1c_real_reference.py`, run against a real
  CDSE download - `S2A_MSIL1C_20260827T100701..._T33UUU...`, 588 MB,
  0% reported cloud cover, Berlin): all 10 configured bands read
  correctly, no SCL present as expected, shapes and CRS correct.
  - **A real, pre-existing bug was found and fixed by this real-data run,
    affecting `read_s2_l2a` too, not just the new L1C reader**: neither
    reader applied the per-band `RADIO_ADD_OFFSET`
    (`BOA_ADD_OFFSET` for L2A) ESA's processing baseline 04.00
    (2022-01-25 onward) introduced - true reflectance is
    `(DN + offset) / 10000`, not `DN / 10000`. The real downloaded product
    carries `RADIO_ADD_OFFSET = -1000` on every band (confirmed by parsing
    its `MTD_MSIL1C.xml` directly), so every pixel of every band read by
    the *old* code was **systematically 0.1 too high** - silent for any
    2022+ product (which is effectively all current Sentinel-2 data,
    L1C or L2A) until this run's own diagnostic band statistics caught
    unusually high reflectance and traced it to the metadata. Fixed by
    `_read_radiometric_offsets()` (new, namespace-agnostic XML parsing of
    the manifest's `band_id`-indexed offsets) in both readers; confirmed
    fixed by re-running the same script against the same cached product
    and seeing every band's min/max/mean shift down by exactly `0.1`, as
    the arithmetic predicts. Unit-tested
    (`test_read_radiometric_offsets_resolves_band_id_and_defaults_empty`)
    against a synthetic manifest. Pre-2022 products have no such element
    at all, so the fix defaults to offset 0 and is a no-op for them -
    nothing regresses for older data.
  - **Residual real-data caveat, not a bug**: after the offset fix, a few
    bands still show isolated pixels above the ~1.0-1.2 physically-typical
    reflectance range (max 2.7 on B8A) and a few slightly negative
    (min -0.09 on B05) - expected for raw, uncorrected TOA data over
    small bright/specular targets (rooftops, glint) and very dark ones,
    not something this reader should clip, since the methane-detection
    pipeline this feeds needs that real per-pixel signal (band statistics
    are the diagnostic that caught the offset bug in the first place, not
    evidence of a further one - the *mean* per band stayed physically
    unremarkable throughout, e.g. B8A mean 0.235).
- **Fixed-size chip partitioning**: `AnalysisGrid.chips(chip_size_m)` -
  splits a grid into uniform sub-grids (e.g. the paper's 2.5x2.5 km
  patches), as opposed to `from_bounds`'s single AOI-shaped extent.
  Unit-tested (tiling math, `drop_partial` behavior, rejection of a
  non-multiple chip size) - pure geometry, no real raster involved.
- **Reference/detection temporal pairing**: `workflow/pairing.py`'s
  `select_temporal_pair` picks the clearest scene in a short window before
  a target date and the clearest scene in an earlier window further back,
  matching the paper's "reference 1-4 months prior, detection within 7
  days prior" rule. Unit-tested (5 cases: window boundaries, cloud-cover
  ties, unknown-cloud-cover handling, empty-window errors) against
  synthetic `ProductRef`s.
- **Carbon Mapper's airborne methane plume catalog was deliberately kept
  out of the library, after initially being added as a first-class
  provider and then reverted.** It first landed as
  `providers/carbon_mapper.py` (`CarbonMapperProvider`, a new
  `carbon_mapper` entry in `AUXILIARY_PROVIDERS`), the same shape as
  ERA5/CAMS/OpenAQ. On reflection that shape is wrong for what this
  actually is: one static, versioned Zenodo file
  (`10.5281/zenodo.7072824`) from one specific 2020-2021 airborne
  (AVIRIS-NG/GAO) campaign - useful only for cross-referencing known
  methane leak events, unlike ERA5/CAMS/OpenAQ's broad, continuously
  updated, many-purpose data. Generalizing `AUXILIARY_PROVIDERS` to accept
  a first-class entry for every paper's fixed reference dataset does not
  scale as a library design principle, and it pulled in a
  dependency (`xlrd`) purely to parse a file that will never change.
  **Decision**: removed from `providers/`, `AUXILIARY_PROVIDERS`, and
  `AnalysisWorkflow`; the equivalent logic (fetch, cache, AOI/date filter)
  now lives as plain code in
  `notebooks/methane_detection_carbon_mapper.ipynb` - see "Guides" in the
  docs nav - built from the same real, downloaded-and-inspected Zenodo
  schema (legacy `.xls`, sheet `carbonmapper_ch4_plumelist_2020`, 2,527
  rows, columns `source_id/candidate_id/plume_lat/plume_lon/date/qplume/
  sigma_qplume/file_names`, Excel-serial dates confirmed round-tripping
  against a real row: serial `44020` -> `2020-07-08`, matching that row's
  `candidate_id` timestamp) and demonstrating the library's own generic
  primitives (`AssetCache`, `http_session`, `Sentinel2Catalog`,
  `select_temporal_pair`) instead of a bespoke provider class. `xlrd`
  moved to the `notebook` extra accordingly.
- **Explicitly not attempted**: the paper's synthetic training-data
  generator (Gaussian plume + Beer-Lambert absorption + autocorrelated
  noise injected into real background scenes) and the deep-learning model
  itself - both out of scope per this pass's request.

## New from this pass (library review: correctness, memory, architecture)

Driven by a full review of the library. Everything below is covered by
offline unit tests; nothing here was run against live data (the
cloud-native readers in particular still need a real-scene check - see the
deferred list).

- **Five correctness bugs confirmed with failing tests first, then fixed**:
  - *Unsorted time*: products are concatenated in catalogue-ranking order
    (online, cloud cover) rather than by date, and nothing sorted them -
    the "`cube.time` is not chronologically sorted" symptom the benchmark
    hit. `align_features`' nearest-time match then paired a target with
    the wrong acquisition (an unsorted feature cube produced `NaT`
    matches). Every adapter now concatenates through `concat_time()`
    (sorts), `combine_sentinel3` sorts, `AnalysisWorkflow.run()` sorts each
    input, and `align_features` sorts its features.
  - *Edge extrapolation*: same-CRS `harmonize_spatial` used nearest
    reindexing with no tolerance, so target cells outside a source's
    footprint silently repeated the source's edge values. Now bounded to
    half a source cell per axis; outside cells are NaN.
  - *Earliest, not best, products*: catalogues return date order and
    `discover()` asked for only `max_products_per_sensor` items before
    ranking, so a long period kept its first N scenes regardless of cloud
    cover. Discovery now scans up to `DISCOVERY_SCAN_LIMIT` (1000)
    candidates, then ranks and truncates.
  - *Lexical date validation*: `start > end` compared `str()` values, so
    `datetime(2025, 6, 12, 10)` vs `"2025-06-12T09:00"` passed
    (`" " < "T"`). Bounds are now parsed; a date-only end covers the day.
  - A duplicated line in `align_features`.
- **Memory**: `sensors/cog.py`'s `read_cog_to_grid` reads through a GDAL
  `WarpedVRT`, fetching only the blocks/overview covering the target grid.
  Landsat now uses it (it previously read the whole ~7.7k x 7.8k scene as
  float64 before reprojecting). `AssetCache` stores size+mtime and only
  re-hashes a file when they change (`valid(..., verify=True)` forces a
  full check); it previously SHA-256'd every multi-GB archive on every
  lookup, and `artifact_from_path` hashed every auxiliary file twice.
- **Cloud-native acquisition (opt-in, unvalidated on real data)**:
  `sentinel1_backend="pc_rtc"` (`Sentinel1RTCSTACCatalog`,
  `read_s1_rtc_cog`: Planetary Computer `sentinel-1-rtc` gamma0 COGs, no
  SNAP/HyP3) and `sentinel2_source="stac_cog"` (`Sentinel2STACCatalog`,
  `read_s2_l2a_cog`: L2A COGs read in place instead of ~1 GB SAFE
  downloads; honours published `raster:bands` scale/offset, otherwise the
  baseline-04.00 `-1000` DN offset; Planetary Computer and Earth Search
  asset names). Tested against synthetic local GeoTIFFs only.
- **Architecture**: `workflow/adapters.py` holds one `SensorAdapter` per
  sensor (search + acquire); `AnalysisWorkflow.discover/execute` are now
  generic loops instead of `if/elif` chains, and auxiliary providers come
  from `providers.AUXILIARY_PROVIDER_FACTORIES`. `AnalysisGrid.for_aoi`
  (UTM zone of the AOI centre, densified edges) replaces the worker's
  private copy; `AnalysisGrid.for_city` uses the same densified
  projection, so a city grid can now be one pixel row/column larger than
  before - it previously projected only the four corners, which cut off
  the bowed edges of the AOI. `AnalysisGrid.to_geobox()/from_geobox()`
  (new `odc` extra) interoperate with odc-geo/odc-stac.
  `AnalysisResult.save()` writes all cubes plus `request.json`/
  `provenance.json` and is shared by the worker and the new
  `citycube run` CLI command. `downscale.py` became the
  `downscale/` package (`regression`, `gwr`, `consistency`,
  `spatiotemporal`) with public imports unchanged; Random Forest and
  XGBoost are now thin subclasses of a generic `SklearnDownscaler`
  (`fit_sklearn_downscaler` takes any scikit-learn-style regressor), and
  `validate_downscaler` predicts once instead of up to three times.
- **Worker**: at most `MAX_CONCURRENT_JOBS` (default 1) jobs execute at
  once; later submissions wait as `PENDING`. Previously every job got its
  own thread immediately, which is how two memory-heavy runs could
  overlap.
- **CI** now also installs the `cloud` and `odc` extras, so the Zarr
  round-trip and GeoBox tests run instead of being skipped.

## New from this pass (real-data validation, Sentinel-3 flip, Phase 4)

- **CRITICAL - every area-gridded Sentinel-3 LST cube was flipped
  north-south.** `grid_l2_lst(method="area")` - the default, and the path
  `AnalysisWorkflow.execute()` always takes - built its target cells with
  row 0 at the *bottom* of the grid but stored them under `grid.y`, which
  runs top to bottom. Confirmed with a synthetic north-is-warmer swath
  (`method="point"` came out correct, `"area"` reversed), fixed, and
  pinned by `tests/test_sentinel3_area.py`. An existing test
  (`test_combine_sentinel3_regrids_each_acquisition_before_concatenating`)
  had been fitted "empirically" to the flipped output and was corrected.
  **Consequences**: every thermal cube produced before this fix pairs
  upside-down LST with correctly oriented Sentinel-2/Landsat/ECOSTRESS
  predictors. All earlier real-data downscaling numbers (the "Benchmark
  Results" runs in `docs/downscaling.md`, the STARFM real-reference check,
  the committed `*_executed.ipynb` outputs) are invalid and must be
  re-run; the near-zero/negative correlation of "Run 2" is the expected
  symptom of a flipped target.
- **CRITICAL - Sentinel-3 clouds were mostly not masked.** The reader
  derived `cloud_mask` from `cloud_in` using only its `gross_cloud`/
  `thin_cirrus`/`medium_high`/`fog`/`stratus` bits; on real Berlin August
  daytime scenes most clouds only trip other tests (e.g. `visible`), so
  cloud tops at 5-15 degC passed as "valid" LST (one fully clouded scene
  had a 15 degC median and 100 % valid pixels). ESA's Bayesian mask
  (`bayes_in`, `single_moderate`) separates them cleanly on the same scenes
  (clear vs cloudy medians 25.6/12.5, 32.6/27.2, 31.8/18.8 degC). The reader
  now prefers `bayes_in` (exposed as `bayes_flags`) and falls back to the
  old `cloud_in` subset only when it is absent; after the change every
  scene's 5th percentile rose from 5-11 degC to 20-30 degC. Together with
  the flip above, this means earlier thermal cubes were both upside down
  and cloud-contaminated.
- **Sentinel-3 area gridding is now vectorized**: the per-pixel version
  built a shapely polygon for every pixel of the full swath (~1.8 M per
  product) and re-read each variable once per footprint intersection; a
  one-week Berlin run sat in it for 40+ minutes. Only pixels whose
  footprint can reach the grid are kept, polygons/intersections are built
  in bulk (shapely 2) and accumulated with `np.add.at`. Output is
  identical to the original (after the orientation fix) on a rotated swath
  with NaNs and flags. Sentinel-3 products are also gridded one at a time
  (a generator), so raw swaths are not all held at once.
- **Boolean masks after regridding**: `harmonize_spatial` turned masks
  into float with NaN outside a source footprint, and `NaN.astype(bool)`
  is True - no-data cells became "valid". Masks now stay boolean with
  False outside (found by the real-scene validation below).
- **Cloud-native readers validated on real scenes**
  (`scripts/validate_cloud_native_real_reference.py`, peak RSS ~0.5 GB):
  `read_s2_l2a_cog` reproduces the SAFE's own 10 m bands bit for bit
  (max |diff| 0); against the `cdse_safe` path B11/B12 agree exactly and
  SCL agrees on 100 % of 20 m pixels; B02/B03/B04 differ by +0.0028/
  +0.0018/+0.0013 reflectance, which is exactly ESA's R20m product minus
  the mean of its R10m pixels (Sen2Cor processes 20 m separately) - a
  property of the SAFE path, not a reader bug. `read_s1_rtc_cog`: STAC
  metadata confirms nodata -32768 and linear gamma0; real medians are
  physically right (Mueggelsee water -22 dB VV, Tiergarten -8 dB,
  Alexanderplatz -2 dB, VH < VV on 99.7 % of pixels). Both options stay
  opt-in (they change the data source), but are no longer unvalidated.
- **Validation split**: `blocked_spatiotemporal_split` gained
  `block_size` (the default single-pixel diagonal leaves every held-out
  cell's neighbours in training) and now rejects an unsorted time index -
  before the time-ordering fix its "latest acquisitions" holdout was an
  arbitrary subset. `blocked_calibration_split` returns the matching
  calibration cells.
- **Spatially varying uncertainty**: `fit_conformal_downscaler` /
  `ConformalDownscaler` (split conformal, normalized by Random Forest tree
  spread via `ensemble_spread`); `validate_downscaler` reports
  `interval_coverage`/`interval_mean_width`.
- **Dask-aware regridding**: `reaggregate_to_target` and the CRS
  reprojection in `harmonize_spatial` go through `xr.apply_ufunc(...,
  dask="parallelized")`, so chunked inputs (e.g. `open_zarr`) stay lazy.
  Not made the default for eager inputs: the real city-scale runs above
  peak at ~0.5 GB, so there is nothing to gain there.
- **Multi-city benchmark with pyDMS**: `scripts/benchmark_multicity.py`
  (five preset cities x two periods, daytime passes only, 5 km block
  holdout, pyDMS as external reference, conformal coverage). See
  "Benchmark results (multi-city)" in `docs/downscaling.md`.
- **Docker**: `worker/Dockerfile` now installs the `landsat` extra, which
  Landsat, `pc_rtc` and `stac_cog` all need to sign Planetary Computer URLs.

## New from this pass (Sentinel-3 processing review, storage, terrain)

Prompted by asking why Sentinel-3 had been processed wrongly and slowly,
compared against ESA's SLSTR Land Handbook, ESA's Sen-ET reference scripts
and other libraries.

- **Quality screening now follows the handbook.** Besides the Bayesian
  cloud mask (previous pass), `QualityPolicy` rejects `confidence_in`
  `unfilled`/`cosmetic`/`duplicate` pixels (8-17 % of otherwise clear
  pixels on real Berlin scenes; a duplicate counted twice in the area
  average) and views beyond 45 degrees (`sat_zenith`, interpolated from the
  16 km tie-point geometry using each file's `track_offset`/`resolution` -
  which real files store as the *string* `'[ 16000 1000 ]'`, unlike the
  first synthetic test). `cloud_buffer_pixels` optionally dilates clouds.
- **The reader leaked every extraction**: each read unzipped the whole
  product into a new temporary directory that was never removed - 358
  directories, 23 GB in `/tmp` on this machine. It now extracts only the
  needed files into a temporary directory that is deleted after loading
  (Sentinel-2 SAFE reads had the same leak and are fixed the same way),
  and crops the swath to the AOI (a Berlin read is ~50 x 48 pixels of
  1200 x 1500; 0.4-1 s per product).
- **Partial downloads**: `CDSEDownloader.download_files` fetches single
  product files through OData `Nodes`; a live Berlin product came down as
  17.2 MB instead of 73.5 MB and read identically.
- **AOI subsets and raw retention** (`raw_retention`, see
  `workflows.md`): Sentinel-3/5P/2-SAFE products are stored as AOI subsets
  and the originals deleted by default. Sentinel-5P (a whole-orbit OFFL
  product is ~625 MB) is cropped to the AOI and de-duplicated per orbit
  (NRTI and OFFL copies of one orbit were both downloaded), and its
  adapter now reads the gas it searched for (it always read NO2).
- **Terrain predictors** (`terrain_predictors=True`, `sensors/terrain.py`):
  Copernicus GLO-30 elevation, slope, aspect and solar-illumination
  `cos_incidence`, as in Sen-ET. Verified live over Mexico City (valley
  2226-2326 m, sierras to 3639 m, no gaps, 16 s, 221 MB).
- **Independent cross-check**: a test compares Sentinel-3 area gridding
  with pyresample's nearest-neighbour resampling (new `dev` dependency);
  the old flipped implementation scores a correlation of -0.69 against
  it, so this test alone would have caught the flip.
- **Linear models clip predictors to their training range** and flag it
  in `downscaled_extrapolation`: EVI outliers at 100 m drove one real OLS
  run to a 737 K RMSE.
- **Pre-commit review fixes**:
  - *Reprojection flipped ascending-y sources north-south* (pre-existing):
    `harmonize_spatial`'s CRS path builds its transform from the top edge
    but passed rows in array order, so every `grid_s5p` output (ascending
    latitude) - and any latitude-ascending ERA5/CAMS file - landed upside
    down on the projected predictor grid. Sources are now sorted north-up
    first; a test covers both latitude orders. Earlier Sentinel-5P results
    on a projected grid are affected.
  - *AOI-subset cache race*: parallel product workers each created their
    own `AssetCache`, whose per-instance lock let manifest
    read-modify-writes interleave through one shared `.part` file; a
    reproduction lost entries or crashed in 5 of 5 runs. Locks are now
    shared per manifest and every write uses a unique temporary file.
  - *Empty AOI crop*: a Sentinel-3 product whose catalogue footprint
    intersects the AOI but whose pixels do not crashed the whole
    acquisition; it now grids to an all-missing slice.

## Explicitly deferred (already documented elsewhere)

- Downscaling roadmap items 5-6 (kriging/land-use regression for OpenAQ
  conditional on station cross-validation results, a deep model only after
  baselines are stable). Item 2's first real benchmarking pass is done (see
  "New from this pass" above); per-model hyperparameter tuning and a
  larger/longer real comparison remain open. Detail: `docs/downscaling.md`,
  "Implementation Roadmap" and "Benchmark Results".
- **Re-run every real-data result produced before the Sentinel-3 flip
  and cloud-mask fixes**: `scripts/benchmark_downscalers.py`, the
  STARFM/ESTARFM real-reference scripts and the committed
  `*_executed.ipynb` notebooks (the multi-city benchmark already was).
- **Calibrate conformal intervals on scene-corrected residuals**: on real
  data the current coarse-residual calibration covers 0.42-0.93 instead of
  0.90 (see `downscaling.md`, "Benchmark Results (multi-city)").
- **Expose per-scene downscaling in the library**: the benchmark shows
  per-scene training anchored to the coarse scene is the best protocol; it
  currently lives only in `scripts/benchmark_multicity.py`.
- FSDAF (a more capable but unimplemented STARFM/ESTARFM sibling) and
  running ESTARFM against real data (currently unit-tested only) - both
  matter most for exactly the case single-pair STARFM handles badly: large
  or heterogeneous change between t0 and t1. Detail: `docs/downscaling.md`
  roadmap item 4.

## Verification performed for this pass

- `worker/frontend`: `npm install && npm run build && npm run lint` - clean.
- `worker`: `pytest -q` - 22 passed (0 skipped, since the frontend build
  above made `dist/` exist for the mount-ordering test).
- Live process run: `uvicorn app.main:app` on :8100 - confirmed `GET /`
  serves the SPA's `index.html` while `GET /jobs`/`GET /health` still
  resolve as the API, with and without auth as expected.
- Real job end to end: submitted a Sentinel-3/Berlin request through the
  bare API (the same contract the UI's Submit page uses), watched it go
  `PENDING` → `RUNNING` → `SUCCEEDED` with `name`/`created_at`/
  `request_params` persisted correctly, downloaded the result and confirmed
  a real `lst` variable inside a valid `thermal_cube.zarr`.
- `docker compose build worker` + a container smoke test repeating the same
  `GET /`, `/health`, `/jobs` checks against the built image.
- Real ECOSTRESS end-to-end: live CMR-STAC search over Berlin found a real
  overpass, then `read_ecostress_lst` against the real asset (with a real
  `EARTHDATA_BEARER_TOKEN`) returned 9-15 degC at a nighttime pass -
  physically correct, not just "didn't crash."
- GWR: a synthetic two-cluster dataset with opposite NDVI-LST slopes -
  confirmed the fitted model recovers each cluster's own slope at its own
  location, and that a global linear model on the same data collapses
  toward a much smaller magnitude (can't represent both at once).
- STARFM: a synthetic two-surface grid with a coarse-scale change localized
  to only one surface - confirmed the similarity filter keeps that change
  from bleeding into the other surface's own prediction despite a window
  spanning both, and that mismatched-grid coarse inputs are reindexed onto
  the fine grid correctly. A real bug was caught here too:
  `0.0 * NaN == NaN`, not `0.0` - a padded/masked window cell being NaN was
  silently poisoning the weighted sum even at weight 0, giving NaN
  predictions everywhere the window touched a border; fixed by zeroing
  masked cells before the weighted multiply, not after.
- `citycube`'s full suite re-run after every change in this pass:
  56 passed, 5 skipped (up from 52/5 - the 4 new GWR/STARFM tests).
- Full downscaler benchmark (`scripts/benchmark_downscalers.py`): a real
  live fetch (Berlin, 2026-08-20 to 2026-08-27, 12 Sentinel-3 + Sentinel-2
  products, 100 m predictor resolution) through all five regression models
  plus the STARFM sanity check, on one real `blocked_spatiotemporal_split`
  holdout - not synthetic data, and not a random pixel split. A real bug
  was caught building it: `cube.time` is not chronologically sorted, and at
  least one real acquisition in the week had zero finite pixels (fully
  cloud-obscured) - a naive first/last-time-index pick for the STARFM
  sanity check silently compared against an all-NaN observation. Fixed by
  sorting first and explicitly selecting the first and last acquisitions
  that have any finite pixel at all.
