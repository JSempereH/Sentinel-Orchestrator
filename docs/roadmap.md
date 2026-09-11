# Production-readiness roadmap

Single authoritative "what's left" list for this platform
(`sentinel_analysis`, `worker/`). Detail that already
has a good home elsewhere is linked, not duplicated. "Production ready"
means this platform's own bar - durable and observable self-hosted,
single-trusted-user tooling, not multi-tenant/SaaS hardening (see
`docs/platform.md`).

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
- **Tests**: `sentinel_analysis` 56 passed / 5 skipped (4 need real local
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

## Explicitly deferred (already documented elsewhere)

- Downscaling roadmap items 5-6 (kriging/land-use regression for OpenAQ
  conditional on station cross-validation results, a deep model only after
  baselines are stable). Item 2's first real benchmarking pass is done (see
  "New from this pass" above); per-model hyperparameter tuning and a
  larger/longer real comparison remain open. Detail: `docs/downscaling.md`,
  "Implementation Roadmap" and "Benchmark Results".
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
- `sentinel_analysis`'s full suite re-run after every change in this pass:
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
