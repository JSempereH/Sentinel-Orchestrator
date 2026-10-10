# Changelog

All notable changes to `citycube` and the `worker/` service.
Detailed evidence for each change (real runs, bugs found) is in
`dev/history.md`; current limitations are in `docs/limitations.md` and `dev/roadmap.md`.

## Unreleased

### Changed (breaking)
- The package is renamed from `sentinel_analysis` to `citycube`
  (`pip install citycube`, `import citycube as cc`, command line `citycube`).
  The build-time commit variable is now `CITYCUBE_GIT_COMMIT`. It covers more
  than Sentinel missions (Landsat, ECOSTRESS, ERA5, CAMS, OpenAQ, terrain), and
  its core is the city or polygon cube.
- `AnalysisResult.predictor_cube` is replaced by `AnalysisResult.predictors`, one
  fine-resolution cube per sensor on its own acquisition times
  (`predictors/<sensor>.zarr` on disk). The merged cube outer-joined every sensor's
  times and was mostly NaN fill; a Berlin run's peak memory fell from over 6 GB to 0.8 GB.
- `downscale_per_scene(cube, fine_predictors, ...)` takes one sensor's cube
  (`result.predictors["sentinel2"]`) and static terrain from `terrain=`.
- Per-scene downscaling defaults to `correction="smooth"` (no coarse-cell steps) and
  `mask_unobserved=True` (no clear-sky values presented under clouds).
- `shapely>=2.0` is a core dependency.

### Added
- `model="local_trees"`: a pyDMS-style downscaler, a global model plus local models in
  moving windows of coarse cells, bagged trees with linear leaves, blended by residual.
  It is now the default model: on Berlin it had the lowest blocked-holdout error and beat
  the repeated 1 km map against Landsat, where the Random Forest did worst.
- Workflow downscaling streams each scene to `downscaled.zarr`, so memory holds one scene.
- Product selection by AOI coverage computed from catalogue footprints
  (`min_aoi_coverage`, default 0.3), with tiles of one pass counted together.
- Sentinel-3 clear-sky probe (`min_clear_fraction`): ~14 MB per candidate measures the
  share of the AOI that is clear and within the 45 degree view-angle cut, before the LST
  is downloaded; cached subsets are judged from their own data; rejects are in
  `provenance["probed_out"]`.
- Polygon AOIs: `AOI.from_geojson`, `AOI.from_geometry`, `AOI.from_dict`/`to_dict`;
  the polygon drives coverage, the cloud probe and masks the downscaled output.
  The worker accepts `aoi.geometry` and its map sends non-rectangular drawings.
- Zonal statistics: `ZoneSet.from_geojson`, `zonal_statistics`, `ZoneSet.to_geojson`,
  plus `aoi_mask` and `clip_to_aoi`.

## 0.2.0 - 2026-10-08

### Added
- Per-scene LST downscaling as a workflow stage: `AnalysisRequest(downscale=DownscaleSpec(...))`
  and `downscale_per_scene`, the protocol the multi-city benchmark found most accurate.
- `thermal_overpass="day"|"night"` filters Sentinel-3 and ECOSTRESS passes before ranking.
- `on_product_error="skip"` (default): a product that fails to download or read is left
  out and listed in `provenance["failed_products"]` instead of aborting the run.
- Request size estimate and limits (`estimate_request`, `RequestLimits`); `execute()`
  rejects oversized requests before downloading; `citycube plan` shows the estimate.
- `check_credentials()` / `citycube check-credentials`: live authentication
  against CDSE, Earthdata (with token expiry), CDS, ADS and OpenAQ.
- `scripts/canary.py` (`make canary`): daily smallest real end-to-end run for cron.
- Every result's provenance records the package version and git commit (`build_info()`).
- Worker: jobs run in isolated child processes with cancellation (`POST /jobs/{id}/cancel`),
  a timeout, out-of-memory containment and a per-job log (`GET /jobs/{id}/log`);
  `DELETE /jobs/{id}`; retention of finished jobs; free-disk guard (HTTP 507);
  `GET /health/ready`; per-job metrics; JSON logs; request size limits (HTTP 422).
- Worker UI: cancel, delete, log viewer, metrics, readiness report, and the new
  request options (daytime passes, terrain, downscaling).
- Container: pinned dependencies (`worker/requirements.lock`), unprivileged user,
  `HEALTHCHECK`, restart policy, memory cap, init process, commit baked in.
- CI: frontend lint and build; worker tests installed from the lock.

### Fixed (found by running the guide notebooks on real data)
- The predictor cube outer-joined hourly ERA5 times onto every sensor variable:
  a 3-week Berlin run needed over 6 GB for a 0.1 GB request (now 0.8 GB). Auxiliary
  sources stay in the fused cube and `result.auxiliary`, regridded to the thermal grid.
- Datetime variables with gaps (`*_matched_time`) were written to Zarr and NetCDF but
  could not be read back; times are now encoded exactly to the microsecond.
- `grid_s5p` binned each TROPOMI pixel into the one 0.01 degree cell holding its centre,
  leaving most of the grid empty; pixels now fill their footprint (`footprint_radius_km`).
- Sentinel-1 indices lacked the metadata contract, so any fusion with Sentinel-1 failed.
- The CAMS forecast request asked for 24 hours x 41 lead times per day; forecasts now
  read as one valid-time axis (`forecast_to_time`), and a reanalysis request for recent
  dates fails with a hint to use the forecast dataset.
- Auxiliary sources are acquired before satellite products, so their errors surface at once.
- ERA5-Land requests for variables it does not have fail before queueing.
- Regridding failed when a small AOI fell inside a single row or column of a coarse
  source (one 0.25 degree ERA5 row over a city).
- `collocate_stations` required satellite and station times to be identical, which they
  never are; it now pairs the nearest observation within a tolerance and reports stations
  outside the cube or without matches instead of failing.
- Products without cloud cover (Sentinel-1, Sentinel-5P) were truncated to the first days
  of the period; they are now sampled evenly through it. Sentinel-5P skips night-side orbits.
- Invalid AOIs (inverted, out of range, non-numeric) are rejected on creation; an inverted
  AOI used to reach the worker as an HTTP 500. `sensors` given as a string is rejected.

- `execute(max_workers>1)` (the worker's default of 4) could segfault: HDF5 is not thread
  safe, and Sentinel-3/5P reads and AOI-subset writes ran concurrently. All threaded
  NetCDF I/O now holds `storage.NETCDF_LOCK`; downloads stay parallel.

### Security
- Worker job logs no longer record the CDSE OAuth client id, which `openeo` logs at INFO.
- Concurrent downloads of one job result no longer share (and delete) one temporary zip.
- Frontend: `maplibre-gl` 5 to 6.13 (critical XSS advisory in its HTML sanitizer).
- Replaced deprecated `rasterio` and `xarray` calls before they become errors.

### Added (continued)
- `write_netcdf` / `netcdf_safe`: NetCDF output of cubes with provenance attributes.
- Downscaling: `correction="smooth"` (default in `DownscaleSpec`) removes 1 km steps from
  the coarse correction; `mask_unobserved=True` leaves cloud-covered cells empty instead of
  showing clear-sky predictions; downscaled variables carry units.
- Guide notebooks rebuilt as a numbered series (01 to 07) executed on real data.

### Changed
- `docs/roadmap.md` now holds the current state; the chronological log moved to
  `docs/history.md`.
- `sentinel1_backend="hyp3_rtc"` and `"s1ard"` emit a `UserWarning` (experimental).
- `CoarseConsistentDownscaler` keeps the base model's `downscaled_extrapolation` flag.

## 0.1.0

Initial multisensor workflow, downscaling toolkit and worker; see `docs/history.md`.
