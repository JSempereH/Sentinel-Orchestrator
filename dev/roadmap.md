# Roadmap and limitations

Current state of `citycube` and `worker/`, as of 2026-10-08. This
page is kept up to date rather than appended to; the evidence behind each
status (real runs, bugs found, decisions) is in the
[change history](history.md). "Production ready" here means durable,
observable, self-hosted tooling for a single trusted user, not a
multi-tenant service (see `platform.md`).

## Status by component

| Component | Status | Evidence |
|---|---|---|
| Sentinel-3 SLSTR LST | Validated on real data | Orientation, Bayesian cloud mask and handbook quality flags fixed and checked on real Berlin scenes; area gridding cross-checked against pyresample in the tests |
| Sentinel-2 L2A (`cdse_safe`, `stac_cog`) and L1C | Validated on real data | COG reads match the SAFE bit for bit at 10 m; baseline-04.00 radiometric offset applied |
| Sentinel-1 `snap` | Validated on one real scene | ~2.6 min and ~11 GB RAM per scene |
| Sentinel-1 `pc_rtc` | Validated on real data | Physically plausible backscatter per land cover |
| Sentinel-1 `hyp3_rtc` | Unit-tested only | Never run against a real HyP3 submission |
| Sentinel-1 `s1ard` | Broken | Bug in the external `spatialist` library; not fixable here |
| Sentinel-5P | Validated on real data | AOI crop, per-orbit de-duplication, gas follows the request |
| Landsat 8/9, ECOSTRESS | Validated on real data | Authenticated reads with physically correct values |
| Terrain (GLO-30) | Validated on real data | Mexico City run |
| ERA5, CAMS, OpenAQ | Validated end to end | `scripts/smoke_e2e.py` |
| Per-scene LST downscaling | Benchmarked on real data | 5 cities x 2 fortnights: ~2.8-3.2 K RMSE against 4.6 K for the no-skill reference (`downscaling.md`); the library stage also runs end to end in the `--full` canary |
| `local_trees` downscaling (default) | Validated against Landsat once | Berlin, August 2026: lowest blocked-holdout RMSE (2.04 K); beats the repeated 1 km map on one of two same-morning Landsat scenes and ties on the other (`downscaling.md`) |
| AOI coverage and Sentinel-3 clear-sky probe | Validated on real data | Berlin: kept scenes average 92 % usable cells against 45 % unprobed (notebook 02) |
| Polygon AOIs and zonal statistics | Validated on real data | Official Berlin district boundaries (notebook 08) |
| STARFM | Validated once on real data | Against an independent Landsat scene: 2.7 K RMSE at full resolution over a 31-day gap (`downscaling.md`) |
| GWR | Real-data benchmark only | 10 % coverage under a blocked holdout (`downscaling.md`, Run 4) |
| ESTARFM | Unit-tested only | No real-data run yet |
| Conformal intervals | Not calibrated on real data | Empirical coverage 0.42-0.93 for a 0.90 target |
| Worker API, UI and container | Validated end to end | Real job submitted, run and downloaded, in a venv and in Docker; process isolation, cancellation, timeout and OOM handling tested with real child processes |
| Guide notebooks | Executed on real data | Eight notebooks (01 to 08), all cells executed, outputs audited for errors and secrets (`guides.md`) |
| Security review | Done (2026-10-09) | `bandit`, `pip-audit` (no known vulnerabilities in the pinned lock), `npm audit` (critical `maplibre-gl` advisory fixed; dev-server-only `vite` advisories remain), secret scan of the tree, notebook outputs and git history, malformed-input tests of the API |
| Operations | In place | Retention, disk guard, `/health/ready` with live credential checks, per-job logs and metrics, daily canary, pinned container dependencies (`platform.md`, "Production operation") |

Test suites: `citycube` 187 passed / 6 skipped (the skips need
real local products or optional extras), `worker` 41 passed. Both use
synthetic data and mocks; real-data checks are the manual scripts under
`scripts/` and are not part of CI.

## Known limitations

### Scientific

- **Downscaling is validated for land-surface temperature only.** The
  models accept any `target`, but no other variable has a physical
  relation to the optical predictors that has been checked.
- **Uncertainty intervals are not trustworthy yet**: conformal intervals
  are calibrated on raw coarse residuals, which include each day's level
  shift.
- **The benchmark is small**: five cities, two two-week periods. Cloudy
  seasons (Guadalajara and Lagos in August) gave too few clear daytime
  scenes to evaluate at all.
- **Results from before the Sentinel-3 orientation and cloud-mask fixes
  are invalid.** The old notebooks were replaced by the guide notebooks
  01 to 08, executed after the fixes; the Berlin benchmark and the STARFM
  real-reference check were re-run too (`downscaling.md`, "Run 4").
- FSDAF is not implemented and there is no deep-learning model, both
  deliberately (see "Next steps").

### Functional

- **Grids are rectangular over the AOI's bounding box**: a polygon AOI
  masks results, coverage and the cloud probe, but the cubes still cover
  (and cost) its whole box. AOIs crossing the antimeridian are rejected, and
  grids use the UTM zone of the AOI centre, so very wide AOIs are distorted
  at their edges.
- **Fusion matches each target time to one observation per sensor** (the
  nearest within the temporal tolerance); there is no compositing across
  several observations, except the optional Sentinel-2 composite.
- **`thermal_overpass` defaults to `"any"`** for backwards compatibility;
  set it to `"day"` for downscaling.
- **Stored AOI subsets are keyed by product, AOI and
  `SUBSET_FORMAT_VERSION`**: a reader change that alters what is stored
  must bump that version, and a larger AOI later means downloading again.
- **Sentinel-5P keeps whole-orbit downloads** of about 600 MB each even though only an AOI subset is stored, and night-side orbits (no NO2 retrieval) are downloaded and then found empty.
- **Downscaling's smooth correction conserves the observation to within a few hundredths of a kelvin** rather than exactly; use `correction="block"` when exact conservation matters more than the absence of 1 km steps.
- **Sentinel-3 clouds over the AOI are only measured when asked**: the
  catalogue's cloud cover is per granule. Coverage of the AOI is always
  computed from footprints (`min_aoi_coverage`), but the clear-sky probe
  costs ~14 MB per candidate and runs only with `min_clear_fraction > 0`.

### Operational

- **Everything is processed in memory.** `execute()` rejects requests
  whose estimated cube size exceeds `RequestLimits` (16 GB by default, 8 GB
  in the worker) before downloading, but the estimate is an order of
  magnitude, peak memory is up to about twice it, and external processors
  are not included (a local SNAP run needs ~11 GB on its own).
- **One failed product no longer aborts a run** (`on_product_error="skip"`
  records it in the provenance), but a sensor whose products *all* fail
  still stops the run.
- **The worker is single-user**: one shared bearer token, no TLS of its
  own, SQLite job store, one job at a time by default. Keep it on a LAN or
  VPN. Multi-user use would need per-user auth, a real queue and object
  storage for results; none of that is planned.
- **Runs cannot resume**: a worker restart fails the running job, which
  must be resubmitted (already-stored AOI subsets are not shared between
  jobs, so it downloads again).
- **External credentials expire** (CDSE OAuth clients, the Earthdata
  bearer token, which lasts 60 days) and providers enforce quotas.
  `/health/ready` and the daily canary report failures and tokens expiring
  within 7 days, but someone still has to renew them.

## Next steps, in priority order

1. **Calibrate conformal intervals on scene-corrected residuals**, then
   re-check coverage on the multi-city benchmark.
2. **Validate `local_trees` more widely**: more same-morning Landsat
   scenes, more cities and seasons; two scenes in one city are not enough
   to call it a measured gain.
3. **Run the unvalidated paths once against real data**: a real HyP3
   submission and `scripts/validate_estarfm_real_reference.py`. Decide on
   FSDAF only after ESTARFM's real behaviour is known.
4. **Schedule the canary** (cron or a systemd timer, see `platform.md`) on
   the machine that runs the worker, with alerting on a non-zero exit.
5. **Broaden the benchmark** (more cities and seasons) before tuning any
   model further; one AOI over a few weeks has already flipped the model
   ranking once.
6. Kriging or land-use regression for OpenAQ only if station
   cross-validation beats the guarded IDW; a deep model only after the
   baselines and uncertainty are stable.

## Operating rules

- Run at most one live-data script at a time, with `max_workers=1`, on a
  16 GB machine (two concurrent runs froze it twice).
- Never print verbose HTTP traces or request headers while a real token is
  in the environment; check success or failure only (a token leaked this
  way once and had to be rotated).
- If optional extras disappear from the root `.venv`, reinstall them with
  `uv pip install --python .venv -e ".[cdse,auxiliary,optical,landsat,ecostress,ml]"`.
