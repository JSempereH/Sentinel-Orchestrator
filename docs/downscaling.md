# Downscaling and Fusion

This document records the scientific implementation plan for fusing Sentinel
and environmental observations into a defensible high-resolution urban heat
product. The goal is not to make a sharper-looking raster: every fine-scale
prediction must remain traceable to observations, predictors, uncertainty and
the original coarse thermal measurement.

## Evidence From The Literature

The current evidence supports a staged approach rather than jumping directly
to a deep super-resolution model:

- The LST downscaling survey by Zhan et al. identifies the main risks as mixed
  pixels, scale mismatch, non-stationarity, and validation leakage
  ([10.1016/j.rse.2012.12.014](https://doi.org/10.1016/j.rse.2012.12.014)).
- TsHARP is a useful physically interpretable baseline based on the relation
  between temperature and vegetation indices
  ([10.1016/j.rse.2006.10.006](https://doi.org/10.1016/j.rse.2006.10.006)).
- Random forest and other nonlinear models generally improve over simple
  sharpening baselines in heterogeneous areas, but their performance depends
  strongly on feature selection and mixed-pixel handling
  ([10.1016/j.rse.2016.03.006](https://doi.org/10.1016/j.rse.2016.03.006),
  [10.1109/JSTARS.2019.2896923](https://doi.org/10.1109/JSTARS.2019.2896923)).
- Geographically weighted machine learning explicitly models local
  non-stationarity and is a strong urban baseline
  ([10.3390/rs13061186](https://doi.org/10.3390/rs13061186)).
- STARFM/ESTARFM and FSDAF remain important spatiotemporal-fusion baselines.
  They should be benchmarked before any neural model
  ([10.1016/j.rse.2010.05.032](https://doi.org/10.1016/j.rse.2010.05.032),
  [10.1016/j.rse.2020.111973](https://doi.org/10.1016/j.rse.2020.111973)).
- A review of multisource fusion describes the trade-off between spatial
  detail, temporal continuity, radiometric consistency and change detection
  ([10.3390/rs10040527](https://doi.org/10.3390/rs10040527)).
- Deep learning can improve texture and nonlinear relationships, but random
  pixel splits produce optimistic scores. Spatial, temporal and cross-region
  holdouts are required before claiming generalization.

## Recommended Model Ladder

### Stage 0: Measurement and scale contract

- Use Sentinel-3 LST as the coarse thermal target.
- Keep Sentinel-2, Sentinel-1, Sentinel-5P, ERA5, CAMS and OpenAQ as named
  predictors or reference observations.
- Harmonize CRS, pixel support, units, quality masks and acquisition times
  before model fitting.
- Store the coarse target, fine prediction, support mask, predictor names,
  model version and training footprint in the result.

### Stage 1: Auditable baselines

Implement and benchmark these in order:

1. Nearest/area-preserving resampling as a no-skill reference.
2. TsHARP/DisTrad-style index sharpening where assumptions are satisfied.
3. Global linear regression with explicit residual uncertainty.
4. Random forest and XGBoost with feature selection and monotonic sanity
   checks.
5. Geographically weighted regression or a geographically weighted ensemble
   such as MFGWML for urban non-stationarity.

The first production candidate should be a constrained ensemble of stages 3
to 5, not an unconstrained neural super-resolution model.

### Stage 2: Coarse-scale consistency

Every fine prediction must be reaggregated to the Sentinel-3 support. The
training objective and post-processing should penalize disagreement with the
observed coarse LST:

```text
coarse_consistency_error = aggregate(fine_prediction) - coarse_lst
```

The error, tolerance and aggregation operator must be stored and reported.
This prevents a model from generating plausible local texture while violating
the measured thermal energy at the source scale.

### Stage 3: Spatiotemporal fusion

- Use STARFM/ESTARFM or FSDAF as classical references for temporal gap filling
  where compatible high- and low-resolution pairs exist.
- For LST, explicitly test whether reflectance-fusion assumptions transfer to
  temperature. Do not reuse an optical fusion result as thermal truth.
- Add change-aware handling for cloud, land-cover change and heatwave dates.
- Compare `blend-then-index` and `index-then-blend` for optical predictors;
  the choice can change the result in nonlinear indices.

### Stage 4: Deep models, only after baselines

Evaluate residual U-Net or CNN/Transformer hybrids only when there are enough
co-located scenes. Use auxiliary losses for:

- coarse-scale consistency;
- valid-pixel and cloud-mask support;
- temporal stability;
- uncertainty calibration.

Use ensembles, quantile regression or conformal calibration for predictive
uncertainty. A neural model must beat the classical baselines on blocked
holdouts, not only on random pixel splits.

## Predictor Design

The initial urban feature set should include:

- Sentinel-2 reflectance, NDVI, EVI, NDMI, NDBI, MNDWI and cloud support;
- Sentinel-1 sigma0/gamma0, polarization ratios and acquisition geometry;
- Sentinel-5P gas columns with explicit column-to-surface caveats;
- ERA5 meteorology and CAMS composition;
- land cover, imperviousness, elevation and morphology where available;
- OpenAQ station observations plus an interpolation-support mask.

Feature selection must be done inside each training fold. Variables that are
not available at inference time must not enter the training set.

## Validation Protocol

The validation dataset must be split by geography and time, not by random
pixels:

- blocked neighborhoods for spatial generalization;
- held-out dates and seasons for temporal generalization;
- heatwave dates as a stress test;
- independent thermal references such as Landsat or ECOSTRESS when available;
- leave-one-station-out validation for OpenAQ interpolation.

Report RMSE, MAE, bias, correlation, high-temperature error, valid coverage,
coarse-scale conservation error and calibrated uncertainty coverage. Report
metrics separately for vegetation, impervious, water and mixed urban pixels.

## OpenAQ Surface Interpolation

OpenAQ measurements are still preserved in `AnalysisResult.auxiliary['openaq']`
as a station table. When a predictor grid is configured, the workflow also
creates a guarded raster using inverse-distance weighting (IDW):

- matching station observations use the nearest time within the configured
  temporal tolerance;
- distances are computed in the projected analysis CRS;
- cells outside `interpolation_max_distance_m` remain missing;
- cells with fewer than `interpolation_min_neighbors` remain missing;
- at most `interpolation_max_neighbors` nearest stations contribute;
- each interpolated variable receives station-count, uncertainty and valid
  support diagnostics.

The default is intentionally conservative. IDW is a reproducible operational
surface, not proof that pollutant concentrations are spatially smooth. Kriging
or a land-use/meteorology regression should only replace it after variogram
diagnostics and leave-one-station-out validation.

Example request options:

```json
{
  "provider": "openaq",
  "variables": ["NO2"],
  "options": {
    "max_locations": 100,
    "interpolation_method": "idw",
    "interpolation_power": 2.0,
    "interpolation_max_distance_m": 10000,
    "interpolation_min_neighbors": 3,
    "interpolation_max_neighbors": 12
  }
}
```

The fused variable is named `auxiliary_openaq_NO2`; diagnostics are named
`auxiliary_openaq_NO2_station_count`,
`auxiliary_openaq_NO2_interpolation_uncertainty` and
`auxiliary_openaq_NO2_interpolation_valid`.

For station-level validation, call
`OpenAQProvider.validate_leave_one_station_out(stations, "NO2")`. The result
contains global RMSE/MAE/bias, supported coverage and per-station metrics. This
must be reported before replacing guarded IDW with kriging or land-use
regression.

## Implemented Model APIs

The package now exposes `fit_tsharp_downscaler` and
`fit_coarse_consistent_tsharp_downscaler` for the interpretable one-predictor
TsHARP/DisTrad relation. `fit_xgboost_downscaler` and
`fit_coarse_consistent_xgboost_downscaler` are available through the optional
`ml` extra and preserve the same finite-support and coarse-consistency
contracts as the existing Random Forest model. Use
`blocked_spatiotemporal_split` or the compatibility wrapper
`split_spatiotemporal`, and `validate_independent_reference` for external
Landsat/ECOSTRESS comparisons.

`fit_gwr_downscaler`/`fit_coarse_consistent_gwr_downscaler` (also `ml`) add
a geographically weighted regression baseline: unlike every other model
here, which fits one global relation, `GWRDownscaler.predict` refits local
coefficients at every fine pixel from its `max_local_samples` nearest
coarse training points (via a `scipy.spatial.cKDTree`), weighted by
distance through a `bandwidth`/`kernel` ("bisquare" or "gaussian") pair -
the local non-stationarity a single global slope cannot represent (a park's
NDVI-LST relation is not a city's). `residual_std` is a genuine leave-one-out
cross-validated residual, not the in-sample residual the other models
report, since a naive in-sample GWR residual would partly explain each
point using itself. This is markedly slower than the other downscalers for
large grids (one small weighted least-squares solve per fine pixel) - a
documented tradeoff, not an oversight.

`fuse_starfm` implements STARFM (Gao et al. 2006) as a classical
spatiotemporal-fusion baseline: given a fine observation at t0 and a coarse
change between t0 and t1 (on the same or a different, coarser grid - it
reindexes onto the fine grid itself), it predicts a fine map at t1 by
weighting spatially-nearby, spectrally-similar pixels' own observed change.
This is the single fine/coarse-pair form most open STARFM implementations
use, not the full multi-pair ensemble from the original paper.

`fuse_estarfm` implements ESTARFM (Zhu et al. 2010) on top of the same
window/similarity machinery (`_make_windows`, `_spatial_weight_grid`,
shared with `fuse_starfm`). The real difference from STARFM is not just
"one more pair" - it's *how* a pair's coarse change becomes a fine
prediction: STARFM adds the raw coarse difference directly (implicitly
assuming a 1:1 fine/coarse relationship); ESTARFM fits a local linear
regression between fine and coarse values within each pixel's window
(`_local_linear_regression`, a closed-form per-pixel OLS over the window,
vectorized) and uses that pair's own *slope* to convert its coarse change -
representing a window where the fine/coarse relationship's slope, not just
its offset, differs between two surfaces, exactly what STARFM's raw
difference cannot. The two bracketing pairs' predictions are combined
weighted by each pair's own regression's residual variance, not averaged
blindly; a pair with no local coarse variance to regress against (e.g. a
uniform patch) falls back to STARFM's own identity-slope behavior rather
than being dropped. FSDAF remains an unimplemented open follow-up.

## Temporal Berlin Notebook

`notebooks/berlin_multisensor_downscaling.ipynb` demonstrates the complete
Berlin workflow over a three-day window. It discovers and downloads Sentinel-3
LST, Sentinel-2 predictors, optional Sentinel-5P columns and OpenAQ stations,
then builds the fused cube, trains the coarse-consistent Random Forest baseline,
and renders an interactive time slider with observed and modelled map layers.

The default notebook configuration uses Sentinel-3 and Sentinel-2 to keep a
fresh run practical. Enable Sentinel-5P explicitly when the selected dates have
valid TROPOMI pixels over Berlin:

```bash
SENTINEL_NOTEBOOK_LIVE=1 \
SENTINEL_NOTEBOOK_SENSORS=sentinel3,sentinel2,sentinel5p \
uv run --extra notebook --extra ml --extra auxiliary --extra optical \
  jupyter notebook notebooks/berlin_multisensor_downscaling.ipynb
```

The date range, product limit, predictor resolution and output directory are
controlled by `SENTINEL_NOTEBOOK_START`, `SENTINEL_NOTEBOOK_END`,
`SENTINEL_NOTEBOOK_MAX_PRODUCTS`, `SENTINEL_NOTEBOOK_RESOLUTION_M` and
`SENTINEL_NOTEBOOK_OUTPUT`. Generated NetCDF cubes and the static dashboard
are written to the selected output directory. A missing or low-quality
Sentinel-5P overpass is reported and skipped; it must not be interpreted as a
zero concentration field.

## Benchmark Results

`scripts/benchmark_downscalers.py` fetches one real, live Sentinel-3 LST +
Sentinel-2 predictor cube over Berlin and evaluates every baseline in
`downscale.py` on a `blocked_spatiotemporal_split` holdout (spatial +
temporal, not a random pixel split - see "Validation Protocol" above) via
`validate_downscaler`. Run it with:

```bash
uv run --extra cdse --extra optical --extra ml python scripts/benchmark_downscalers.py
```

### Run 1: one week (first pass)

`BENCHMARK_START=2026-08-20`, `BENCHMARK_END=2026-08-27`, 12 Sentinel-3 +
Sentinel-2 products each, 100 m predictor resolution, a 22x25-cell coarse
grid, `max_workers=2`:

| Model | RMSE (K) | MAE (K) | Bias (K) | Correlation | Validation n |
|---|---|---|---|---|---|
| OLS | 5.41 | 4.29 | 0.93 | 0.14 | 498 |
| TsHARP (NDVI) | 5.54 | 4.38 | 0.50 | 0.10 | 498 |
| GWR (5x bandwidth) | 7.11 | 5.96 | 2.34 | 0.33 | **51** |
| STARFM (sanity check) | 7.20 | 6.11 | - | - | - |
| Random Forest | 8.96 | 7.71 | 3.96 | 0.21 | 498 |
| XGBoost | 9.16 | 7.85 | 3.85 | 0.20 | 498 |

Reading this run alone would suggest OLS/TsHARP clearly beat the tree
ensembles. Run 2 below shows that conclusion did not survive more data.

### Run 2: four weeks, coverage-aware, two GWR bandwidths, tuned RF/XGBoost

`BENCHMARK_START=2026-08-01`, `BENCHMARK_END=2026-08-27` (24 products/sensor,
`max_workers=1` - this machine froze twice running two such live fetches
concurrently at `max_workers=2`; serial processing is slower but safe, see
`docs/roadmap.md`). Fused cube: 24 real coarse observations, 664 validation
pixels:

| Model | RMSE (K) | MAE (K) | Bias (K) | Correlation | Coverage |
|---|---|---|---|---|---|
| XGBoost (default) | 7.255 | 6.355 | 3.547 | 0.015 | 100% |
| XGBoost (regularized: depth 3, lr 0.1, 100 trees) | 7.262 | 6.366 | 3.754 | 0.026 | 100% |
| TsHARP (NDVI) | 7.294 | 6.365 | 3.740 | -0.092 | 100% |
| OLS | 7.318 | 6.413 | 3.892 | -0.007 | 100% |
| Random Forest (default) | 7.327 | 6.399 | 3.688 | 0.017 | 100% |
| Random Forest (100 trees) | 7.352 | 6.422 | 3.723 | 0.016 | 100% |
| GWR (5x bandwidth) | 7.591 | 6.614 | 3.171 | 0.146 | **11.6%** |
| GWR (15x bandwidth) | 7.605 | 6.621 | 3.173 | 0.140 | **11.6%** |
| STARFM (sanity check) | 5.923 | 5.825 | - | - | n/a |

What changed, and what it actually means:

- **The stark Run 1 gap disappeared.** Every regression model now lands
  within 0.1 K of every other (7.255-7.352) - XGBoost's plain defaults are
  now marginally *best*, not worst. This confirms Run 1's "simple models
  win" conclusion was itself a small-sample artifact, exactly the outcome
  the "one real run... not a statistically powered conclusion" caveat
  warned about - not a case for trusting Run 2's ranking as final either,
  just a demonstration of how much a single week's noise can move these
  numbers.
- **Correlation collapsed to near zero, even negative** (TsHARP: -0.092,
  OLS: -0.007) despite a *lower* RMSE than Run 1's "worse" models.
  Low/negative correlation with a moderate RMSE means these models are
  close to just predicting something like the mean over this longer,
  presumably more thermally-varied four-week window - real, weak
  predictive skill, not a bug. A longer window is not automatically a
  "better" benchmark; it can just as easily be a harder one.
- **Widening GWR's bandwidth 3x (5x -> 15x grid spacing) did not move
  coverage at all** (11.6% both times, rmse within 0.02 K) - contrary to
  the natural assumption that a wider bandwidth alone fixes the coverage
  problem Run 1 flagged. Something other than raw bandwidth is capping it
  in this range: likely `min_local_samples=10` still not being met for
  most held-out cells, or the specific geometry of
  `blocked_spatiotemporal_split`'s checkerboard-style spatial holdout
  meaning most held-out cells simply have no *training* cell within even
  15x spacing. This is a real, useful negative result - the fix an
  untested assumption would have reached for does not actually work here,
  and whoever tunes this next should check `min_local_samples` and the
  holdout geometry before reaching for an even wider bandwidth.
- **STARFM's sanity check improved** (rmse 5.923, still `within_tolerance=
  False`) and is now numerically the best of any row - but it was seeded
  from `xgboost_default`'s own modelled map, not an independent
  observation, and bridges a real 6-day gap (Aug 1 to Aug 7) this time.
  Lower rmse here says more about `fine_t0`'s source model than about
  STARFM's own skill; `scripts/validate_starfm_real_reference.py` (see
  below) is the check that actually uses a real fine-resolution reference.

Both runs together say the same thing more clearly than either alone: this
benchmark is real and reproducible, but one AOI over days-to-weeks is not
enough data to declare a winner. Per-model hyperparameter tuning (beyond
the two named RF/XGBoost configurations here) and a genuinely large-sample
run (a full season, or multiple cities) remain open before ranking these
models with any confidence.

### Run 3: same 4-week window, after the GWR `_distinct_locations()` fix

Same `BENCHMARK_START`/`BENCHMARK_END`/AOI/`max_workers=1` as Run 2, rerun
after fixing `GWRDownscaler`'s k-NN search to dedupe training rows by
distinct `(x, y)` location instead of searching raw `(time, y, x)` rows
(see `CHANGELOG.md` - a training pixel observed at T valid times previously
appeared T times at the same location, so `max_local_samples` could exhaust
itself on time-duplicates of a handful of pixels instead of reaching truly
distinct neighbors):

| Model | RMSE (K) | Coverage |
|---|---|---|
| GWR (5x bandwidth) | 7.681 | **11.6%** |
| GWR (15x bandwidth) | 7.863 | **11.6%** |

(Every other model's numbers were unaffected by this fix, as expected -
`ols`/`tsharp_ndvi`/`random_forest_*`/`xgboost_*` all reproduced Run 2's
rmse to 3 decimal places.)

- **The fix is real but was not the dominant cause of low coverage.** rmse
  moved (7.591->7.681 and 7.605->7.863 for the two bandwidths) - proof the
  dedup fix does change which training points a local fit actually uses -
  but coverage stayed at exactly the same 11.6% (77/664) both before and
  after, and still identical between 5x and 15x bandwidth. A synthetic
  reproduction had confirmed the dedup fix alone restores 100% coverage
  (see `CHANGELOG.md`), so real coverage being capped at a bandwidth-
  independent 11.6% points at a second, separate bottleneck the synthetic
  test's fully-populated grid did not exercise.
- **Working hypothesis, not yet confirmed**: Run 2 already flagged the two
  candidates - `min_local_samples=10` still not being met, or
  `blocked_spatiotemporal_split`'s checkerboard geometry
  (`spatial_block_period=3`: training keeps cells where
  `(row+col) % 3 != 0`, validation holds out exactly the complementary
  `(row+col) % 3 == 0` cells) combined with real cloud/quality gaps in the
  Sentinel-3 LST cube. `fit_gwr_downscaler` only keeps training rows where
  *every* requested variable is finite at that `(time, y, x)`
  (`np.isfinite(table).all(axis=1)`), so real cloud cover can leave large
  contiguous regions of the training half with **zero** fully-finite
  observations across the whole training window - a gap no bandwidth
  increase closes, since increasing bandwidth only changes *how far* the
  k-NN search looks, not whether any qualifying point exists to find.
  This is consistent with bandwidth having zero effect at both 5x and 15x.
- **Not yet re-diagnosed with real numbers**: a follow-up diagnostic script
  (count distinct training locations surviving the finite-filter, and how
  many fall within each bandwidth for a sample of held-out cells) was
  started but had to be aborted mid-run - this machine was under real
  memory pressure from unrelated concurrent processes (not part of this
  benchmark) at the time, and swap was fully exhausted. Re-run
  `gwr_coverage_diag.py`-style instrumentation (or add it permanently
  behind a `--diagnose-gwr` flag in `benchmark_downscalers.py`) once the
  machine is free, before touching `min_local_samples` or the split
  geometry - per the standing rule here, report the dominant cause with
  real numbers before changing any parameter.

### STARFM against a real independent reference

`scripts/validate_starfm_real_reference.py` closes the gap the sanity
checks above leave open: it uses a real Landsat 8/9 scene as `fine_t0`
(a real observation, not another model's prediction) paired with two real
Sentinel-3 coarse dates only ~1 day apart (2026-08-15 -> 2026-08-16, from a
50-real-observation Sentinel-3 cube over a 45-day search window):

- **Coarse-scale conservation: rmse=2.15 K, mae=1.68 K** (still outside its
  own 1 K tolerance, but by far the best STARFM result seen so far -
  plausibly because both inputs this time are genuinely real: a real
  independent `fine_t0`, not a regression model's noisy output, and a
  1-day t0-t1 gap instead of the benchmark's 3-6 days, leaving far less
  time for unrelated real weather change STARFM cannot represent).
- **A real bug, not a real finding, in the first version of the
  full-resolution check**: it found a second Landsat scene within its
  4-day tolerance of t1 and reported rmse=34 K against it - a nonsense
  number. The two Landsat scenes involved
  (`LC09_L2SP_192023_20260815...` and `LC09_L2SP_192024_20260815...`)
  are 24 seconds apart with adjacent WRS-2 rows on the same path: the
  *same overpass* split into adjacent along-track tiles, covering
  different footprints - not a later, independent observation of this
  AOI at all. The script's "second scene near t1" search only excluded
  matching `product_id`, not matching acquisition *date*, so it happily
  accepted a same-day neighboring tile as if it were a real t1 reference.
  Fixed by requiring the candidate scene's date to differ from `fine_t0`'s
  own date, not just its product ID - a genuine full-resolution check
  needs a rerun to get a real number.

This is exactly the kind of bug real independent-reference validation is
supposed to surface, and could not have been caught by unit tests alone -
only a live search over real Landsat metadata revealed that "a second
scene near t1" and "a second scene at a genuinely different date" are not
the same filter.

## Implementation Roadmap

1. Complete IDW and support diagnostics, now used automatically by the
   workflow when a predictor grid exists.
2. **Done, with real caveats - see "Benchmark Results" above.** Benchmark
   OLS, TsHARP/DisTrad, Random Forest, XGBoost, GWR and a STARFM sanity
   check on one real blocked spatial-temporal holdout
   (`scripts/benchmark_downscalers.py`). Still open: hyperparameter tuning
   per model, a larger AOI/longer period for a statistically meaningful
   comparison, and a real independent-reference STARFM validation.
3. Independent reference ingestion: **Landsat 8/9 and ECOSTRESS done and
   fully verified against real data** (`sensors/landsat.py`,
   `sensors/ecostress.py` - `LandsatCatalog`/`read_landsat_lst`,
   `EcostressCatalog`/`read_ecostress_lst`), registered as
   `landsat_lst`/`ecostress_lst` so both coexist with Sentinel-3's `lst` in
   the same fused cube - see `validation.validate_independent_reference`.
   Landsat: search + authenticated read + physically plausible values
   confirmed against real Planetary Computer data. ECOSTRESS: search
   confirmed against real NASA CMR-STAC data, and a real authenticated read
   over Berlin (2026-09, `EARTHDATA_BEARER_TOKEN` set) returned 9-15 degC at
   a nighttime overpass - physically correct. That live run surfaced and
   fixed two real bugs neither could have been caught without real data:
   GDAL's `vsicurl` doesn't work cleanly against this host (curl
   auto-attaches a `~/.netrc` Earthdata entry mid-OAuth-redirect and loops
   forever; fixed by fetching via plain `requests` + a temp file instead),
   and the scale/offset fallback assumed the ECOSTRESS ATBD's *0.02
   digital-number formula, which turned out to apply to the older Swath
   product, not the real L2T tiles actually served here (identity
   scale/offset, already Kelvin) - see `sensors/ecostress.py`'s module
   docstring for both. **A geographically weighted baseline is now
   implemented**: `fit_gwr_downscaler`/`fit_coarse_consistent_gwr_downscaler`
   (see "Implemented Model APIs" below) - not yet benchmarked against the
   other baselines on real blocked holdouts (that benchmarking is roadmap
   item 2, still open for every model here, GWR included).
4. Add classical spatiotemporal fusion experiments with explicit optical
   versus thermal validation. **STARFM and ESTARFM done** (`fuse_starfm`,
   `fuse_estarfm` - see "Implemented Model APIs"), both unit-tested against
   synthetic cases (STARFM: a spatially localized change; ESTARFM: two
   surfaces with a genuinely different fine/coarse *slope*, the case
   STARFM's raw difference cannot represent - see their tests in
   `tests/test_local.py`). Real-data validation is partial:
   `scripts/validate_starfm_real_reference.py` uses a real Landsat scene as
   `fine_t0` (not another model's prediction) and checks coarse-scale
   conservation against real observed Sentinel-3 LST at t1. **Run**:
   rmse=2.15 K (still outside its own 1 K tolerance, but by far STARFM's
   best real result so far - see "STARFM against a real independent
   reference" above) against a real 1-day t0/t1 gap. The full-resolution
   half of that same run caught a real bug rather than producing a real
   number (a same-overpass adjacent Landsat tile was accepted as a "later"
   reference by mistake) - fixed, but not yet rerun for a genuine
   full-resolution number.
   ESTARFM itself has no real-data validation run yet (unlike STARFM) - a
   `scripts/validate_estarfm_real_reference.py` mirroring the STARFM one
   (bracketing t0/t2 Landsat scenes instead of a single fine_t0) is the
   planned next step. **Decision: FSDAF stays deliberately unimplemented
   until ESTARFM gets that real-data run** - adding a third
   spatiotemporal-fusion method before validating the second one already
   built is premature, not a scope gap. FSDAF and ESTARFM both matter most
   for exactly the case the single-pair STARFM form handles badly: large or
   heterogeneous change between t0 and t1, e.g. genuine land-cover change,
   not just a temperature swing - so FSDAF only earns its place once
   ESTARFM's real-data behavior on that same case is known.
5. Add kriging or land-use regression for OpenAQ only if station-level
   cross-validation demonstrates improvement over guarded IDW.
6. Evaluate a deep model only after the baselines and uncertainty protocol are
   stable.
