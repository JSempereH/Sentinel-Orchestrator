# Land surface temperature downscaling: literature review, October 2026

Why this review: to decide whether citycube's downscaling (per-scene
regression sharpening of Sentinel-3 with Sentinel-2, `local_trees` by
default) is still the right design, whether Landsat should be sharpened
too, and which of ATPRK or FSDAF to add. Figures quoted from papers come
from their abstracts or from studies citing them; each is linked so it can
be checked.

## 1. Families of methods

### Regression sharpening

A model relates coarse temperature to fine predictors aggregated to the
coarse grid, then predicts at fine resolution, assuming the relation holds
across scales (scale invariance).

- **DisTrad and TsHARP**: one predictor (NDVI or vegetation cover). Simple
  and interpretable, and still the usual baseline
  ([Agam et al. 2007](https://doi.org/10.1016/j.rse.2006.10.006)). Weak in
  built-up areas and over water, the parts of a city that matter most
  ([Xu et al. 2020](https://doi.org/10.3390/rs12071082)).
- **Data Mining Sharpener (DMS)**: regression trees with linear leaves, a
  global model plus local moving-window models, blended by residual (Gao
  et al. 2012). It is what ESA's Sen-ET uses to sharpen Sentinel-3 with
  Sentinel-2, with relative errors below 20 % in the downstream
  evapotranspiration ([Guzinski & Nieto 2019](https://doi.org/10.1016/j.rse.2018.11.019)).
  citycube's `local_trees` follows it.
- **Random Forest, XGBoost and other machine learning**: generally better
  than TsHARP in heterogeneous areas; a Landsat-7 study found up to 19 %
  lower error than TsHARP in a same-sensor test with RMSE of 1.02 to 1.43 K
  ([Hutengs & Vohland 2016](https://www.sciencedirect.com/science/article/abs/pii/S0034425716300992)).
  Random Forest and XGBoost perform about equally in a MODIS-to-30 m
  comparison ([Remote Sensing 17:2350, 2025](https://doi.org/10.3390/rs17142350)).

**The main warning of recent work is scale invariance.** A 2026 study over
four Danish urban areas found that Gradient Boosting fitted best at the
coarse scale but was, with Random Forest, among the worst at the fine
scale: trees cannot extrapolate beyond the training range, and a model
that wins where it is trained need not win where it predicts. Linear
regression with residual correction, fitted per scene, was the most
reliable at the fine scale and is what the authors recommend for
operations ([Pereira et al. 2026](https://doi.org/10.3390/rs18132263)).
A comparison over plateau and mountain terrain also found Random Forest
and XGBoost not uniformly better than TsHARP and DisTrad
([Frontiers in Earth Science, 2024](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2024.1488711/full)).

### Residual correction and area-to-point regression kriging (ATPRK)

Every regression leaves a coarse residual (observation minus aggregated
prediction). How it is put back on the fine grid matters:

- added uniformly per coarse cell: exact, but leaves coarse-cell steps;
- interpolated (bilinear or cubic, as in pyDMS): smooth, not exact;
- **area-to-point kriging (ATPK)**: kriging that accounts for the size of
  the coarse cells and the spatial structure of the residuals, fitted by
  deconvolving their variogram. Regression plus ATPK is ATPRK
  ([Wang et al. 2015, Remote Sensing of Environment 166:191-204](https://www.researchgate.net/publication/305541292_Downscaling_MODIS_images_with_area-to-point_regression_kriging);
  [Wang, Shi & Atkinson 2016, ISPRS J. 114:151-165](https://www.sciencedirect.com/science/article/abs/pii/S0924271616000496)).
  It is coherent: the fine prediction averaged over a coarse cell returns
  the observation.

Urban evidence: Random Forest plus ATPK (RFATPK) sharpened 90 m ASTER LST
over two parts of Guangzhou to the lowest RMSE of the methods compared,
1.01 K and 0.68 K against 0.81 K for TsHARP in the second area
([Xu et al. 2020](https://doi.org/10.3390/rs12071082)). Variants make the
regression local (geographically weighted ATPRK, [Jin et al. 2018](https://www.researchgate.net/publication/324421313_Geographically_Weighted_Area-to-Point_Regression_Kriging_for_Spatial_Downscaling_in_Remote_Sensing))
or replace it by neural networks ([Remote Sensing 16:2542, 2024](https://doi.org/10.3390/rs16142542)).

### Sharpening Landsat with Sentinel-2

Landsat 8/9 measure thermal radiance at about 100 m; the Collection 2
product is resampled to 30 m by cubic convolution and carries no 30 m
detail. Sharpening it with Sentinel-2 indices is established practice:

- Landsat 8 to 10 m with NDVI, NDBI and NDWI in Google Earth Engine,
  R² about 0.64 ([Onačillová et al. 2022](https://doi.org/10.3390/rs14164076),
  code: [palubad/LST-downscaling-to-10m-GEE](https://github.com/palubad/LST-downscaling-to-10m-GEE));
- a per-land-cover nonlinear sharpener for Landsat 8/9 with Sentinel-2
  ([HUTS-LC, 2025](https://www.tandfonline.com/doi/full/10.1080/24694452.2025.2574331));
- Warsaw, 2026: computing at 20 m and only then resampling to 10 m reduced
  artifacts at land-cover boundaries
  ([Advances in Space Research](https://www.sciencedirect.com/science/article/pii/S027311772600949X));
- the same DMS with harmonised Landsat-Sentinel reflectance sharpens
  ECOSTRESS and VIIRS to 30 m
  ([Remote Sensing of Environment, 2020](https://www.sciencedirect.com/science/article/pii/S0034425720304259)).

A recurring caution: sharpening 30 m to 10 m only redistributes a signal
that was already smoothed at 100 m; the real gain is from ~100 m.

### Spatiotemporal fusion (STARFM, ESTARFM, FSDAF)

These fill the days between fine acquisitions by adding the change seen at
coarse resolution to the last fine image. They solve a different problem
from sharpening one scene.

- **FSDAF** combines spectral unmixing with thin plate spline
  interpolation, needs few inputs, and handles land-cover change
  ([Zhu et al. 2016, RSE 172:165-177](https://data.fs.usda.gov/research/pubs/iitf/ja_iitf_2016_Zhu_001.pdf)).
  It tends to blur ([FSDAF 2.0, Guo et al. 2020](https://ira.lib.polyu.edu.hk/handle/10397/90623)).
  For LST it is the most used fusion method, with errors against Landsat
  of roughly 0.9 to 1.7 °C in one Landsat-MODIS study
  ([ResearchGate](https://www.researchgate.net/publication/344379138_Spatio-temporal_fusion_of_Landsat_and_MODIS_land_surface_temperature_data_using_FSDAF_algorithm)).
- STARFM and ESTARFM were designed for reflectance and may not suit LST's
  strong sub-daily dynamics; an unbiased ESTARFM corrects local bias with
  the coarse LST and was checked against in-situ and ECOSTRESS LST across
  Australia ([RSE 2023](https://www.sciencedirect.com/science/article/pii/S0034425723003358)).

### Deep learning

- **GrokLST / MoCoLSK** (2024): a 30 m SDGSAT-1 dataset and a toolkit with
  40+ models; in that benchmark every deep model beat Random Forest,
  XGBoost, LightGBM and CatBoost ([arXiv 2409.19835](https://arxiv.org/abs/2409.19835),
  [code](https://github.com/GrokCV/GrokLST)).
- Attention-based super-resolution to 100 m: RMSE 4.39 K against
  ECOSTRESS on held-out station patches
  ([Science of Remote Sensing, 2025](https://www.sciencedirect.com/science/article/pii/S2666017225001415)).
- Daily 10 m LST by weakly supervised generative fusion
  ([WGAST, arXiv 2508.06485](https://arxiv.org/pdf/2508.06485)); 30 m
  Landsat by progressive self-training ([arXiv 2603.29478](https://arxiv.org/pdf/2603.29478)).

The strong results come from single-sensor benchmarks where the target
was degraded from the same sensor, or from random pixel splits; deep
models need large training sets and their transfer to unseen cities is
rarely tested. A random-forest study chose RF over deep learning
explicitly for that reason ([PeerJ CS, 2025](https://peerj.com/articles/cs-3246/)).

## 2. Validation

- Degrading a fine image and sharpening it back (Wald's protocol) is the
  usual test when no fine truth exists; it is optimistic, since input and
  reference come from the same sensor ([Hutengs & Vohland 2016](https://www.sciencedirect.com/science/article/abs/pii/S0034425716300992)).
- Random splits inflate scores because of spatial autocorrelation; blocked
  or independent-sensor tests are needed.
- Coarse-scale fit says little about fine-scale accuracy
  ([Pereira et al. 2026](https://doi.org/10.3390/rs18132263)): report both.
- Bias and spatial-pattern metrics can disagree and should be reported
  separately ([Remote Sensing 8:975, 2016](https://doi.org/10.3390/rs8120975)).

citycube already validates this way: a blocked spatial holdout, an
independent sensor (Landsat against downscaled Sentinel-3), RMSE,
correlation and pattern error separately.

## 3. Decisions for citycube

| Question | Decision | Why |
|---|---|---|
| Keep per-scene regression sharpening? | Yes | Operational standard (DMS, Sen-ET); recent work recommends per-scene fits with residual correction over multi-date models. |
| ATPRK or FSDAF? | **ATPRK now** (`correction="atpk"`) | It improves the step citycube already has (the residual), is coherent by construction, has urban LST evidence, and applies to every model and sensor. FSDAF fills days between Landsat passes, a separate problem STARFM/ESTARFM already cover, with known blurring and doubts for LST. |
| Sharpen Landsat? | **Yes** (`sharpen_landsat`) | Established practice, same machinery; the gain is from the native ~100 m. |
| Deep learning? | Not yet | Needs a multi-city training set with an independent target and a cross-city test; current evidence is mostly single-sensor or random-split. |
| Trees or linear? | Measure per case | Trees can break scale invariance (Pereira et al. 2026); citycube's comparison below includes linear models with each correction. |

## 4. citycube's own results

All on Berlin, 2026, with the scripts in `scripts/`.

**ATPK as the residual correction of Sentinel-3 downscaling** (1 km to
100 m, 17 passes kept by the cloud probe, `validate_downscaling_landsat.py`):

| Model + correction | Blocked holdout RMSE | Landsat pass B RMSE / r |
|---|---|---|
| `local_trees` + atpk | **2.13 K** | 2.06 K / 0.67 |
| `local_trees` + smooth | 2.15 K | 2.07 K / 0.66 |
| `local_trees` + block | 2.21 K | 2.06 K / 0.67 |
| linear + atpk | 2.28 K | 2.13 K / 0.67 |
| linear + smooth | 2.33 K | 2.14 K / 0.67 |
| Random Forest + smooth | 2.46 K | 2.54 K / 0.58 |

The first version used ordinary kriging, which carried the local mean
residual into unobserved blocks and scored worse than `smooth` on the
holdout (2.29 K). Simple kriging with a zero mean, the right assumption for
regression residuals, fixed it. Against Landsat the three corrections are
within 0.01 K: at 1 km to 100 m the model matters far more than how the
residual is spread. In notebook 04 (15 passes, no cloud probe) switching
from `smooth` to `atpk` changed each model's holdout RMSE by at most
0.03 K, in either direction (`local_trees` 2.02 to 2.05 K, linear 2.19 to
2.16 K, Random Forest and XGBoost 2.26 to 2.23 K). ATPK became the default
for exact conservation without 1 km steps; its accuracy gain for
Sentinel-3 is within noise.

**Sharpening Landsat with Sentinel-2** (Wald's protocol, two passes with a
Sentinel-2 image within 5 days, `validate_landsat_sharpening.py`):

| Degraded and sharpened | linear + atpk | `local_trees` + atpk | Random Forest + atpk | Bilinear | Repeated |
|---|---|---|---|---|---|
| 810 to 270 m | **1.03 K** | 1.06 K | 1.18 K | 1.29 K | 1.32 K |
| 540 to 180 m | **1.00 K** | 1.02 K | 1.22 K | 1.15 K | 1.21 K |
| 270 to 90 m | 0.99 K | 1.00 K | 1.27 K | **0.84 K** | 0.93 K |

ATPK was the best correction for every model at every scale. Where the
reference is resolved (180 and 270 m) sharpening beats interpolation by
13 to 20 %; at 90 m the reference is blurred by the sensor's own ~100 m
response and resampling, which favours smooth estimates, so that scale
cannot judge sharpening. Linear regression did best, consistent with the
scale-invariance findings of Pereira et al. (2026), and is the default of
`sharpen_landsat`.

Limits of this evidence: one city, one summer, two Landsat passes for the
Landsat test and two for the Sentinel-3 test. Each conclusion above held
on every pass, but more cities and seasons are needed before quoting the
gains as general.

## 5. Not covered

Directional (angular) effects on sharpened LST, emissivity, and sub-daily
dynamics between Sentinel-3 and Landsat overpasses; the new Copernicus and
NASA thermal missions (LSTM, TRISHNA, SBG) that will change the inputs.
