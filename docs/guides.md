# Guides

Worked, runnable notebooks showing how to use `sentinel_analysis` for a
specific task end to end - not API reference, not synthetic-only demos.
Each one narrates every step in markdown before the code that does it, so
it can be read top to bottom without a running kernel, or executed for
real against the sensors/providers it uses. An `_executed.ipynb` twin next
to each source notebook is the last real run's actual output, committed as
evidence rather than a claim.

## Sentinel-3 land surface temperature over Berlin

`notebooks/sentinel3_lst_demo.ipynb` - the CDSE OData catalogue, SAFE
reading, the documented L2 LST exception flags, Kelvin-to-Celsius, and a
two-week Berlin temperature series with both a time series and one spatial
snapshot.

```bash
uv run --extra notebook --extra optical jupyter notebook notebooks/sentinel3_lst_demo.ipynb
```

## Multisensor thermal downscaling over Berlin

`notebooks/berlin_multisensor_downscaling.ipynb` - the complete
discover/download/fuse workflow (Sentinel-3 LST, Sentinel-2 predictors,
optional Sentinel-5P and OpenAQ), then the coarse-consistent Random Forest
baseline and an interactive observed-vs-modelled time slider. See
[`downscaling.md`](downscaling.md) for what's validated on real data versus
synthetic-only before trusting any one model's numbers from this notebook.

```bash
uv run --extra notebook --extra ml --extra auxiliary --extra optical \
  jupyter notebook notebooks/berlin_multisensor_downscaling.ipynb
```

## City temperature animation

`notebooks/city_temperature_animation.ipynb` - turning a fused thermal
cube into an animated map over time for one city.

```bash
uv run --extra notebook --extra auxiliary --extra optical \
  jupyter notebook notebooks/city_temperature_animation.ipynb
```

## Methane point-source detection data: Sentinel-2 L1C + Carbon Mapper

`notebooks/methane_detection_carbon_mapper.ipynb` - assembles the *data*
side of Varon et al. 2024 (Nat. Commun. s41467-024-47754-y): given one real
airborne-confirmed methane leak from the
[Carbon Mapper catalog](https://doi.org/10.5281/zenodo.7072824), finds and
downloads a real reference/detection Sentinel-2 L1C pair around it
(`Sentinel2Catalog(product_type=SENTINEL2_L1C_PRODUCT_TYPE)`,
`select_temporal_pair`), reads both with `read_s2_l1c`, and plots the B12
reflectance change the paper's model would be shown.

Unlike the other guides, fetching and filtering the Carbon Mapper catalog
itself is written as plain code in this notebook, not a library class - it
is a static, one-off research dataset (one campaign, 2020-2021, never
updated), not a general-purpose data source like ERA5/CAMS/OpenAQ, so it
does not belong in `sentinel_analysis.providers`. This notebook is the
demonstration of how to combine the library's generic pieces
(`AssetCache`, `http_session`, the CDSE catalogue, `select_temporal_pair`)
with a one-off external dataset, not a reason to add one.

**Explicitly out of scope**, called out again at the end of the notebook:
the paper's synthetic training-data generator (a Gaussian plume dispersion
model plus Beer-Lambert absorption, injected into real background scenes)
and the deep-learning model itself. This guide gets you real validation
data in the paper's shape, not a trained detector.

```bash
uv run --extra notebook --extra optical jupyter notebook notebooks/methane_detection_carbon_mapper.ipynb
```
