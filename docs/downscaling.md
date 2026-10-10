# Downscaling

Sentinel-3 measures land surface temperature every day, but in 1 km cells: a
whole neighbourhood in one number. citycube sharpens it to 100 m using what
Sentinel-2 sees at that resolution (vegetation, built-up surfaces, water) and
the terrain.

![A Sentinel-3 pass at 1 km and downscaled to 100 m, with the NDVI used and the correction applied](assets/figures/04_downscaled_scene.png)

## How it works

For each Sentinel-3 pass, on its own:

1. The Sentinel-2 indices (NDVI, NDBI, EVI, NDMI) and the terrain (elevation,
   slope, solar illumination at that hour) are averaged to the 1 km grid.
2. A model learns how temperature relates to them from the clear 1 km cells
   of that pass.
3. The model predicts temperature from the same indices at 100 m.
4. A smooth correction makes the 100 m map average back to what Sentinel-3
   observed, so the sharpened map never contradicts the measurement.

Cells that were cloudy at 1 km stay empty at 100 m. A pass with no
Sentinel-2 image within the temporal tolerance, or too few clear cells, is
skipped and listed in the result.

## Using it

```python
request = dataclasses.replace(
    request,
    sensors=("sentinel3", "sentinel2"),
    thermal_overpass="day",
    downscale=cc.DownscaleSpec(),           # model="local_trees" by default
)
result = cc.AnalysisWorkflow(request).execute("output/city")
result.downscaled["lst_downscaled"]          # 100 m maps, one per pass
```

Options of `DownscaleSpec`:

| Option | Default | Meaning |
|---|---|---|
| `model` | `"local_trees"` | also `"linear"`, `"random_forest"`, `"xgboost"` |
| `correction` | `"smooth"` | `"block"` matches each 1 km cell exactly, with visible steps |
| `mask_unobserved` | `True` | leave cloudy cells empty |
| `model_options` | `{}` | e.g. `{"window": 10}` for `local_trees`, in 1 km cells |

`local_trees` combines one model for the whole city with local models in
moving 10 km windows, because the relation between vegetation and
temperature is not the same in the centre as on the outskirts. It follows
the Data Mining Sharpener (Gao et al. 2012) used by ESA's Sen-ET.

## How good it is

Two checks on Berlin, August 2026, 15 passes.

**Hidden cells.** Part of the city (5 km blocks) is hidden from training and
the model's error is measured there, at 1 km:

| Model | RMSE (K) |
|---|---|
| `local_trees` | 2.02 |
| linear | 2.19 |
| Random Forest | 2.26 |
| XGBoost | 2.26 |
| TsHARP (NDVI only) | 2.46 |
| no skill (each pass's mean) | 2.86 |

![Blocked-holdout error of every model](assets/figures/04_model_comparison.png)

**Landsat.** Landsat measures temperature at 100 m from its own instrument,
and is never used for training. On the two mornings with both satellites:

| Landsat pass | Downscaled RMSE / correlation | 1 km repeated at 100 m |
|---|---|---|
| A | 3.00 K / 0.46 | 2.97 K / 0.45 |
| B | 2.07 K / 0.66 | 2.28 K / 0.51 |

On B the 100 m map is clearly closer to Landsat than the 1 km observation;
on A neither is, because the two sensors disagree in level that day by more
than the detail adds. Two passes over one city are a small sample.

![Landsat against the downscaled map, same morning](assets/figures/04_landsat_validation.png)

## Other tools

- **STARFM and ESTARFM** fill the days between Landsat passes by adding the
  change Sentinel-3 saw to the last Landsat map (`cc.fuse_starfm`,
  `cc.fuse_estarfm`). Against a real Landsat scene a month later STARFM's
  error was 2.7 K.
- **Prediction intervals** (`cc.fit_conformal_downscaler`) work on synthetic
  data but are not yet calibrated on real data: 90 % intervals covered
  between 42 % and 93 % of real cells.

![STARFM predicting a Landsat map a month later, next to the real one](assets/figures/04_starfm.png)

Notebook [04](guides.md) runs all of this on real data. The literature
behind these choices and every benchmark run are in
[`dev/downscaling-research.md`](https://github.com/JSempereH/citycube/blob/main/dev/downscaling-research.md).
