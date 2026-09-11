# Architecture

![Sentinel Analysis architecture](assets/architecture.svg)

The editable source is available as [`architecture.drawio`](assets/architecture.drawio).

```text
AnalysisRequest
      |
      v
WorkflowPlan -> discovery -> cache/download -> sensor readers
      |                                  |
      +------------ harmonize/fuse <-----+
                         |
                         v
             thermal_cube / predictor_cube / auxiliary
```

## Names

`sentinel_analysis` is the current and only package name. The old names
`sentinel3_lst` and `urban_heat` have no active source modules; they only
remain in historical artifacts or ignored generated files.

`Sentinel3LST` is not an old project name. It is the local facade for
Sentinel-3 LST products. `Sentinel3LSTClient` builds openEO graphs and remains
separate because it is a different remote backend.

## Data Separation

- `thermal_cube`: Sentinel-3 thermal analysis grid.
- `predictor_cube`: Sentinel-1/2/5P and gridded auxiliary data.
- `auxiliary['openaq']`: stations, never an invented raster surface.

Gridded auxiliary variables are renamed to
`auxiliary_<provider>_<variable>` when merged to avoid collisions with
satellite variables.
