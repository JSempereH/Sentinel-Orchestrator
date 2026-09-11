# Sentinel Analysis

Library for combining Sentinel-1/2/3/5P with ERA5, CAMS, and OpenAQ over an
area of interest. The workflow keeps thermal observations, predictors, and
stations separate.

## Three Commands

```bash
uv sync --extra dev --extra auxiliary --extra optical
uv run sentinel-analysis plan request.json
uv run --extra auxiliary --extra optical python scripts/smoke_e2e.py
```

## Scientific Rule

A Sentinel-5P or CAMS atmospheric column is not automatically a surface
concentration. Conversion to a surface estimate must be explicit and marked
as modeled.

## Project Status

- Canonical package: `sentinel_analysis`.
- Sentinel-3: `Sentinel3LST` for local processing and `Sentinel3LSTClient` for openEO.
- Sentinel-1: three `sentinel1_backend` options - `"snap"` (default, local
  SNAP GPT), `"hyp3_rtc"` (ASF HyP3 cloud RTC, no SNAP needed), `"s1ard"`
  (pyroSAR NRB, currently broken - see `roadmap.md`).
- Independent thermal references: Landsat 8/9 (`sensors/landsat.py`) and
  ECOSTRESS (`sensors/ecostress.py`), alongside Sentinel-3's own `lst`.
- Downscaling/fusion (`downscale.py`): OLS/TsHARP, Random Forest, XGBoost,
  GWR (geographically weighted regression), and STARFM/ESTARFM
  spatiotemporal fusion - see `downscaling.md` for what's validated on
  real data versus synthetic-only.
- Auxiliary sources: `providers/era5.py`, `providers/cams.py`, and `providers/openaq.py`.
- Cache: SHA-256 checksums and a JSON manifest for each download.
- `worker/`: an optional FastAPI service (+ bundled UI) that runs
  `AnalysisWorkflow.execute()` as a submit/poll/download job over HTTP -
  see `platform.md`.
