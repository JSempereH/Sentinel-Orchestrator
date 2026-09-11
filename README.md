![Sentinel Orchestrator](docs/assets/banner.svg)

[![CI](https://github.com/JSempereH/Sentinel-Orchestrator/actions/workflows/ci.yml/badge.svg)](https://github.com/JSempereH/Sentinel-Orchestrator/actions/workflows/ci.yml)
[![License: EUPL-1.2](https://img.shields.io/badge/license-EUPL--1.2-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

Python library for analyzing an area of interest with Sentinel-1, Sentinel-2,
Sentinel-3 LST, Sentinel-5P, Landsat 8/9, ECOSTRESS, and auxiliary
meteorological and air-quality sources - discovery, download, and fusion into
one analysis-ready cube, plus a coarse-to-fine thermal downscaling toolkit
(OLS/TsHARP, Random Forest, XGBoost, GWR, STARFM/ESTARFM) and an optional
`worker/` HTTP service to run it all as a job.

Known gaps and what's validated on real data versus synthetic-only are
tracked honestly in [`docs/roadmap.md`](docs/roadmap.md) - worth reading
before relying on any one model's numbers.

---

## Quick Start

```bash
cp .env.example .env
uv sync --extra dev --extra auxiliary --extra optical
uv run pytest -q
```

For local documentation:

```bash
uv sync --extra docs
uv run mkdocs serve
```

Credentials (CDSE, ERA5, CAMS, OpenAQ, Earthdata) are covered in
[`docs/setup.md`](docs/setup.md).

---

## E2E Smoke Test

With credentials configured and the CDS/ADS licenses accepted:

```bash
uv run --extra auxiliary --extra optical python scripts/smoke_e2e.py
```

The test downloads one Sentinel-3 scene, one ERA5 variable, one CAMS variable,
and a small set of OpenAQ stations for Berlin on one day. Results are written
to `output/e2e-smoke/`, which is ignored by Git.

You can also inspect a request without downloading data:

```bash
uv run sentinel-analysis plan request.json
uv run sentinel-analysis discover request.json
uv run sentinel-analysis auxiliary request.json output/auxiliary
```

---

## Architecture

```text
src/sentinel_analysis/
  sensors/       Sentinel-1/2/3/5P, Landsat 8/9, and ECOSTRESS adapters
  providers/     ERA5, CAMS, and OpenAQ
  workflow/      requests, planning, downloads, fusion, and results
  downscale.py   OLS/TsHARP, Random Forest, XGBoost, GWR, STARFM/ESTARFM
  cube.py        grids and the spatial contract
  metadata.py    units, provenance, and scientific contracts
```

Sentinel-1 has three interchangeable `sentinel1_backend` options -
`"snap"` (default, local SNAP GPT), `"hyp3_rtc"` (ASF HyP3 cloud
processing, no SNAP needed), and `"s1ard"` (pyroSAR NRB, currently broken
- see `docs/roadmap.md`). Landsat/ECOSTRESS are independent thermal
references used as validation/predictors alongside Sentinel-3's own `lst`.
`downscale.py`'s models turn a coarse thermal field into a fine-resolution
one using Sentinel-2 predictors - see
[`docs/downscaling.md`](docs/downscaling.md) for what's validated on real
data versus synthetic-only.

`sentinel_analysis` is the only current package name. `Sentinel3LST` is the
local Sentinel-3 facade, while `Sentinel3LSTClient` is the remote openEO
client. `sentinel3_lst` and `urban_heat` were historical prototype names and
are not active source packages.

### sentinel-worker

`worker/` wraps `AnalysisWorkflow.execute()` behind a small FastAPI service
(`POST /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/result`) so a multisensor run
can be driven over HTTP from a separate machine with the heavy geospatial
extras installed. It ships its own UI (`worker/frontend/`, served by the
worker itself) and remains fully drivable via its bare HTTP API too - see
[`worker/README.md`](worker/README.md) and [`docs/platform.md`](docs/platform.md).

---

## Validation

```bash
uv run --extra dev --extra auxiliary --extra optical pytest -q
uv build
```

Real-product tests are optional and never download files by themselves. See
the complete guide in [`docs/`](docs/index.md).

The scientific downscaling and multisensor-fusion roadmap, including guarded
OpenAQ station interpolation, is documented in
[`docs/downscaling.md`](docs/downscaling.md).

---

## License

Licensed under the [European Union Public Licence v. 1.2](LICENSE) (EUPL-1.2).
