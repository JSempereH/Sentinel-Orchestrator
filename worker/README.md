# citycube-worker

A thin FastAPI wrapper around this repo's `citycube` library,
deployable to any machine with the geospatial dependencies installed (a
beefier PC, a cloud VM, reachable over SSH tunnel/VPN/LAN). It turns a
multisensor `AnalysisRequest` into a job: `POST /jobs` to submit, `GET
/jobs/{id}` to poll status, `GET /jobs/{id}/result` to download once
`SUCCEEDED`.

This worker ships its own UI (`frontend/`, see below) and remains fully
drivable via its bare HTTP API too, for scripts or automation.

## Setup

### `s1ard`/pyroSAR is currently broken - use `sentinel1_backend="snap"` (the default)

Confirmed against a real end-to-end run, not inferred: `s1ard` 2.13.1's own
scene-registration step (`pyroSAR.Archive.insert()`) calls
`spatialist.vector.Vector.reproject()`, which passes an `ogr.DataSource` to
`gdal.VectorTranslate()` - two SWIG-wrapped Python types that represent the
same underlying GDAL object but are not interchangeable (a known GDAL
Python-binding gotcha; see the `gdal-dev` mailing list thread on
`wrapper_GDALVectorTranslateDestName`). It fails with `TypeError: in method
'wrapper_GDALVectorTranslateDestName', argument 2 of type
'GDALDatasetShadow *'` before SNAP ever runs. This is a bug in `spatialist`,
not in this worker or `citycube` - `process_s1_ard`/
`sentinel1_backend="s1ard"` now raises a `UserWarning` pointing here.

`sentinel1_backend="snap"` (`process_s1_grd`) is the default and does not
go through `pyroSAR`/`spatialist` at all - it builds its own SNAP GPT XML
graph and calls `gpt` directly. Use that until upstream fixes this. The
prerequisites below (`libpq-dev`, `gdal-bin`/`libgdal-dev`) are still needed
if you install the `s1ard` extra anyway (e.g. to keep the option ready for
when it's fixed), but are **not** required for the default SNAP path -
that one only needs a SNAP installation with `gpt` on `PATH`.

A third option, `sentinel1_backend="hyp3_rtc"`, needs neither SNAP nor
`s1ard`/pyroSAR - it submits the granule name to ASF HyP3 and processes
entirely in the cloud (install the `hyp3` extra; set
`EARTHDATA_BEARER_TOKEN` or `EARTHDATA_USERNAME`/`EARTHDATA_PASSWORD`, see
`.env.example`). It never downloads the raw GRD from CDSE at all, so it's
the lightest option dependency-wise, at the cost of ASF HyP3 processing
credits and queue time instead of local compute.

### System prerequisites: the `s1ard` extra needs native libraries

`s1ard` (Sentinel-1 ARD processing, via `pyrosar`) has two transitive
dependencies that only ship as source distributions, so `uv`/`pip` must
compile them locally against system libraries:

**PostgreSQL client headers**, for `psycopg2` - `pyrosar` hard-depends on it
(built from source, not `psycopg2-binary`) for an optional scene-catalog
feature this worker never touches:

```bash
sudo apt-get install libpq-dev       # Debian/Ubuntu
sudo dnf install libpq-devel         # Fedora/RHEL
sudo pacman -S postgresql-libs       # Arch
```

Without this, install fails at `Failed to build psycopg2==...` with
`pg_config executable not found`.

**GDAL native library + headers**, for the `gdal` Python package - unlike
`rasterio` (which ships prebuilt wheels bundling GDAL), the plain `gdal`
package has no wheels and always compiles against your system's GDAL:

```bash
sudo apt-get install gdal-bin libgdal-dev g++   # Debian/Ubuntu
```

Without this, install fails at `Failed to build gdal==...` with `No such
file or directory: 'gdal-config'`. The resolved `gdal` Python package
version must also match your system's (`gdal-config --version`) - if your
distro ships an older GDAL than what gets resolved, the build fails again
with a version-mismatch error instead; add the `ubuntugis-unstable` PPA
(Debian/Ubuntu) for a newer GDAL if that happens.

Even once it builds, the `gdal` wheel built this way is silently missing
`osgeo._gdal_array` (pyroSAR/s1ard need it) unless numpy is visible *at
build time* - `uv pip install`'s default isolated build environment doesn't
see this venv's already-installed numpy. `make install-worker` (below)
rebuilds it with `--no-build-isolation` after installing `setuptools`/
`wheel` into the venv to fix this automatically; if you install manually,
do the same and check with
`python -c "from osgeo import gdal_array"`.

```bash
make install-worker   # from the repo root: creates worker/.venv, installs deps
                       # + citycube[cdse,auxiliary,optical,sar,s1ard,cloud],
                       # copies .env.example to .env
```

Or manually:

```bash
cd worker
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt -r requirements-dev.txt
uv pip install --python .venv -e "..[cdse,auxiliary,optical,sar,s1ard,cloud]"

cp .env.example .env   # fill in WORKER_API_TOKEN + CDSE/CDS/CAMS/OpenAQ credentials
```

`cloud` (zarr/dask) is not optional despite the name: `execute_and_persist()`
always writes the result cube(s) via `write_zarr()`, so a run without it
reaches `SUCCEEDED`-in-spirit and then fails at the last step.

### Frontend

```bash
cd worker/frontend
npm install
npm run build   # writes dist/, which the worker serves at / automatically
# or: npm run dev, for a hot-reloading dev server proxied to :8100
```

## Running

```bash
cd worker && .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8100 --timeout-keep-alive 30
# or from the repo root: make worker
```

Open `http://localhost:8100/` for the UI. `/health`, `/jobs`, `/usage` keep
resolving as the API underneath it - the frontend is a fallback for
everything else, not a replacement for the API routes.

Every endpoint except `/health` requires `Authorization: Bearer <WORKER_API_TOKEN>`.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check, no auth |
| POST | `/jobs?name=` | Submit an `AnalysisRequest.to_dict()`-shaped body (or the simpler `{aoi, start, end, sensors, ...}` shape - see `app/runner.py:build_request`), returns `{job_id, status}` |
| GET | `/jobs` | List every known job, most recent first: `[{job_id, name, status, progress, error_message, created_at}]` |
| GET | `/jobs/{id}` | As above, plus `request_params` (exactly what was submitted) |
| GET | `/jobs/{id}/result` | Zip of the job's output cubes (409 unless `SUCCEEDED`) |
| GET | `/usage` | Best-effort quota snapshot for OpenAQ (rate-limit headers), CDSE and CDS/ADS (credential presence only - neither exposes a numeric balance) |
| GET | `/*` | Falls through to the built frontend's `index.html` if `frontend/dist/` exists, otherwise 404 |

## Testing

```bash
cd worker && .venv/bin/pytest
# or from the repo root: make test-worker
```

Tests mock `app.jobs.execute_and_persist`, so they do not need
`citycube`'s heavy extras or real credentials installed.

## Notes

- Each job runs in its own child process (so it can be cancelled with
  `POST /jobs/{id}/cancel`, is stopped after `JOB_TIMEOUT_HOURS`, and an
  out-of-memory kill fails only that job) and logs to
  `<WORKER_OUTPUT_DIR>/<job id>/job.log` (`GET /jobs/{id}/log`). Status is
  mirrored to `<WORKER_OUTPUT_DIR>/jobs.db` (SQLite) so job *history*
  survives a restart; a run cannot resume mid-way, so any job still
  `PENDING`/`RUNNING` at startup is marked `FAILED` with an explanatory
  `error_message`.
- Finished jobs are deleted with their files after `JOB_RETENTION_DAYS`
  (or at once with `DELETE /jobs/{id}`); a successful job keeps only
  `result/` and `job.log`. `GET /health/ready` checks disk space and every
  provider credential. Full operating guide: `docs/platform.md`,
  "Production operation".
- A bearer token is sufficient for a single trusted worker reached over a
  LAN, VPN, or SSH tunnel. If this worker is ever reachable over the open
  internet, put TLS in front of it (reverse proxy) first.
- Oversized requests are rejected at submission with HTTP 422, before the
  job is queued: `MAX_AOI_KM2` (default 10000), `MAX_PRODUCTS_PER_SENSOR`
  (100) and `MAX_ESTIMATED_GB` (8, the estimated in-memory size of the
  cubes; peak RAM is up to about twice that). Set any of them to 0 to
  disable it. Products that fail to download or read are skipped and listed
  under `failed_products` in the result's `provenance.json`.
