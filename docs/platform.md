# Platform: sentinel_analysis and sentinel-worker

This repo hosts two things:

- `src/sentinel_analysis/` - the library (docs, tests, CLI).
- `worker/` - a thin FastAPI wrapper around it (`sentinel-worker`), so a
  multisensor `AnalysisRequest` can be submitted/polled/downloaded over HTTP
  from a machine with the heavy geospatial extras installed. It ships its
  own UI (`worker/frontend/`, served by the worker itself at `/`) and
  remains fully drivable via its bare HTTP API too, for scripts or
  automation.

Both are self-contained: nothing here depends on any other tool, and
nothing here should grow a dependency on one - if a future integration ever
seems to need that, model it as this worker's plain HTTP API being called
from wherever, not as shared data models or in-process coupling.

## Running the worker

```bash
make install-worker   # creates worker/.venv, installs deps + sentinel_analysis extras
make worker           # :8100
```

See `worker/README.md` for the full API and its `.env` setup
(`WORKER_API_TOKEN` + CDSE/CDS/CAMS/OpenAQ credentials).

## Worker UI

`worker/frontend/` (React 18 + Vite + TS) is a small SPA for submitting
runs, watching them, and downloading results, so `curl`/scripts are no
longer the only way to drive the worker:

- **New run**: draw an AOI (its map component was originally built for a
  different tool and adapted here), pick sensors (including
  Landsat/ECOSTRESS) and date range, submit.
- **Jobs**: history table, polling automatically (every 5s) only while a job
  is `PENDING`/`RUNNING`; expand a row for its full request and a result
  download.
- **Settings**: worker URL override (only needed if this build is hosted
  separately from the worker) + bearer token, saved to `localStorage`; a
  "Test connection" button surfaces `/usage`. Provider credentials
  (CDSE/CDS/OpenAQ) are never entered here - they live in the worker's own
  `.env`.

One deployable service, no CORS anywhere: the built SPA is mounted at `/`
via Starlette's `StaticFiles(html=True)`, registered *after* every API
route, so `/health`, `/jobs`, `/usage` keep matching first and only
unmatched paths fall through to `index.html`. In dev, Vite's `server.proxy`
forwards those same paths to the worker so the browser still only talks to
one origin. Run it with:

```bash
cd worker/frontend && npm install && npm run dev   # dev server, proxied to :8100
# or, for the production build the worker itself serves:
npm run build                                       # writes worker/frontend/dist/
```

`docker compose build worker` bakes the frontend build into the image via a
`node:20-slim` build stage in `worker/Dockerfile` - the final image needs no
Node.js at all.

## Production hardening (single-user / LAN deployment)

This is self-hosted, single-trusted-user tooling, so "production" here means
*durable and observable*, not multi-tenant:

- **CI**: `.github/workflows/ci.yml` lints and tests `sentinel_analysis` and
  `worker` on every push/PR.
- **Lint**: `ruff` (`select = ["E", "F", "B"]`) in `pyproject.toml`, `eslint`
  for the frontend. All clean as of this change.
- **Containers**: `worker/Dockerfile` (multi-stage: a `node:20-slim` stage
  builds the frontend, the final Python stage copies in the static `dist/` -
  no Node.js in the shipped image), run via `docker-compose.yml`
  (`make docker-up`). The image excludes `s1ard` (confirmed broken against
  a real scene, see `docs/roadmap.md`) and SNAP itself (a large Java
  desktop app this Dockerfile does not bundle), so `sentinel1_backend`
  defaults to `"snap"` but that backend is only usable from a separate
  image with SNAP's `gpt` on `PATH`; the lightweight `hyp3` extra *is*
  installed, so `sentinel1_backend="hyp3_rtc"` works out of the box (needs
  Earthdata credentials, see `docs/setup.md`). `cloud` (zarr/dask) *is*
  installed too - `execute_and_persist()` always calls `write_zarr()`, so
  without it every job fails at the last step regardless of sensor. This
  was caught by a real end-to-end container run, not just inferred from
  the code - see `docs/roadmap.md`.
- **Worker job durability**: job status is cached in memory and mirrored to
  `<WORKER_OUTPUT_DIR>/jobs.db` (SQLite); a restart marks any job still
  `PENDING`/`RUNNING` as `FAILED` with an explanatory message instead of
  losing it.
- **Worker `/usage`**: `sentinel_analysis`'s own providers (CDSE, CDS/ADS,
  OpenAQ) have no numeric credit balance the way a paid processing service
  might; `GET /usage` reports what each actually exposes (OpenAQ's real
  rate-limit headers; credential presence only for CDSE/CDS-ADS).
- **Git hygiene**: `worker/runs/` is in this repo's `.gitignore` - it's the
  worker's job-output directory (regenerable by re-running a job), not
  source.
