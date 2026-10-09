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

## Production operation (single-user / LAN deployment)

This is self-hosted, single-trusted-user tooling, so "production" here means
*reliable, recoverable and observable*, not multi-tenant: one shared bearer
token, no TLS of its own, SQLite job store. Keep it on a LAN or VPN, or put
a TLS reverse proxy in front before exposing it anywhere else.

### Deploying

```bash
cp worker/.env.example worker/.env   # WORKER_API_TOKEN + provider credentials
make docker-build                    # records the current git commit in the image
docker compose up -d
curl -s localhost:8100/health        # liveness: version and commit
curl -s -H "Authorization: Bearer $TOKEN" localhost:8100/health/ready
```

The container runs as an unprivileged user (uid 1000), restarts unless
stopped, reaps the job processes it spawns (`init: true`), and is capped at
`WORKER_MEM_LIMIT` (default `12g`) so a runaway job is killed inside the
container instead of freezing the host. A named volume created by an older,
root-running image must be handed over once:
`docker compose run --rm --user root worker chown -R 1000:1000 /data`.
Every Python dependency comes from `worker/requirements.lock`; re-pin with
`make lock-worker` and commit the result when upgrading. CI installs from
the same lock, so a lock that no longer resolves fails there first.

SNAP is not in the image (`sentinel1_backend="snap"` needs a separate image
with `gpt` on `PATH`); `hyp3_rtc` and `pc_rtc` need nothing extra. `s1ard`
is excluded because it is broken (see `roadmap.md`).

### Jobs

- **Isolation**: each job runs in its own child process and process group
  (`JOB_PROCESS_ISOLATION`, on by default). An out-of-memory kill or crash
  fails that job with an explanatory message and leaves the API running;
  the child also dies with the worker instead of running orphaned.
- **Concurrency**: at most `MAX_CONCURRENT_JOBS` (default 1) jobs run at
  once; the rest wait as `PENDING`. One Sentinel-1 SNAP job peaked at
  ~11 GB of RAM, so raise this only on a machine sized for it.
- **Cancellation and timeouts**: `POST /jobs/{id}/cancel` (or "Cancel job"
  in the UI) stops a pending job at once and a running one within about a
  second, including anything it started (SNAP's `gpt`). A job still running
  after `JOB_TIMEOUT_HOURS` (default 12) is stopped and marked `FAILED`.
- **Size limits**: requests beyond `MAX_AOI_KM2`, `MAX_PRODUCTS_PER_SENSOR`
  or `MAX_ESTIMATED_GB` are rejected at submission with HTTP 422.
- **Durability**: job state is mirrored to `<WORKER_OUTPUT_DIR>/jobs.db`;
  a restart marks jobs that were `PENDING`/`RUNNING` as `FAILED` instead of
  losing them. Runs cannot resume mid-way.

### Disk

- A successful job keeps only `result/` and `job.log`; its downloads and
  AOI subsets (`work/`) are deleted (`KEEP_WORK_DIR=true` keeps them).
  Failed jobs keep `work/` for diagnosis.
- Finished jobs older than `JOB_RETENTION_DAYS` (default 30) are deleted
  with all their files, at startup and hourly. `DELETE /jobs/{id}` (or
  "Delete" in the UI) removes one at once.
- Below `MIN_FREE_DISK_GB` (default 20) free in the output directory, new
  submissions get HTTP 507 and queued jobs fail instead of starting.

### Monitoring

- `GET /health` (no auth): liveness, version and git commit; the
  container's `HEALTHCHECK`.
- `GET /health/ready` (auth): free disk, the job database, and a live
  authentication against every configured provider (CDSE OAuth client and
  account, Earthdata with its token expiry, CDS, ADS, OpenAQ). HTTP 503 when
  any check fails; credentials expiring within 7 days are a `warning`.
  Cached for 10 minutes (`?refresh=true` forces a new check). The UI's
  Settings page shows the same report.
- `GET /jobs/{id}` reports `started_at`, `finished_at` and `metrics`
  (`duration_s`, `peak_rss_mb` - the larger of the job and any external
  tool it ran - and `failed_products`); `GET /jobs/{id}/log` returns the job's own log.
- `LOG_FORMAT=json` switches the worker's logs to one JSON object per line
  for a log collector; job processes tag every line with the job id.
- **Daily canary**: `make canary` (or `scripts/canary.py` from cron or a
  systemd timer) checks every credential and runs the smallest real
  Sentinel-3 analysis, exiting non-zero on failure so the scheduler can
  alert; `ARGS=--full` adds Sentinel-2 COGs and downscaling. Results are
  appended to `output/canary/history.jsonl`. `make check-credentials` runs
  only the credential part.
- Every result's `provenance.json` records the `sentinel_analysis` version
  and git commit (`software`) that produced it.

### Security notes

- Keep `.env` and `worker/.env` readable by your user only (`chmod 600`): they hold every
  provider credential.
- The UI stores the worker token in the browser's `localStorage`; anything able to run
  script in that origin could read it. The UI renders no server or user HTML, and the map
  library is kept on a version without known advisories.
- `npm audit` still reports `vite`/`esbuild` advisories that affect only the development
  server (`npm run dev`), not the built UI the worker serves. Fixing them needs Vite 8,
  which needs Node.js 20.19 or newer for local frontend development.
- Requests run as the token holder: parameters such as `downscale.model_options` are passed
  to the models as given, so the token must only go to trusted users.

### Quality gates

- **CI** (`.github/workflows/ci.yml`): ruff, mypy, the `sentinel_analysis`
  tests and package build; the worker tests installed from
  `requirements.lock`; and the frontend's `npm ci`, eslint and build
  (which type-checks with `tsc`).
- Live-data checks (`scripts/validate_*`, the benchmarks, the canary) are
  run by hand or on a schedule, not in CI, since they need credentials and
  download real data.
- **Git hygiene**: `worker/runs/` is in `.gitignore` - it's the worker's
  job-output directory (regenerable by re-running a job), not source.
