# Build context must be this repo's root, NOT worker/ - see docker-compose.yml
# at that root, or build manually with:
#   docker build -f worker/Dockerfile -t sentinel-worker .
#   (run from Sentinel-Orchestrator/)

# ── Frontend build stage ─────────────────────────────────────────────────
# Built separately so the final image needs no Node.js at all - only the
# static dist/ output is copied into it below.
FROM node:20-slim AS frontend-build
WORKDIR /app/worker/frontend
COPY worker/frontend/package.json worker/frontend/package-lock.json ./
RUN npm ci
COPY worker/frontend/ ./
RUN npm run build

FROM python:3.12-slim

# rasterio/pyproj/xarray native deps for the sentinel_analysis[optical,sar] extras
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Every Python dependency at the exact version in worker/requirements.lock
# (regenerate with `make lock-worker`), so two builds of the same commit
# install the same thing. Installed before copying the source so this
# layer is cached across code-only changes.
COPY worker/requirements.lock /app/worker/requirements.lock
RUN pip install --no-cache-dir -r /app/worker/requirements.lock

# sentinel_analysis itself (this repo). Neither `s1ard` (Sentinel-1 ARD,
# confirmed broken - see docs/roadmap.md) nor SNAP itself (a large Java
# desktop application this Dockerfile does not attempt to bundle) is
# installed here. `sentinel1_backend="snap"` (the default) needs a separate
# image with SNAP's `gpt` on PATH; `sentinel1_backend="hyp3_rtc"` needs
# neither - it's a pure-Python client for ASF's cloud RTC processing, so
# the lightweight `hyp3` extra is included below.
# The lock covers the extras it needs: `cloud` (every run writes Zarr),
# `landsat` (signs Planetary Computer URLs for landsat, pc_rtc and
# stac_cog), `ecostress`, and `ml` (per-scene downscaling's default model).
COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src
RUN pip install --no-cache-dir --no-deps .

COPY worker/app /app/worker/app
COPY --from=frontend-build /app/worker/frontend/dist /app/worker/frontend/dist
WORKDIR /app/worker

# Recorded in /health and in every result's provenance.json (there is no
# .git inside the image). `make docker-build` passes the current commit.
ARG GIT_COMMIT=unknown
ENV SENTINEL_ANALYSIS_GIT_COMMIT=${GIT_COMMIT}

# Run unprivileged. /data is created owned by that user before VOLUME so a
# new named volume inherits the ownership.
RUN useradd --create-home --uid 1000 worker \
    && mkdir -p /data/runs && chown -R worker:worker /data
USER worker

ENV WORKER_OUTPUT_DIR=/data/runs
VOLUME ["/data"]

EXPOSE 8100
# Liveness only (no credentials or network): /health/ready is the deeper
# check and needs the API token.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/health', timeout=4)"]
# Keep-alive above the UI's 5 s polling interval: at uvicorn's default of 5 s
# a poll could reuse a connection the server was just closing.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100", "--timeout-keep-alive", "30"]
