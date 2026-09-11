# Build context must be this repo's root, NOT worker/ - see docker-compose.yml
# at that root, or build manually with:
#   docker build -f worker/Dockerfile -t sentinel-worker .
#   (run from Sentinel-Orchestrator/)

# ── Frontend build stage ─────────────────────────────────────────────────
# Built separately so the final image needs no Node.js at all - only the
# static dist/ output is copied into it below.
FROM node:20-slim AS frontend-build
WORKDIR /app/worker/frontend
COPY worker/frontend/package.json worker/frontend/package-lock.json* ./
RUN npm install
COPY worker/frontend/ ./
RUN npm run build

FROM python:3.12-slim

# rasterio/pyproj/xarray native deps for the sentinel_analysis[optical,sar] extras
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# sentinel_analysis itself (this repo). Neither `s1ard` (Sentinel-1 ARD,
# confirmed broken - see docs/roadmap.md) nor SNAP itself (a large Java
# desktop application this Dockerfile does not attempt to bundle) is
# installed here. `sentinel1_backend="snap"` (the default) needs a separate
# image with SNAP's `gpt` on PATH; `sentinel1_backend="hyp3_rtc"` needs
# neither - it's a pure-Python client for ASF's cloud RTC processing, so
# the lightweight `hyp3` extra is included below.
COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src
# `cloud` is required, not optional here: execute_and_persist() always
# writes the result cube(s) via write_zarr() - a run that reaches
# SUCCEEDED without it just fails at the last step instead.
RUN pip install --no-cache-dir ".[cdse,auxiliary,optical,sar,cloud,hyp3]"

# the worker service
COPY worker/requirements.txt /app/worker/requirements.txt
RUN pip install --no-cache-dir -r /app/worker/requirements.txt

COPY worker/app /app/worker/app
COPY --from=frontend-build /app/worker/frontend/dist /app/worker/frontend/dist
WORKDIR /app/worker

ENV WORKER_OUTPUT_DIR=/data/runs
VOLUME ["/data"]

EXPOSE 8100
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]
