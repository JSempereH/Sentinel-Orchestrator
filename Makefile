.DEFAULT_GOAL := help

WORKER_DIR   := worker
WORKER_PORT  ?= 8100
DOCS_PORT    ?= 8001

.PHONY: help install install-worker \
        worker docs docs-build \
        test test-worker test-all \
        lint docker-build docker-up docker-down clean

help: ## Show this help
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

## --- Setup -----------------------------------------------------------------

install: ## Install everything: sentinel_analysis (uv) + sentinel-worker
	uv sync --extra dev --extra auxiliary --extra optical --extra docs
	$(MAKE) install-worker

install-worker: ## Create/refresh the sentinel-worker venv and install its deps
	@if [ ! -d $(WORKER_DIR)/.venv ]; then uv venv --python 3.12 $(WORKER_DIR)/.venv; fi
	uv pip install --python $(WORKER_DIR)/.venv -r $(WORKER_DIR)/requirements.txt -r $(WORKER_DIR)/requirements-dev.txt
	@# The `gdal` Python package (pulled in by the s1ard extra) has no
	@# prebuilt wheels and must match the system's libgdal exactly. Pin it to
	@# that version via a constraint before resolving the rest, or uv picks
	@# the latest PyPI release and the build fails with a version mismatch.
	@command -v gdal-config >/dev/null || { echo "error: gdal-config not found. Install libgdal-dev (see worker/README.md)."; exit 1; }
	@echo "gdal==$$(gdal-config --version).*" > $(WORKER_DIR)/.venv/gdal-constraint.txt
	uv pip install --python $(WORKER_DIR)/.venv -e ".[cdse,auxiliary,optical,sar,s1ard,hyp3,cloud]" --constraint $(WORKER_DIR)/.venv/gdal-constraint.txt
	@# `gdal` built above under uv's normal (isolated) build environment does
	@# not see this venv's numpy, so it silently produces a binding with no
	@# `osgeo._gdal_array` extension (pyroSAR/s1ard need it). Rebuild it
	@# with build isolation off so it picks up the numpy already installed
	@# here - confirmed this is otherwise reproducible, not a one-off.
	uv pip install --python $(WORKER_DIR)/.venv setuptools wheel
	uv pip install --python $(WORKER_DIR)/.venv --no-build-isolation --force-reinstall --no-deps --no-cache "gdal==$$(gdal-config --version).*"
	@$(WORKER_DIR)/.venv/bin/python -c "from osgeo import gdal_array" 2>/dev/null && echo "osgeo._gdal_array OK" || echo "warning: osgeo._gdal_array still missing - pyroSAR/s1ard will fail to import"
	@if [ ! -f $(WORKER_DIR)/.env ]; then cp $(WORKER_DIR)/.env.example $(WORKER_DIR)/.env; echo "created $(WORKER_DIR)/.env - fill in WORKER_API_TOKEN and credentials"; fi

## --- Run ---------------------------------------------------------------------

worker: ## Run the sentinel-worker FastAPI service (:8100)
	cd $(WORKER_DIR) && .venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port $(WORKER_PORT)

docs: ## Serve the sentinel_analysis docs locally with mkdocs (:8001)
	uv run --extra docs mkdocs serve --dev-addr 127.0.0.1:$(DOCS_PORT)

docs-build: ## Build the static docs site into ./site
	uv run --extra docs mkdocs build

## --- Tests ---------------------------------------------------------------

test: ## Run sentinel_analysis test suite
	uv run --extra dev --extra auxiliary --extra optical pytest -q

test-worker: ## Run the sentinel-worker test suite
	cd $(WORKER_DIR) && .venv/bin/pytest

test-all: test test-worker ## Run every test suite in this repo

## --- Lint ------------------------------------------------------------------

lint: ## Run ruff on this repo
	uv run --with ruff ruff check .

## --- Docker (alternative to bare venvs; see docker-compose.yml) ------------

docker-build: ## Build the sentinel-worker container image
	docker compose build

docker-up: ## Run the sentinel-worker in a container (:8100)
	docker compose up

docker-down: ## Stop and remove the container started by docker-up
	docker compose down

## --- Housekeeping ----------------------------------------------------------

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -not -path "*/.venv/*" -not -path "*/node_modules/*" -exec rm -rf {} +
	rm -rf .pytest_cache site dist
