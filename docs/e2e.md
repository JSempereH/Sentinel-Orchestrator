# E2E Smoke Test

The `scripts/smoke_e2e.py` script uses one date and one Sentinel-3 scene. It
also downloads one ERA5 variable, one CAMS variable, and up to two OpenAQ
stations.

This tests the complete orchestrator with a small data volume. It deliberately
does not process Sentinel-1/2/5P or run SNAP. Those readers have separate real
product tests in `tests/test_integration.py`.

```bash
uv sync --extra auxiliary --extra optical
uv run --extra auxiliary --extra optical python scripts/smoke_e2e.py
```

Expected output:

```text
E2E ok
variables: [...]
auxiliary: ['cams', 'era5', 'openaq']
```

Files are written to `output/e2e-smoke/`. The directory is ignored by Git and
contains the cache, downloaded products, and manifests.

## Last Validated Run

The live smoke test completed with `E2E ok`. The result included `lst`,
`auxiliary_cams_NO2`, `auxiliary_era5_air_temperature_2m`, and the `cams`,
`era5`, and `openaq` datasets in `AnalysisResult.auxiliary`.

## Troubleshooting

- `401`: missing or invalid credentials.
- `403 required licences not accepted`: accept the dataset license in the CDS/ADS portal.
- `SNAP GPT was not found`: install SNAP for Sentinel-1; this smoke test does not use Sentinel-1.
- `No products found`: change the date or AOI; the catalog may not contain a scene.
