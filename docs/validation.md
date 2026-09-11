# Validation

## Local Suite

```bash
uv run --extra dev --extra auxiliary --extra optical pytest -q
uv build
```

Synthetic tests cover contracts, units, cache, regridding, fusion, OpenAQ v3,
and serialization.

## Real Products

Real-product tests do not download files automatically:

```bash
export SENTINEL_ANALYSIS_RUN_INTEGRATION=1
export SENTINEL1_GRD_TIF_PATH=/data/process_s1_grd_output_sigma0_tc.tif  # sentinel1_backend="snap" (default)
export SENTINEL1_ARD_PATH=/data/s1ard/ARD/...                            # sentinel1_backend="s1ard" (confirmed broken, see docs/roadmap.md)
export SENTINEL2_SAFE_PATH=/data/S2.SAFE
export SENTINEL3_SAFE_PATH=/data/S3.SEN3
export SENTINEL5P_PATH=/data/S5P_NO2.nc
uv run --extra dev --extra optical --extra sar pytest -m integration -q
```

## E2E Criteria

- CDSE, CDS/ADS, and OpenAQ tokens are accepted;
- at least one Sentinel-3 scene is downloaded;
- ERA5, CAMS, and OpenAQ variables are present;
- CRS, units, timestamps, and provenance are preserved;
- a SHA-256 manifest is written;
- `result.cube` and `result.auxiliary` are non-empty.

The minimal live smoke test has been executed successfully with one Sentinel-3
scene, ERA5, CAMS, and OpenAQ. Sentinel-1/2/5P tests remain optional because
they require local products, and `sentinel1_backend="snap"` (the default)
requires SNAP GPT to produce `SENTINEL1_GRD_TIF_PATH` in the first place -
see `docs/setup.md` for the three `sentinel1_backend` options.
