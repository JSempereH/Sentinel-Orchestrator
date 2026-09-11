# Real Product Fixtures

The repository does not commit satellite archives. To run the real-product
integration tests, set `SENTINEL_ANALYSIS_RUN_INTEGRATION=1` and provide local
paths:

```bash
export SENTINEL1_ARD_PATH=/data/s1ard/ARD/...
export SENTINEL2_SAFE_PATH=/data/S2A_....SAFE
export SENTINEL3_SAFE_PATH=/data/S3A_....SEN3
export SENTINEL5P_PATH=/data/S5P_OFFL_L2__NO2____....nc
export SENTINEL5P_GAS=NO2
uv run pytest -m integration
```

Use one real product from each processing baseline supported by the project.
Tests skip rather than downloading products implicitly.
