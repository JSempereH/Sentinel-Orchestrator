# Workflows

## Declarative Request

```python
from sentinel_analysis import AnalysisRequest, AuxiliarySpec, get_city

request = AnalysisRequest.for_city(
    get_city("Berlin"),
    "2025-06-19",
    "2025-06-19",
    sensors=("sentinel3",),
    auxiliary=(
        AuxiliarySpec("era5", variables=("air_temperature_2m",)),
        AuxiliarySpec("cams", variables=("NO2",)),
        AuxiliarySpec("openaq", variables=("NO2",), options={"max_locations": 2}),
    ),
)
```

## Execute

```python
from sentinel_analysis import AnalysisWorkflow, ClientConfig

result = AnalysisWorkflow(request).execute(
    "output/berlin",
    config=ClientConfig.from_env(),
)
```

## CLI

```bash
sentinel-analysis plan request.json
sentinel-analysis discover request.json
sentinel-analysis auxiliary request.json output/auxiliary
```

`plan` does not access the network. `discover` queries Sentinel catalogs.
`auxiliary` downloads only the configured auxiliary sources.
