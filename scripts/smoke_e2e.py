"""Small live end-to-end check for CDSE, CDS/ADS and OpenAQ."""

from __future__ import annotations

from pathlib import Path

from sentinel_analysis import (
    AnalysisRequest,
    AnalysisWorkflow,
    AuxiliarySpec,
    ClientConfig,
    get_city,
)


def main() -> int:
    request = AnalysisRequest.for_city(
        get_city("Berlin"),
        "2025-06-19",
        "2025-06-19",
        sensors=("sentinel3",),
        variables=("lst",),
        max_products_per_sensor=1,
        auxiliary=(
            AuxiliarySpec("era5", variables=("air_temperature_2m",)),
            AuxiliarySpec("cams", variables=("NO2",)),
            AuxiliarySpec("openaq", variables=("NO2",), options={"max_locations": 2}),
        ),
    )
    output = Path("output/e2e-smoke")
    result = AnalysisWorkflow(request).execute(output, config=ClientConfig.from_env(), max_workers=1)
    print("E2E ok")
    print("variables:", sorted(result.cube.data_vars))
    print("auxiliary:", sorted((result.auxiliary or {}).keys()))
    print("output:", output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
