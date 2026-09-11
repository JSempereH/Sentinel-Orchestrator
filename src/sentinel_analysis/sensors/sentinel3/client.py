"""High level openEO client for Sentinel-3 SLSTR L2 LST."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

try:
    import openeo
except (ImportError, ModuleNotFoundError):  # pragma: no cover - depends on optional client compatibility
    openeo = None

if TYPE_CHECKING:
    from openeo.rest.datacube import DataCube
else:
    DataCube = Any

from ...config import AOI, ClientConfig, extent_from_mapping
from .processing import statistics as local_statistics
from .quality import QualityPolicy, apply_quality_mask
from .reader import read_l2_lst


COLLECTION_ID = "SENTINEL3_SLSTR_L2_LST"
LST_BAND = "LST"
CONFIDENCE_BAND = "confidence_in"
# Kept only for compatibility with older openEO exports that used confidence_in.
# Official L2 LST products must be decoded through quality.py instead.
CLOUD_BIT = 1 << 14


def _require_openeo():
    if openeo is None:
        raise RuntimeError("The openEO client is unavailable; install a compatible openeo/pystac environment")
    return openeo


class Sentinel3LSTClient:
    """Build and execute server-side Sentinel-3 LST datacube workflows.

    The client loads only the requested area, period and bands. Processing is
    kept in the openEO graph so the backend can optimize execution before any
    result is transferred to the local machine.
    """

    def __init__(self, config: ClientConfig):
        self.config = config
        self.connection = _require_openeo().connect(config.backend_url)
        self.connection.authenticate_oidc_client_credentials(
            client_id=config.client_id,
            client_secret=config.client_secret,
        )

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> "Sentinel3LSTClient":
        """Create the remote openEO client from environment credentials."""

        return cls(ClientConfig.from_env(env_file))

    def load(
        self,
        aoi: AOI | dict[str, float],
        temporal_extent: Sequence[str],
    ) -> DataCube:
        """Load the processed Sentinel-3 LST collection as a datacube."""

        if len(temporal_extent) != 2 or temporal_extent[0] > temporal_extent[1]:
            raise ValueError("temporal_extent must contain an ordered start and end date")
        spatial_extent = (
            aoi.as_extent() if isinstance(aoi, AOI) else extent_from_mapping(aoi)
        )
        return self.connection.load_collection(
            COLLECTION_ID,
            spatial_extent=spatial_extent,
            temporal_extent=list(temporal_extent),
            bands=[LST_BAND, CONFIDENCE_BAND],
        )

    @staticmethod
    def to_celsius(cube: DataCube) -> DataCube:
        """Convert LST from Kelvin to Celsius in the process graph."""

        return cube - 273.15

    def processed_cube(
        self,
        aoi: AOI | dict[str, float],
        temporal_extent: Sequence[str],
    ) -> DataCube:
        """Load LST and convert it to Celsius in the backend graph."""

        return self.to_celsius(
            self.load(aoi, temporal_extent).filter_bands([LST_BAND])
        )

    def quality_cube(
        self,
        aoi: AOI | dict[str, float],
        temporal_extent: Sequence[str],
    ) -> DataCube:
        """Load LST and confidence_in for local quality processing."""

        return self.load(aoi, temporal_extent)

    @staticmethod
    def process_netcdf(
        input_path: str | Path,
        *,
        legacy_cloud_bit: int | None = CLOUD_BIT,
        min_celsius: float | None = None,
        max_celsius: float | None = None,
    ) -> dict[str, float | int]:
        """Process a downloaded openEO NetCDF, retaining legacy compatibility.

        New SAFE L2 LST products should use :func:`read_l2_lst` and the
        official exception flags. ``legacy_cloud_bit`` exists for old exports
        containing ``confidence_in`` and can be set to ``None`` to disable it.
        """

        import numpy as np
        import xarray as xr

        with xr.open_dataset(input_path) as dataset:
            lst = np.asarray(dataset[LST_BAND].values, dtype=float) - 273.15
            valid = np.isfinite(lst)
            if "exception_flags" in dataset:
                exception = np.asarray(dataset["exception_flags"].values, dtype=np.int64)
                valid &= (exception & int(QualityPolicy.strict().reject_exceptions)) == 0
            elif CONFIDENCE_BAND in dataset and legacy_cloud_bit is not None:
                confidence = np.asarray(dataset[CONFIDENCE_BAND].values, dtype=np.int64)
                valid &= (confidence & legacy_cloud_bit) == 0
            if min_celsius is not None:
                valid &= lst >= min_celsius
            if max_celsius is not None:
                valid &= lst <= max_celsius
        values = lst[valid]
        if values.size == 0:
            raise ValueError("The downloaded cube contains no valid LST pixels")
        return {
            "pixels_total": int(lst.size),
            "pixels_valid": int(values.size),
            "mean_celsius": float(values.mean()),
            "median_celsius": float(np.median(values)),
            "min_celsius": float(values.min()),
            "max_celsius": float(values.max()),
        }

    @staticmethod
    def read_l2_lst(path: str | Path, *, chunks: dict[str, int] | None = None):
        """Read a local Sentinel-3 SLSTR L2 LST SAFE product."""

        return read_l2_lst(path, chunks=chunks)

    @staticmethod
    def apply_quality(dataset, policy: QualityPolicy | None = None):
        """Apply official L2 LST exception flags to a local Dataset."""

        return apply_quality_mask(dataset, policy)

    @staticmethod
    def local_statistics(dataset) -> dict[str, float | int]:
        """Compute statistics from a standardized local Dataset."""

        return local_statistics(dataset)

    @staticmethod
    def daily_mean(cube: DataCube, start: str, end: str) -> DataCube:
        """Aggregate the cube to one temporal mean image per calendar day."""

        first = date.fromisoformat(start)
        last = date.fromisoformat(end)
        if first > last:
            raise ValueError("start must not be after end")
        intervals: list[list[str]] = []
        labels: list[str] = []
        current = first
        while current <= last:
            intervals.append([current.isoformat(), (current + timedelta(days=1)).isoformat()])
            labels.append(current.isoformat())
            current += timedelta(days=1)
        return cube.aggregate_temporal(
            intervals=intervals,
            reducer=lambda data: data.mean(),
            labels=labels,
        )

    @staticmethod
    def save_sync(cube: DataCube, output_path: str | Path, file_format: str = "NetCDF") -> None:
        """Run a small workflow synchronously and save the result."""

        cube.save_result(format=file_format).download(str(output_path))

    @staticmethod
    def submit_batch(
        cube: DataCube,
        title: str = "Sentinel-3 LST analysis",
        file_format: str = "NetCDF",
    ):
        """Start a larger workflow without blocking until completion."""

        job = cube.save_result(format=file_format).create_job(title=title)
        job.start()
        return job

    @staticmethod
    def download_batch(job, output_folder: str | Path) -> None:
        """Download the results of a finished batch job."""

        Path(output_folder).mkdir(parents=True, exist_ok=True)
        job.download_results(str(output_folder))
