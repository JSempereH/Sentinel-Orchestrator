"""High-level local Sentinel-3 LST workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import xarray as xr

from .catalog import CDSECatalog, ProductQuery
from ...catalog import ProductRef
from ...config import AOI, ClientConfig
from ...download import CDSEDownloader
from .processing import daily_mean, statistics, to_celsius
from .quality import QualityPolicy, apply_quality_mask, quality_summary
from .reader import read_l2_lst


class Sentinel3LST:
    """Easy-to-use facade for CDSE acquisition and local L2 LST analysis."""

    def __init__(
        self,
        *,
        catalog: CDSECatalog | None = None,
        downloader: CDSEDownloader | None = None,
    ):
        self.catalog = catalog or CDSECatalog()
        self.downloader = downloader

    @classmethod
    def from_cdse(cls, config: ClientConfig | None = None) -> "Sentinel3LST":
        """Build a CDSE client; credentials are only needed for downloads."""

        config = config or ClientConfig.from_env()
        return cls(catalog=CDSECatalog(config), downloader=CDSEDownloader(config))

    def search(
        self,
        aoi: AOI | Mapping[str, float],
        start: str,
        end: str,
        *,
        timeliness: str | None = "NTC",
        platform: str | None = None,
        limit: int = 1000,
    ) -> list[ProductRef]:
        """Search exact Sentinel-3 SLSTR L2 LST products in CDSE."""

        return self.catalog.search(
            ProductQuery(
                aoi=aoi,
                start=start,
                end=end,
                timeliness=timeliness,
                platform=platform,
            ),
            limit=limit,
        )

    def download(self, product: ProductRef, output: str | Path) -> Path:
        """Download one product, requiring a configured authenticated provider."""

        if self.downloader is None:
            raise RuntimeError("Create this client with Sentinel3LST.from_cdse() to download products")
        return self.downloader.download(product, output)

    @staticmethod
    def read(path: str | Path, *, chunks: Mapping[str, int] | None = None) -> xr.Dataset:
        """Read a SAFE directory or archive into a standardized Dataset."""

        return read_l2_lst(path, chunks=chunks)

    @staticmethod
    def combine(paths: Iterable[str | Path], *, chunks: Mapping[str, int] | None = None) -> xr.Dataset:
        """Read and concatenate products along their acquisition time."""

        datasets = [read_l2_lst(path, chunks=chunks) for path in paths]
        if not datasets:
            raise ValueError("At least one product path is required")
        return xr.concat(datasets, dim="time", data_vars="minimal", coords="minimal", compat="override")

    @staticmethod
    def to_celsius(dataset: xr.Dataset) -> xr.Dataset:
        return to_celsius(dataset)

    @staticmethod
    def apply_quality(dataset: xr.Dataset, policy: QualityPolicy | None = None) -> xr.Dataset:
        return apply_quality_mask(dataset, policy)

    @staticmethod
    def daily_mean(dataset: xr.Dataset, policy: QualityPolicy | None = None) -> xr.Dataset:
        return daily_mean(dataset, quality=policy)

    @staticmethod
    def quality_summary(dataset: xr.Dataset, policy: QualityPolicy | None = None) -> dict[str, int]:
        return quality_summary(dataset, policy)

    @staticmethod
    def statistics(dataset: xr.Dataset) -> dict[str, float | int]:
        return statistics(dataset)
