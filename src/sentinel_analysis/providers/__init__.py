"""Reproducible auxiliary data providers for Sentinel analyses."""

from .base import (
    AUXILIARY_PROVIDERS,
    AuxiliaryArtifact,
    AuxiliaryProviderError,
    AuxiliarySpec,
    artifact_from_path,
    open_gridded_dataset,
)
from .cams import CAMSConfig, CAMSProvider
from .carbon_mapper import CarbonMapperConfig, CarbonMapperProvider
from .era5 import ERA5Config, ERA5Provider
from .openaq import OpenAQConfig, OpenAQInterpolationConfig, OpenAQProvider

__all__ = [
    "AUXILIARY_PROVIDERS",
    "AuxiliaryArtifact",
    "AuxiliaryProviderError",
    "AuxiliarySpec",
    "CAMSConfig",
    "CAMSProvider",
    "CarbonMapperConfig",
    "CarbonMapperProvider",
    "ERA5Config",
    "ERA5Provider",
    "OpenAQConfig",
    "OpenAQInterpolationConfig",
    "OpenAQProvider",
    "artifact_from_path",
    "open_gridded_dataset",
]
