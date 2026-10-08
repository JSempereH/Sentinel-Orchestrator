"""Reproducible auxiliary data providers for Sentinel analyses."""

from .base import (
    AUXILIARY_PROVIDERS,
    AuxiliaryArtifact,
    AuxiliaryProvider,
    AuxiliaryProviderError,
    AuxiliarySpec,
    artifact_from_path,
    open_gridded_dataset,
)
from .cams import CAMSConfig, CAMSProvider
from .era5 import ERA5Config, ERA5Provider
from .openaq import OpenAQConfig, OpenAQInterpolationConfig, OpenAQProvider

# Provider name -> zero-argument factory (credentials come from the
# environment). Every name in AUXILIARY_PROVIDERS must appear here.
AUXILIARY_PROVIDER_FACTORIES = {
    "era5": ERA5Provider,
    "cams": CAMSProvider,
    "openaq": OpenAQProvider,
}

__all__ = [
    "AUXILIARY_PROVIDERS",
    "AUXILIARY_PROVIDER_FACTORIES",
    "AuxiliaryArtifact",
    "AuxiliaryProvider",
    "AuxiliaryProviderError",
    "AuxiliarySpec",
    "CAMSConfig",
    "CAMSProvider",
    "ERA5Config",
    "ERA5Provider",
    "OpenAQConfig",
    "OpenAQInterpolationConfig",
    "OpenAQProvider",
    "artifact_from_path",
    "open_gridded_dataset",
]
