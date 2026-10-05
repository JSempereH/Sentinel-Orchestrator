"""Quality flags and masks for Sentinel-3 SLSTR L2 LST."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
import re
from typing import Iterable, Sequence

import numpy as np
import xarray as xr


class LSTExceptionFlag(IntFlag):
    """Official SLSTR L2 LST exception flags (bits 0-10)."""

    ISP_ABSENT = 1 << 0
    PIXEL_ABSENT = 1 << 1
    NOT_DECOMPRESSED = 1 << 2
    NO_SIGNAL = 1 << 3
    SATURATION = 1 << 4
    INVALID_RADIANCE = 1 << 5
    NO_PARAMETERS = 1 << 6
    UNFILLED_PIXEL = 1 << 7
    LST_UNDERFLOW = 1 << 8
    LST_OVERFLOW = 1 << 9
    BIOME = 1 << 10


ALL_EXCEPTION_FLAGS = int(sum(flag.value for flag in LSTExceptionFlag))


class LSTConfidenceFlag(IntFlag):
    """``confidence_in`` bits (SLSTR Land Handbook, Table 6)."""

    COASTLINE = 1 << 0
    OCEAN = 1 << 1
    TIDAL = 1 << 2
    LAND = 1 << 3
    INLAND_WATER = 1 << 4
    UNFILLED = 1 << 5
    COSMETIC = 1 << 8
    DUPLICATE = 1 << 9
    DAY = 1 << 10
    TWILIGHT = 1 << 11
    SUN_GLINT = 1 << 12
    SNOW = 1 << 13
    SUMMARY_CLOUD = 1 << 14
    SUMMARY_POINTING = 1 << 15


# Pixels that are not genuine, independent observations: unfilled, filled
# cosmetically from neighbours, or duplicated by the instrument-grid
# regridding (a duplicate would count twice in area-weighted aggregation).
DEFAULT_CONFIDENCE_REJECT = int(LSTConfidenceFlag.UNFILLED | LSTConfidenceFlag.COSMETIC | LSTConfidenceFlag.DUPLICATE)


@dataclass(frozen=True)
class QualityPolicy:
    """Explicit quality policy; no physical-temperature cutoff is implicit."""

    reject_exceptions: int = ALL_EXCEPTION_FLAGS
    reject_cloud: bool = True
    reject_nonfinite: bool = True
    legacy_confidence_cloud_bit: int | None = None
    cloud_flag_mask: int | None = None
    reject_confidence: int = DEFAULT_CONFIDENCE_REJECT
    # SLSTR views reach ~55 degrees at the swath edges, where angular
    # anisotropy biases LST; 45 degrees is the common practical cut.
    max_view_zenith: float | None = 45.0
    # Dilate the cloud mask by this many pixels: cloud edges and shadows are
    # the usual residual contamination of any cloud mask.
    cloud_buffer_pixels: int = 0

    @classmethod
    def strict(cls) -> "QualityPolicy":
        """Reject all documented exceptions and discovered cloud masks."""

        return cls()

    @classmethod
    def permissive(cls) -> "QualityPolicy":
        """Keep finite LST values while retaining quality variables."""

        return cls(reject_exceptions=0, reject_cloud=False, reject_confidence=0, max_view_zenith=None)


def _integer_flags(values: xr.DataArray) -> xr.DataArray:
    values = values.fillna(0)
    if values.dtype.kind not in "iu":
        values = values.astype("int64")
    return values


def cf_flag_mask(
    flags: xr.DataArray,
    *,
    meanings: Sequence[str] = ("cloud", "cirrus", "fog", "stratus"),
) -> int:
    """Return the bit mask for CF flag meanings matching ``meanings``."""

    masks = np.asarray(flags.attrs.get("flag_masks", []), dtype=np.int64)
    names = str(flags.attrs.get("flag_meanings", "")).split()
    if len(masks) != len(names):
        return 0
    patterns = tuple(re.compile(re.escape(term), re.IGNORECASE) for term in meanings)
    result = 0
    for mask, name in zip(masks, names):
        if any(pattern.search(name) for pattern in patterns):
            result |= int(mask)
    return result


def cf_flag_mask_array(
    flags: xr.DataArray,
    *,
    meanings: Sequence[str] = ("cloud", "cirrus", "fog", "stratus"),
) -> xr.DataArray:
    """Build a boolean mask from CF flag meanings, preserving laziness."""

    mask = cf_flag_mask(flags, meanings=meanings)
    if not mask:
        return xr.zeros_like(flags, dtype=bool).rename("cf_flag_mask")
    return ((_integer_flags(flags) & mask) != 0).rename("cf_flag_mask")


def valid_mask(
    dataset: xr.Dataset,
    policy: QualityPolicy | None = None,
    *,
    lst_name: str = "lst",
) -> xr.DataArray:
    """Build a boolean validity mask from standardized quality variables."""

    policy = policy or QualityPolicy.strict()
    if lst_name not in dataset:
        raise KeyError(f"Dataset does not contain {lst_name!r}")
    lst = dataset[lst_name]
    valid = xr.ones_like(lst, dtype=bool)
    if policy.reject_nonfinite:
        valid &= xr.apply_ufunc(np.isfinite, lst)
    if policy.reject_exceptions and "exception_flags" in dataset:
        flags = _integer_flags(dataset["exception_flags"])
        valid &= (flags & int(policy.reject_exceptions)) == 0
    cloudy = xr.zeros_like(lst, dtype=bool)
    if policy.reject_cloud and "cloud_mask" in dataset:
        cloudy |= dataset["cloud_mask"].astype(bool)
    cloud_flag_mask = policy.cloud_flag_mask or policy.legacy_confidence_cloud_bit
    if policy.reject_cloud and "cloud_flags" in dataset and cloud_flag_mask:
        cloudy |= (_integer_flags(dataset["cloud_flags"]) & cloud_flag_mask) != 0
    if policy.cloud_buffer_pixels > 0:
        cloudy = _dilate(cloudy, policy.cloud_buffer_pixels)
    valid &= ~cloudy
    if policy.reject_confidence and "confidence_flags" in dataset:
        valid &= (_integer_flags(dataset["confidence_flags"]) & int(policy.reject_confidence)) == 0
    if policy.max_view_zenith is not None and "sat_zenith" in dataset:
        valid &= dataset["sat_zenith"].fillna(90.0) <= policy.max_view_zenith
    return valid.rename("valid_mask")


def _dilate(mask: xr.DataArray, pixels: int) -> xr.DataArray:
    """8-connected binary dilation over the last two (y, x) dimensions."""

    values = np.asarray(mask.values, dtype=bool)
    for _ in range(pixels):
        padded = np.pad(values, [(0, 0)] * (values.ndim - 2) + [(1, 1), (1, 1)])
        grown = np.zeros_like(values)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                grown |= padded[..., dy:dy + values.shape[-2], dx:dx + values.shape[-1]]
        values = grown
    return mask.copy(data=values)


def apply_quality_mask(
    dataset: xr.Dataset,
    policy: QualityPolicy | None = None,
    *,
    lst_name: str = "lst",
) -> xr.Dataset:
    """Add ``valid_mask`` and mask continuous LST variables."""

    mask = valid_mask(dataset, policy, lst_name=lst_name)
    result = dataset.copy()
    result["valid_mask"] = mask
    for name in (lst_name, "lst_uncertainty"):
        if name in result:
            result[name] = result[name].where(mask)
    result.attrs["quality_policy"] = "documented L2 LST exception and cloud flags"
    return result


def quality_summary(dataset: xr.Dataset, policy: QualityPolicy | None = None) -> dict[str, int]:
    """Return total, valid and invalid pixel counts."""

    mask = valid_mask(dataset, policy)
    return {
        "pixels_total": int(mask.size),
        "pixels_valid": int(mask.sum().item()),
        "pixels_invalid": int((~mask).sum().item()),
    }


def exception_counts(
    dataset: xr.Dataset,
    flags: Iterable[LSTExceptionFlag] = tuple(LSTExceptionFlag),
) -> dict[str, int]:
    """Count each documented exception bit independently."""

    if "exception_flags" not in dataset:
        return {}
    values = _integer_flags(dataset["exception_flags"])
    return {str(flag.name).lower(): int(((values & flag.value) != 0).sum().item()) for flag in flags}
