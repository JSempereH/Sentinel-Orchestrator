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


@dataclass(frozen=True)
class QualityPolicy:
    """Explicit quality policy; no physical-temperature cutoff is implicit."""

    reject_exceptions: int = ALL_EXCEPTION_FLAGS
    reject_cloud: bool = True
    reject_nonfinite: bool = True
    legacy_confidence_cloud_bit: int | None = None
    cloud_flag_mask: int | None = None

    @classmethod
    def strict(cls) -> "QualityPolicy":
        """Reject all documented exceptions and discovered cloud masks."""

        return cls()

    @classmethod
    def permissive(cls) -> "QualityPolicy":
        """Keep finite LST values while retaining quality variables."""

        return cls(reject_exceptions=0, reject_cloud=False)


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
    if policy.reject_cloud and "cloud_mask" in dataset:
        valid &= ~dataset["cloud_mask"].astype(bool)
    cloud_flag_mask = policy.cloud_flag_mask or policy.legacy_confidence_cloud_bit
    if policy.reject_cloud and "cloud_flags" in dataset and cloud_flag_mask:
        flags = _integer_flags(dataset["cloud_flags"])
        valid &= (flags & cloud_flag_mask) == 0
    return valid.rename("valid_mask")


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
