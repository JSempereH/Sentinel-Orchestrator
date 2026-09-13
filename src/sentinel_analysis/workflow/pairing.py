"""Reference/detection acquisition pairing for change-detection workflows.

Mirrors the pairing rule in Varon et al. 2024 (Nat. Commun.
s41467-024-47754-y): a change-detection model compares one "reference"
Sentinel-2 acquisition from well before a candidate event to one
"detection" acquisition close to it, both filtered to an acceptable cloud
cover, so the pixel difference between the two brackets a genuine change
rather than sensor noise or illumination drift. This is workflow-level
logic - it operates on `ProductRef`s a sensor catalog already returned, not
on how any one sensor is read.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

from ..catalog import ProductRef


@dataclass(frozen=True)
class TemporalPair:
    """One reference/detection acquisition pair selected for change detection."""

    reference: ProductRef
    detection: ProductRef

    @property
    def gap_days(self) -> float:
        """Days between the reference and detection acquisitions."""

        return (_as_datetime(self.detection.start_datetime) - _as_datetime(self.reference.start_datetime)).total_seconds() / 86400.0


def _as_datetime(value: str | date | datetime | None) -> datetime:
    if value is None:
        raise ValueError("ProductRef has no start_datetime to pair on")
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _clear_enough(product: ProductRef, cloud_cover_max: float | None) -> bool:
    return cloud_cover_max is None or product.cloud_cover is None or product.cloud_cover <= cloud_cover_max


def _in_window(product: ProductRef, start: datetime, end: datetime) -> bool:
    if product.start_datetime is None:
        return False
    acquired = _as_datetime(product.start_datetime)
    if acquired.tzinfo and start.tzinfo is None:
        start, end = start.replace(tzinfo=acquired.tzinfo), end.replace(tzinfo=acquired.tzinfo)
    elif acquired.tzinfo is None and start.tzinfo:
        acquired = acquired.replace(tzinfo=start.tzinfo)
    return start <= acquired <= end


def _clearest_then_latest(candidates: list[ProductRef]) -> ProductRef | None:
    """Prefer the clearest scene, breaking ties by recency (closest to the window's end)."""

    if not candidates:
        return None
    return min(
        candidates,
        key=lambda product: (
            product.cloud_cover if product.cloud_cover is not None else 101.0,
            -_as_datetime(product.start_datetime).timestamp(),
        ),
    )


def select_temporal_pair(
    products: Iterable[ProductRef],
    detection_date: str | date | datetime,
    *,
    detection_lookback_days: int = 7,
    reference_lookback_days: tuple[int, int] = (30, 120),
    cloud_cover_max: float | None = 25.0,
) -> TemporalPair:
    """Select one reference and one detection acquisition around a target date.

    ``detection_date`` is the date of the event being investigated (e.g. a
    reported methane leak). The detection acquisition is the clearest
    product in the ``detection_lookback_days`` window immediately before
    it; the reference acquisition is the clearest product further back, in
    the ``reference_lookback_days`` window (as ``(min_days, max_days)``
    before the detection date - the paper's default is roughly one to four
    months prior). Both windows are filtered to ``cloud_cover_max`` first
    (``None`` disables the filter; a product with unknown cloud cover is
    kept rather than discarded).

    Raises ``ValueError`` if either window has no qualifying acquisition.
    """

    if detection_lookback_days <= 0:
        raise ValueError("detection_lookback_days must be positive")
    min_days, max_days = reference_lookback_days
    if not 0 < min_days < max_days:
        raise ValueError("reference_lookback_days must satisfy 0 < min_days < max_days")

    target = _as_datetime(detection_date)
    detection_start = target - timedelta(days=detection_lookback_days)
    reference_start = target - timedelta(days=max_days)
    reference_end = target - timedelta(days=min_days)

    candidates = [product for product in products if _clear_enough(product, cloud_cover_max)]
    detection = _clearest_then_latest([p for p in candidates if _in_window(p, detection_start, target)])
    reference = _clearest_then_latest([p for p in candidates if _in_window(p, reference_start, reference_end)])

    if detection is None:
        raise ValueError(
            f"No detection acquisition within {detection_lookback_days} day(s) before {target.date()} "
            f"meets cloud_cover_max={cloud_cover_max}"
        )
    if reference is None:
        raise ValueError(
            f"No reference acquisition {min_days}-{max_days} days before {target.date()} meets cloud_cover_max={cloud_cover_max}"
        )
    return TemporalPair(reference=reference, detection=detection)
