from __future__ import annotations

import pytest

from citycube import ProductRef, TemporalPair, select_temporal_pair


def _product(product_id: str, start: str, cloud_cover: float | None) -> ProductRef:
    return ProductRef(
        product_id=product_id,
        name=product_id,
        product_type="S2MSI1C",
        start_datetime=start,
        end_datetime=start,
        timeliness=None,
        online=True,
        download_url=f"https://example.test/{product_id}",
        metadata={"attributes": {"cloudCover": cloud_cover}} if cloud_cover is not None else {"attributes": {}},
    )


def test_select_temporal_pair_picks_clearest_scene_in_each_window():
    products = [
        _product("ref-cloudy", "2025-05-01T10:00:00Z", 60.0),
        _product("ref-clear", "2025-05-15T10:00:00Z", 5.0),
        _product("too-old", "2025-01-01T10:00:00Z", 0.0),
        _product("det-cloudy", "2025-08-05T10:00:00Z", 40.0),
        _product("det-clear", "2025-08-09T10:00:00Z", 10.0),
        _product("after-target", "2025-08-13T10:00:00Z", 0.0),
    ]

    pair = select_temporal_pair(
        products,
        "2025-08-12",
        detection_lookback_days=7,
        reference_lookback_days=(30, 120),
        cloud_cover_max=25.0,
    )

    assert isinstance(pair, TemporalPair)
    assert pair.reference.product_id == "ref-clear"
    assert pair.detection.product_id == "det-clear"
    assert pair.gap_days == pytest.approx(86.0, abs=1e-6)


def test_select_temporal_pair_ties_break_towards_most_recent():
    products = [
        _product("det-early", "2025-08-06T10:00:00Z", 5.0),
        _product("det-late", "2025-08-09T10:00:00Z", 5.0),
        _product("ref", "2025-06-01T10:00:00Z", 5.0),
    ]

    pair = select_temporal_pair(products, "2025-08-12", detection_lookback_days=7)

    assert pair.detection.product_id == "det-late"


def test_select_temporal_pair_keeps_products_with_unknown_cloud_cover():
    products = [
        _product("det-unknown", "2025-08-09T10:00:00Z", None),
        _product("ref-unknown", "2025-06-01T10:00:00Z", None),
    ]

    pair = select_temporal_pair(products, "2025-08-12", detection_lookback_days=7)

    assert pair.detection.product_id == "det-unknown"
    assert pair.reference.product_id == "ref-unknown"


def test_select_temporal_pair_raises_when_a_window_is_empty():
    products = [_product("det-clear", "2025-08-09T10:00:00Z", 5.0)]

    with pytest.raises(ValueError, match="reference"):
        select_temporal_pair(products, "2025-08-12", detection_lookback_days=7)


def test_select_temporal_pair_rejects_invalid_windows():
    with pytest.raises(ValueError):
        select_temporal_pair([], "2025-08-12", detection_lookback_days=0)
    with pytest.raises(ValueError):
        select_temporal_pair([], "2025-08-12", reference_lookback_days=(120, 30))
