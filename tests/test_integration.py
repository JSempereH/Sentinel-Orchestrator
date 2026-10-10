"""Optional real-product integration tests.

Run with ``CITYCUBE_RUN_INTEGRATION=1`` and paths to local products.
The suite deliberately never downloads large satellite archives implicitly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from citycube import (
    AnalysisGrid,
    Sentinel5PReadConfig,
    get_city,
    grid_l2_lst,
    read_l2_lst,
    read_s1_ard,
    read_s1_grd,
    read_s2_l1c,
    read_s2_l2a,
    read_s5p_l2,
    sentinel1_indices,
    sentinel2_indices,
)


pytestmark = pytest.mark.integration


def _product(name: str) -> Path:
    if os.getenv("CITYCUBE_RUN_INTEGRATION") != "1":
        pytest.skip("Set CITYCUBE_RUN_INTEGRATION=1 to run real-product tests")
    value = os.getenv(name)
    if not value:
        pytest.skip(f"Set {name} to a local product path")
    path = Path(value)
    if not path.exists():
        pytest.fail(f"Configured product does not exist: {path}")
    return path


def test_real_sentinel1_grd():
    """The default `sentinel1_backend="snap"` path - a real
    `process_s1_grd`-produced terrain-corrected sigma0 GeoTIFF, not the
    small synthetic one `test_sentinel1_processing_graph_and_backscatter_reader`
    already covers as a unit test."""

    sentinel1_indices(read_s1_grd(_product("SENTINEL1_GRD_TIF_PATH")))


def test_real_sentinel1_ard():
    read_s1_ard(_product("SENTINEL1_ARD_PATH"))


def test_real_sentinel2_l2a():
    sentinel2_indices(read_s2_l2a(_product("SENTINEL2_SAFE_PATH")))


def test_real_sentinel2_l1c():
    """The methane-detection product level (Nat. Commun.
    s41467-024-47754-y) - a distinct SAFE layout/manifest from L2A above
    (no SCL, no R10m/R20m/R60m resolution triplicates)."""

    dataset = read_s2_l1c(_product("SENTINEL2_L1C_SAFE_PATH"))
    assert dataset.attrs["product_type"] == "S2MSI1C"


def test_real_sentinel3_georeferencing():
    dataset = read_l2_lst(_product("SENTINEL3_SAFE_PATH"))
    city = get_city(os.getenv("CITYCUBE_CITY", "Berlin"))
    grid = AnalysisGrid.for_city(city, resolution_m=1000)
    result = grid_l2_lst(dataset, grid)
    assert result["lst_observation_count"].sum().item() > 0


def test_real_sentinel5p_product():
    gas = os.getenv("SENTINEL5P_GAS", "NO2")
    read_s5p_l2(_product("SENTINEL5P_PATH"), config=Sentinel5PReadConfig(gas=gas))
