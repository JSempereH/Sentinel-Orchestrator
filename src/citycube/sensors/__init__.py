"""Sensor-specific acquisition and processing adapters."""

from .sentinel5p import Sentinel5PCatalog, Sentinel5PReadConfig, grid_s5p, read_s5p_l2

__all__ = ["Sentinel5PCatalog", "Sentinel5PReadConfig", "grid_s5p", "read_s5p_l2"]
