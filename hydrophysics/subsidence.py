"""Subsidence science for the Choushui explorer: IDW heads, MLCW compaction, Sk fit.

Pure NumPy / pandas (+ pyarrow for parquet). No plotly/geopandas here so the science is
importable and testable on its own. See
docs/superpowers/specs/2026-06-19-choushui-head-subsidence-explorer-design.md.
"""

from __future__ import annotations

import glob  # noqa: F401
import os  # noqa: F401
import re  # noqa: F401

import numpy as np  # noqa: F401
import pandas as pd

from .data import GWData  # noqa: F401


def load_mlcw_stations(path) -> pd.DataFrame:
    """Read the 14 MLCW coordinates -> DataFrame[sub_id, x, y, lon, lat] (EPSG:3826)."""
    df = pd.read_csv(path)
    return df[["sub_id", "x", "y", "lon", "lat"]]
