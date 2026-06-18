"""Tests for the Choushui head + subsidence explorer (subsidence.py + explorer.py)."""

from __future__ import annotations

import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
import pytest


def test_load_mlcw_stations():
    from hydrophysics.subsidence import load_mlcw_stations
    from hydrophysics.config import default_config

    cfg = default_config()
    path = cfg.data_dir / "mlcw_stations.csv"
    if not path.exists():
        pytest.skip("mlcw_stations.csv only present alongside the real data")
    df = load_mlcw_stations(path)
    assert list(df.columns[:3]) == ["sub_id", "x", "y"]
    assert len(df) == 14
    assert df["x"].between(170000, 210000).all()
    assert df["y"].between(2600000, 2660000).all()
