"""Geometry extraction must preserve measured depths and reject invented boundaries."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydrophysics.twin.release import sha256
from hydrophysics.twin.ring_geometry import build, reference_depths


def test_first_complete_survey_preserves_numbered_ring_depths():
    wide = pd.DataFrame({"NO1": [np.nan, 1.5, 1.501],
                         "NO2": [5.0, 20.0, 19.999],
                         "NO10": [np.nan, 300.0, 299.998]},
                        index=pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]))
    got = reference_depths(wide)
    assert got["survey_date"].startswith("2023-02-01")
    assert list(got["ring_depths_m_bgs"]) == ["NO1", "NO2", "NO10"]
    assert list(got["ring_depths_m_bgs"].values()) == [1.5, 20.0, 300.0]


def test_reversed_ring_order_is_rejected():
    wide = pd.DataFrame({"NO1": [20.0], "NO2": [10.0]},
                        index=pd.to_datetime(["2023-02-01"]))
    with pytest.raises(ValueError, match="increasing"):
        reference_depths(wide)


def test_manifest_joins_measured_depths_to_grid_and_hashes_sources(tmp_path):
    pytest.importorskip("pyarrow")
    cache = tmp_path / "clean"
    cache.mkdir()
    rings = cache / "ls-wra-mlcw-obs___41.parquet"  # percent-hex station A
    pd.DataFrame({"NO1": [2.0, 2.1], "NO2": [200.0, 199.9]},
                 index=pd.to_datetime(["2023-01-01", "2023-02-01"])).to_parquet(rings)
    stations = tmp_path / "stations.csv"
    pd.DataFrame({"sub_id": ["A"], "x": [500], "y": [500]}).to_csv(stations, index=False)
    forward = tmp_path / "forward.npz"
    np.savez(forward, nx=1, ny=1, dx=1000., x0=0., y0=0.,
             mask=np.ones((1, 1), dtype=bool))
    report = build(cache, stations, forward)
    assert report["forward_grid_sha256"] == sha256(forward)
    assert report["sites"][0]["grid_active_index"] == 0
    assert report["sites"][0]["ring_depths_m_bgs"] == {"NO1": 2.0, "NO2": 200.0}
    assert report["sites"][0]["source_sha256"] == sha256(rings)
