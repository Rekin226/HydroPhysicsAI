"""CPU unit tests of the S1b r3 fix-round helpers (synthetic values only, no real data)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from s1_rings.s1b import s1b_io as IO1  # noqa: E402
from s1_rings.s1b import s1b_r3f_symmetry as SYM  # noqa: E402


# ----------------------------------------------------------------------------- fix round
def _gnss_fake():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    daily = pd.DataFrame({"family": "wra", "station": "AAAA", "date": dates,
                          "value": np.zeros(10), "flag": 0})
    steps = pd.DataFrame({"family": ["wra", "wra"], "station": ["AAAA", "AAAA"],
                          "date": ["2020-01-04", "2020-01-08"], "size_m": [0.01, -0.03],
                          "se_m": [0.001, 0.002],
                          "basis": ["own_series", "metadata_or_catalogue"]})
    return {"daily": daily, "steps": steps, "stations": None, "common_mode": None}


def test_catalogue_all_adds_back_detector_steps_only():
    g = _gnss_fake()
    h = SYM.catalogue_all(g)
    v = h["daily"]["value"].to_numpy()
    assert np.allclose(v[:3], 0.0) and np.allclose(v[3:], 0.01)     # own-series step added back
    assert list(h["steps"]["basis"]) == ["metadata_or_catalogue"]    # catalogue step kept
    assert np.allclose(g["daily"]["value"].to_numpy(), 0.0)         # input not modified


def test_catalogue_all_matches_registered_in_gap_rule_inside_gap():
    g = _gnss_fake()
    gap = (pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-06"))   # holds the detector step
    t1, s1, v1 = IO1.gnss_series(g, "wra", "AAAA", flags=(0,), steps_mode="catalogue_in_gap",
                                 gap=gap)
    t2, s2, v2 = IO1.gnss_series(SYM.catalogue_all(g), "wra", "AAAA", flags=(0,),
                                 steps_mode="product", gap=gap)
    assert np.allclose(s1, s2) and math.isclose(v1, v2)
