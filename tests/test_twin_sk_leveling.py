import numpy as np
import pandas as pd
import pytest

from hydrophysics.twin import sk_leveling


class _FakeData:
    """Minimal GWData stand-in: two wells, monthly heads declining linearly."""

    def __init__(self):
        self.dates = pd.date_range("2012-01-31", periods=96, freq="ME")
        decline = np.linspace(0.0, -12.0, 96)
        self.target = np.stack([decline, decline])                       # (W, T)
        self.attrs = pd.DataFrame({"tm_x": [175000.0, 185000.0],
                                   "tm_y": [2615000.0, 2625000.0]})


def test_build_pairs_aligns_drawdown_and_subsidence_at_survey_dates():
    data = _FakeData()
    times = pd.DatetimeIndex([pd.Timestamp(f"{y}-06-30") for y in range(2013, 2019)])
    sub = {"S1": pd.Series(np.linspace(0.0, 0.06, len(times)), index=times)}
    xy = {"S1": (180000.0, 2620000.0)}

    pairs = sk_leveling.build_pairs(data, sub, xy)
    D, C = pairs["S1"]
    assert D.shape == C.shape == (len(times),)
    assert D[0] == pytest.approx(0.0)          # drawdown re-zeroed to first survey
    assert C[0] == pytest.approx(0.0)
    assert np.all(np.diff(D) >= -1e-9)         # heads fall -> drawdown accumulates
    assert D[-1] > 0.0


def test_build_pairs_skips_sites_with_no_overlapping_surveys():
    data = _FakeData()
    times = pd.DatetimeIndex([pd.Timestamp("2001-06-30"), pd.Timestamp("2002-06-30")])
    sub = {"OLD": pd.Series([0.0, 0.01], index=times)}
    pairs = sk_leveling.build_pairs(data, sub, {"OLD": (180000.0, 2620000.0)})
    assert pairs == {}
