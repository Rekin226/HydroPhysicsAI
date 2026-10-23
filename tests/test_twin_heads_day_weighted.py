"""Monthly heads as the mean of daily means (``--day-weighted-months``, 2026-10-20).

The historical monthly value is the mean of every sample, so in February 2019, when logging
switched from hourly to 10-minute on the 20th, the last nine days count six times. The opt-in
day-weighted mean weights every observed day alike; the default stays bit-for-bit historical,
and a calibration records the switch so every consumer rebuilds the same heads."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from hydrophysics.twin.calibrate_flow import _input_opts_record  # noqa: E402
from hydrophysics.twin.heads import build_head_field  # noqa: E402
from hydrophysics.twin.inputs import input_options  # noqa: E402


def _cadence_switch_well(tmp_path):
    hourly = pd.date_range("2019-02-01", "2019-02-20", freq="h", inclusive="left")
    tenmin = pd.date_range("2019-02-20", "2019-03-01", freq="10min", inclusive="left")
    s = pd.concat([pd.Series(10.0, index=hourly), pd.Series(11.0, index=tenmin)])
    wells = tmp_path / "wells"
    wells.mkdir()
    pd.DataFrame({"value": s}).to_parquet(wells / "W1.parquet")
    stations = pd.DataFrame({"sid": ["W1"], "GroundwaterLayerCode": ["2"],
                             "LocationByTWD97": ["200000 2640000"]})
    return str(wells), stations


def test_day_weighted_month_is_the_mean_of_daily_means(tmp_path):
    wells, stations = _cadence_switch_well(tmp_path)
    kw = dict(t0="2019-02-01", t1="2019-03-01")
    legacy = build_head_field(wells, stations, **kw).heads[0, 0]
    daily = build_head_field(wells, stations, day_weighted=True, **kw).heads[0, 0]
    # historical: 456 hourly samples at 10 m and 1296 ten-minute samples at 11 m
    assert legacy == pytest.approx((456 * 10.0 + 1296 * 11.0) / 1752)
    # day-weighted: 19 days at 10 m and 9 days at 11 m
    assert daily == pytest.approx((19 * 10.0 + 9 * 11.0) / 28)
    assert legacy - daily == pytest.approx(0.418, abs=1e-3)


def test_uniform_cadence_gives_the_same_month_either_way(tmp_path):
    idx = pd.date_range("2019-03-01", "2019-04-01", freq="h", inclusive="left")
    s = pd.Series(np.linspace(5.0, 6.0, len(idx)), index=idx)
    wells = tmp_path / "wells"
    wells.mkdir()
    pd.DataFrame({"value": s}).to_parquet(wells / "W2.parquet")
    st = pd.DataFrame({"sid": ["W2"], "GroundwaterLayerCode": ["1"],
                       "LocationByTWD97": ["200000 2640000"]})
    kw = dict(t0="2019-03-01", t1="2019-04-01")
    a = build_head_field(str(wells), st, **kw).heads
    b = build_head_field(str(wells), st, day_weighted=True, **kw).heads
    assert np.allclose(a, b)


def test_switch_is_recorded_only_when_set_and_read_back():
    base = dict(ic_merged_proximal=False, ground_elev="wells", strict_coverage=False,
                dem_npz="x.npz", ic_layered_proximal=False, proximal_layered=False)
    assert "day_weighted_months" not in _input_opts_record(SimpleNamespace(**base))
    rec = _input_opts_record(SimpleNamespace(**base, day_weighted_months=True))
    assert rec["day_weighted_months"] is True
    assert input_options(rec)["day_weighted_months"] is True
    assert "day_weighted_months" not in input_options({"ground_elev": "wells"})
