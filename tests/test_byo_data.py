"""Bring-your-own-data API: GWData.from_arrays, validate(), load_dataset_from_frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydrophysics import GWData, load_dataset_from_frames
from hydrophysics.baselines import climatology_prediction


def _arrays(n_wells=3, start="2018-11-01", end="2019-02-28"):
    dates = pd.date_range(start, end, freq="D")
    T = len(dates)
    rng = np.random.default_rng(0)
    target = rng.normal(10, 1, (n_wells, T))
    rainfall = rng.gamma(2.0, 3.0, (n_wells, T))
    well_ids = [f"w{i+1}" for i in range(n_wells)]
    return well_ids, dates, target, rainfall


def test_from_arrays_builds_usable_dataset():
    well_ids, dates, target, rainfall = _arrays()
    attrs = pd.DataFrame({
        "st_id": well_ids,
        "tm_x": np.linspace(190000, 210000, len(well_ids)),
        "tm_y": np.linspace(2630000, 2660000, len(well_ids)),
        "group": ["coastal", "inland", "inland"],
    })
    d = GWData.from_arrays(well_ids=well_ids, dates=dates, target=target,
                           rainfall=rainfall, attrs=attrs, split_date="2019-01-01")
    assert d.n_wells == 3 and d.n_days == len(dates)
    assert d.target.shape == (3, len(dates))
    # derived fields
    assert d.doy[0] == dates[0].dayofyear
    assert d.train_mask.sum() > 0 and d.val_mask.sum() > 0
    assert d.train_mask.sum() + d.val_mask.sum() == d.n_days
    # upstream defaults to all-NaN (no coupling supplied)
    assert not np.isfinite(d.upstream).any()
    # is_coastal derived from group
    assert d.attrs.loc["w1", "is_coastal"] == 1
    assert d.attrs.loc["w2", "is_coastal"] == 0
    # fully usable by the foundation baselines
    pred = climatology_prediction(d)
    assert pred.shape == d.target.shape


def test_from_arrays_shape_mismatch_raises():
    well_ids, dates, target, rainfall = _arrays()
    with pytest.raises(ValueError, match="shape"):
        GWData.from_arrays(well_ids=well_ids, dates=dates,
                           target=target[:, :-5],  # wrong T
                           rainfall=rainfall)


def test_from_arrays_attrs_as_dict_aligned_to_wells():
    well_ids, dates, target, rainfall = _arrays()
    d = GWData.from_arrays(
        well_ids=well_ids, dates=dates, target=target, rainfall=rainfall,
        attrs={"tm_x": [1.0, 2.0, 3.0], "tm_y": [4.0, 5.0, 6.0]})
    assert list(d.attrs.index) == well_ids
    assert d.attrs.loc["w2", "tm_x"] == 2.0


def test_validate_clean_dataset_has_no_issues():
    well_ids, dates, target, rainfall = _arrays()
    attrs = pd.DataFrame({
        "st_id": well_ids,
        "tm_x": [1.0, 2.0, 3.0], "tm_y": [4.0, 5.0, 6.0],
        "is_coastal": [1, 0, 0], "dist_to_coast_m": [100.0, 200.0, 300.0],
        "dom_amp": [0.1, 0.2, 0.3], "ups_lag_days": [1, 1, 1], "rf_lag_days": [1, 1, 1],
    })
    d = GWData.from_arrays(well_ids=well_ids, dates=dates, target=target,
                           rainfall=rainfall, attrs=attrs, split_date="2019-01-01")
    assert d.validate() == []


def test_validate_flags_missing_features_and_dead_well():
    well_ids, dates, target, rainfall = _arrays()
    target[0] = np.nan  # a well with no finite target
    d = GWData.from_arrays(well_ids=well_ids, dates=dates, target=target,
                           rainfall=rainfall, attrs={"tm_x": [1, 2, 3], "tm_y": [4, 5, 6]})
    issues = d.validate()
    assert any("no finite target" in s for s in issues)
    assert any("static feature" in s for s in issues)  # is_coastal, dist_to_coast_m, ...


def test_validate_flags_empty_split():
    well_ids, dates, target, rainfall = _arrays(start="2018-01-01", end="2018-06-30")
    d = GWData.from_arrays(well_ids=well_ids, dates=dates, target=target,
                           rainfall=rainfall, split_date="2019-01-01")  # all before split
    assert any("no validation days" in s for s in d.validate())


def test_load_dataset_from_frames_long_format_with_column_map():
    dates = pd.date_range("2018-11-01", "2019-02-28", freq="D")
    wells = ["A", "B"]
    rng = np.random.default_rng(1)
    # Long-format frames with the user's own column names -> aliased via column_map.
    gw_long = pd.DataFrame([
        {"DateTime": t, "WellID": w, "Level": 10 + rng.normal()}
        for w in wells for t in dates])
    rf_long = pd.DataFrame([
        {"DateTime": t, "GaugeID": w, "Rain": max(0.0, rng.normal(2, 2))}
        for w in wells for t in dates])
    stations = pd.DataFrame({"WellID": wells, "X": [190000.0, 200000.0],
                             "Y": [2630000.0, 2640000.0], "group": ["coastal", "inland"]})
    cmap = {"DateTime": "date", "WellID": "st_id", "Level": "level",
            "GaugeID": "rf_id", "Rain": "rainfall", "X": "tm_x", "Y": "tm_y"}
    d = load_dataset_from_frames(gw_long, rf_long, stations, column_map=cmap,
                                 split_date="2019-01-01")
    assert d.n_wells == 2 and d.n_days == len(dates)
    assert np.isfinite(d.target).all() and np.isfinite(d.rainfall).all()
    assert d.attrs.loc["A", "is_coastal"] == 1
    assert d.validate() == [] or all("static feature" in s for s in d.validate())


def test_load_dataset_from_frames_pairing_sets_upstream():
    dates = pd.date_range("2018-12-01", "2019-01-31", freq="D")
    wells = ["A", "B"]
    gw_long = pd.DataFrame([{"date": t, "st_id": w, "level": 10.0}
                            for w in wells for t in dates])
    rf_long = pd.DataFrame([{"date": t, "rf_id": w, "rainfall": 1.0}
                            for w in wells for t in dates])
    stations = pd.DataFrame({"st_id": wells, "tm_x": [1.0, 2.0], "tm_y": [3.0, 4.0]})
    pairing = pd.DataFrame({"st_id": wells, "rf_id": wells, "ups_id": ["B", "A"],
                            "lag_days": [1, 1]})
    d = load_dataset_from_frames(gw_long, rf_long, stations, pairing=pairing,
                                 split_date="2019-01-01")
    # upstream now populated (A driven by B, B by A)
    assert np.isfinite(d.upstream).all()
    assert "rf_lag_days" in d.attrs.columns
