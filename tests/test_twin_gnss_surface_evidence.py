"""Scientific gates for cached GNSS surface diagnostics."""
import json

import numpy as np
import pandas as pd
import pytest

from hydrophysics.twin.gnss_surface_evidence import (
    analyze,
    checked_height,
    segment_contrasts,
    station_match,
    step_candidates,
)


def test_persistent_step_is_not_scored_across(tmp_path):
    days = pd.date_range("2023-01-01", "2023-04-30")
    values = np.full(len(days), 40.0)
    values[days >= pd.Timestamp("2023-03-01")] += 0.10
    s = pd.Series(values, index=days)
    steps = step_candidates(s)
    assert len(steps) == 1 and steps[0]["date"] == "2023-03-01"
    dates = pd.date_range("2023-01-01", "2023-04-01", freq="MS")
    # A continuous zero prediction would spuriously score a 10 cm displacement
    # if the undocumented reference step were treated as physical motion.
    assert segment_contrasts(s, steps, dates, np.zeros(len(dates))) == []


def test_identity_and_raw_status_are_required(tmp_path):
    x, y = 187770.14, 2620603.69
    meta = tmp_path / "meta.parquet"
    pd.DataFrame([{"ObservatoryIdentifier": "A130600LSTKJS",
                   "ObservatoryName": "土庫國中", "X_3826": x, "Y_3826": y}]).to_parquet(meta)
    assert station_match(meta)["ring_distance_m"] == pytest.approx(6.368, abs=0.01)
    days = pd.date_range("2023-01-01", "2023-03-31")
    cache = tmp_path / "ls-wra-gnss-obs__TKJS.parquet"
    pd.DataFrame({"value": np.full(len(days), 40.0)}, index=days).to_parquet(cache)
    mask = np.array([[True]])
    forward = tmp_path / "forward.npz"
    np.savez(forward, mask=mask, x0=187000., y0=2620000., dx=1000.,
             dates=np.array(["2023-01-01", "2023-02-01", "2023-03-01"]),
             subs_mean=np.zeros((1, 1, 3)), scenario_names=["baseline"])
    report = analyze(cache, meta, [forward])
    assert report["status"].startswith("NOT_VALIDATED")
    assert report["models"]["forward"]["segments"][0]["rmse_raw_model_cm"] == 0
    assert "NOT_VALIDATION" in report["models"]["forward"]["segments"][0]["status"]
    assert len(checked_height(cache)) == len(days)


def test_official_json_selects_only_tuku_and_rejects_duplicate_days(tmp_path):
    rows = [
        {"observatoryidentifier": "A130600LSTKJS", "observatoryname": "土庫國中",
         "observationttime": "2025-05-01T00:00:00", "height": "40.38"},
        {"observatoryidentifier": "A130600LSHLES", "observatoryname": "宏崙國小",
         "observationttime": "2025-05-01T00:00:00", "height": "42.00"},
    ]
    source = tmp_path / "official.json"
    source.write_text(json.dumps(rows))
    series = checked_height(source)
    assert len(series) == 1 and series.iloc[0] == pytest.approx(40.38)
    source.write_text(json.dumps(rows + rows[:1]))
    with pytest.raises(ValueError, match="invalid"):
        checked_height(source)


def test_missing_month_does_not_bridge_unknown_reference():
    jan_feb = pd.date_range("2023-01-01", "2023-02-28")
    apr_may = pd.date_range("2023-04-01", "2023-05-31")
    days = jan_feb.append(apr_may)
    series = pd.Series(np.full(len(days), 40.0), index=days)
    model_months = pd.date_range("2023-01-01", "2023-05-01", freq="MS")
    assert segment_contrasts(series, [], model_months,
                             np.zeros(len(model_months))) == []
    with pytest.raises(ValueError, match="threshold"):
        step_candidates(series, 0.001)


def test_nonmidnight_or_gross_unit_mismatch_rejected(tmp_path):
    source = tmp_path / "official.json"
    row = {"observatoryidentifier": "A130600LSTKJS",
           "observatoryname": "土庫國中",
           "observationttime": "2025-05-01T12:00:00", "height": "40.38"}
    source.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="invalid"):
        checked_height(source)
    row["observationttime"] = "2025-05-01T00:00:00"
    row["height"] = "4038"
    source.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="implausible"):
        checked_height(source)
