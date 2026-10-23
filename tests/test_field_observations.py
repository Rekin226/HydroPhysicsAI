"""Field imports must not turn partial totals or later revisions into truth."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from hydrophysics.twin.field_observations import (
    observations_available_at,
    read_codis_rainfall,
    read_gweb_groundwater,
)


def _write(tmp_path, payload):
    path = tmp_path / "response.json"
    path.write_text(json.dumps(payload))
    return path


def _rain(tmp_path, values):
    return _write(tmp_path, {
        "metadata": {"timezone": "+08:00"},
        "data": [{"StationID": "test", "dts": [
            {"DataDate": f"2024-01-{day:02d}T00:00:00", "Precipitation": {
                "Accumulation": value, "Accumulationf": flag,
            }} for day, (value, flag) in enumerate(values, 1)
        ]}],
    })


def test_partial_rainfall_and_trace_are_not_complete_daily_totals(tmp_path):
    path = _rain(tmp_path, [(0, 0), (33, 9), (-9.8, 0), (-9.5, 0), (2, None)])
    out = read_codis_rainfall(path, station_id="test", start="2024-01-01",
                             end="2024-01-05", acquired_at="2024-02-01T00:00:00Z")
    assert out.value.iloc[0] == 0
    assert out.value.iloc[1:].isna().all()
    assert out.value_raw.iloc[1] == 33
    assert out.qa_status.iloc[2] == "trace_less_than_0.5_mm"
    assert not out.validation_approved.any()
    assert not out.independent_holdout.any()


def test_later_revision_is_unavailable_to_a_historical_forecast(tmp_path):
    path = _rain(tmp_path, [(5, 0)])
    out = read_codis_rainfall(path, station_id="test", start="2024-01-01",
                             end="2024-01-01", acquired_at="2025-01-01T00:00:00Z")
    assert observations_available_at(out, "2024-02-01T00:00:00Z").empty
    assert len(observations_available_at(out, "2025-01-01T00:00:00Z")) == 1
    assert out.period_end.iloc[0] == pd.Timestamp("2024-01-02T00:00:00+08:00")
    with pytest.raises(ValueError, match="timezone"):
        observations_available_at(out, "2025-01-01")


@pytest.mark.parametrize("values", [[1], [1, 2, 3]])
def test_undated_groundwater_length_must_match_window(tmp_path, values):
    path = _write(tmp_path, {"StationID": "test", "WaterLevelData": values})
    with pytest.raises(ValueError, match="array length"):
        read_gweb_groundwater(path, station_id="test", start="2024-01-01",
                              end="2024-01-02", acquired_at="2025-01-01T00:00:00Z")


def test_negative_heads_remain_valid_and_sentinels_do_not(tmp_path):
    path = _write(tmp_path, {"StationID": "test", "WaterLevelData": [-8.2, -9998, None]})
    out = read_gweb_groundwater(path, station_id="test", start="2024-01-01",
                               end="2024-01-03", acquired_at="2025-01-01T00:00:00Z")
    assert out.value.iloc[0] == -8.2
    assert np.isnan(out.value.iloc[1:]).all()
    assert out.qa_status.iloc[0] == "not_manually_checked"
    assert not out.validation_approved.any()
    assert (out.screen_status == "unconfirmed").all()


def test_wrong_station_and_impossible_acquisition_time_fail(tmp_path):
    path = _rain(tmp_path, [(5, 0)])
    with pytest.raises(ValueError, match="station mismatch"):
        read_codis_rainfall(path, station_id="other", start="2024-01-01",
                           end="2024-01-01", acquired_at="2025-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="precedes"):
        read_codis_rainfall(path, station_id="test", start="2024-01-01",
                           end="2024-01-01", acquired_at="2024-01-01T00:00:00Z")


def test_duplicate_and_out_of_window_dates_fail(tmp_path):
    path = _rain(tmp_path, [(5, 0), (6, 0)])
    with pytest.raises(ValueError, match="outside"):
        read_codis_rainfall(path, station_id="test", start="2024-01-01",
                           end="2024-01-01", acquired_at="2025-01-01T00:00:00Z")
    payload = json.loads(path.read_text())
    payload["data"][0]["dts"][1]["DataDate"] = "2024-01-01T00:00:00"
    path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="duplicate"):
        read_codis_rainfall(path, station_id="test", start="2024-01-01",
                           end="2024-01-02", acquired_at="2025-01-01T00:00:00Z")
