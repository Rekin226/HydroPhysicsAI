from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("matplotlib")

from hydrophysics.twin.calibrate_flow import monthly_mean_rate  # noqa: E402
from hydrophysics.twin.cwa_rain import (  # noqa: E402
    build,
    compare_with_cache,
    requests_for,
    year_window,
)


def _codis_file(tmp_path, sid, year, days, fetched_at="2026-10-11T12:00:00+00:00"):
    """days: {date: (Accumulation, Accumulationf)}"""
    payload = {"code": 200, "message": "", "metadata": {"count": len(days), "timezone": "+08:00"},
               "data": [{"StationID": sid,
                         "dts": [{"DataDate": f"{d}T00:00:00",
                                  "Precipitation": {"Accumulation": v, "Accumulationf": f}}
                                 for d, (v, f) in days.items()]}]}
    if not days:
        payload["data"] = []
    p = tmp_path / f"{sid}_{year}.json"
    p.write_text(json.dumps(payload))
    (tmp_path / f"{sid}_{year}.meta.json").write_text(json.dumps(
        {"station_id": sid, "fetched_at": fetched_at}))


def test_requests_follow_the_inventory_windows():
    chains = {"x": [("OLD", "auto_C0", "1992-01-01", "2015-05-28"),
                    ("NEW", "auto_C0", "2015-05-07", None)]}
    req = requests_for(chains, "2012-01-01", "2016-12-31")
    assert [(s, y) for _, s, _, y in req] == [("OLD", 2012), ("OLD", 2013), ("OLD", 2014),
                                             ("OLD", 2015), ("NEW", 2015), ("NEW", 2016)]
    assert year_window(2016, "2012-01-01", "2016-10-10") == ("2016-01-01", "2016-10-10")


def test_build_merges_a_chain_newest_first_and_keeps_gaps_missing(tmp_path):
    old = {f"2015-01-{d:02d}": (1.0, 0) for d in range(1, 32)}
    old["2015-01-05"] = (-9.8, 0)          # trace -> 0.0
    old["2015-01-06"] = (7.5, 1)           # CWA missing-value mark -> NaN
    old["2015-01-07"] = (-9999.6, 0)       # sentinel -> NaN
    new = {"2015-01-30": (3.0, 0), "2015-01-31": (4.0, 0)}   # overlaps the old station
    _codis_file(tmp_path, "OLD", 2015, old)
    _codis_file(tmp_path, "NEW", 2015, new)
    chains = {"g": [("OLD", "auto_C0", "1992-01-01", "2015-01-31"),
                    ("NEW", "auto_C0", "2015-01-30", None)]}
    daily, src, qa = build(str(tmp_path), "2015-01-01", "2015-02-10", chains, ["g"])
    g = daily["g"]
    assert g["2015-01-01"] == 1.0 and g["2015-01-05"] == 0.0
    assert np.isnan(g["2015-01-06"]) and np.isnan(g["2015-01-07"])
    assert g["2015-01-30"] == 3.0 and src["g"]["2015-01-30"] == "NEW"
    assert src["g"]["2015-01-29"] == "OLD"
    assert g["2015-02-01":].isna().all()               # nothing fetched for February
    assert qa["g"]["OLD"]["trace_less_than_0.5_mm"] == 1
    assert qa["g"]["OLD"]["source_missing_value_mark"] == 1


def test_build_refuses_a_station_mismatch(tmp_path):
    _codis_file(tmp_path, "OTHER", 2015, {"2015-01-01": (1.0, 0)})
    (tmp_path / "OLD_2015.json").write_bytes((tmp_path / "OTHER_2015.json").read_bytes())
    (tmp_path / "OLD_2015.meta.json").write_bytes((tmp_path / "OTHER_2015.meta.json").read_bytes())
    with pytest.raises(ValueError, match="station mismatch"):
        build(str(tmp_path), "2015-01-01", "2015-01-31",
              {"g": [("OLD", "auto_C0", "1992-01-01", None)]}, ["g"])


def test_compare_with_cache_counts_the_zero_fill_signature():
    idx = pd.date_range("2021-01-01", "2021-01-10", freq="D")
    new = pd.DataFrame({"g": [0, 5, 0, 2, np.nan, np.nan, 1, 1, 0, 3]}, index=idx, dtype=float)
    cache = pd.DataFrame({"g": [0, 5, 0, 0, 0, 4, 1, 1.5, 0, 3]}, index=idx, dtype=float)
    c = compare_with_cache(new, cache).iloc[0]
    assert c.both_numeric == 8 and c.equal == 6
    assert c.cache_zero_codis_wet == 1 and c.cache_zero_codis_wet_mm == 2.0
    assert c.codis_missing_cache_value == 2 and c.codis_missing_cache_nonzero == 1
    assert c.other_disagreements == 1 and np.isclose(c.other_mad_mm, 0.5)


def test_monthly_mean_rate_masks_thin_months_and_leaves_full_ones():
    idx = pd.date_range("2021-01-01", "2021-02-28", freq="D")
    df = pd.DataFrame({"a": 2.0, "b": 2.0}, index=idx)
    df.loc["2021-01-10":"2021-01-20", "b"] = np.nan          # 11 of 31 days missing -> 65 %
    df.loc["2021-02-01":"2021-02-03", "b"] = np.nan          # 25 of 28 -> 89 %
    m = monthly_mean_rate(df)
    assert m["a"].tolist() == [2.0, 2.0]
    assert np.isnan(m["b"].iloc[0]) and m["b"].iloc[1] == 2.0
