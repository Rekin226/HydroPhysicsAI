"""calibrate_coupled --withhold-surveys and --column-zones 3 (WP N1 policy-response tests).

A withheld survey must stay out of the column fit AND out of every in-run check that would
otherwise print a score on it; a three-zone column over a county-split flow model must run
in twin.forward on the collapsed zone map and nowhere else.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from hydrophysics.twin.calibrate_coupled import collapse_zones, withhold_rows  # noqa: E402
from hydrophysics.twin.calibrate_flow import mask_month_flags  # noqa: E402
from hydrophysics.twin.explorer3d import validate_against_leveling  # noqa: E402
from hydrophysics.twin.forward import load_or_fit_vep  # noqa: E402
from hydrophysics.twin.grid import FanGrid  # noqa: E402
from hydrophysics.twin.leveling import PANEL  # noqa: E402


def _col_params():
    return [{"log_ske": float(np.log(1e-3)), "log_skv": float(np.log(0.2)),
             "log_tau": float(np.log(t)), "h_pc0": 0.0} for t in (24.0, 3960.0, 821.0)]


def test_withhold_rows_removes_flagged_months_and_short_rows():
    mask = np.array([[1, 1, 1, 1, 1, 0],
                     [1, 0, 1, 1, 0, 0],
                     [0, 1, 1, 1, 1, 1]], dtype=bool)
    flags = np.array([0, 0, 1, 1, 0, 0], dtype=bool)
    out, keep = withhold_rows(mask, flags)
    assert not out[:, 2:4].any()
    assert out[0].tolist() == [True, True, False, False, True, False]
    assert keep.tolist() == [True, False, True]        # row 1 keeps 1 survey < 3


def test_withhold_rows_without_flags_is_the_identity():
    mask = np.array([[1, 0, 1]], dtype=bool)
    out, keep = withhold_rows(mask, None)
    assert np.array_equal(out, mask) and keep.tolist() == [True]


def test_withhold_window_uses_the_mask_months_grammar():
    dates = pd.date_range("2012-01-01", periods=36, freq="MS")
    f = mask_month_flags("2013-05:2013-07", dates)
    assert f.sum() == 3 and f[16] and f[18] and not f[19]
    assert mask_month_flags(None, dates) is None


def test_collapse_zones_maps_the_county_copies_onto_their_zone():
    assert collapse_zones(np.array([0, 1, 2, 3, 4, 5])).tolist() == [0, 1, 2, 0, 1, 2]


def _panel(tmp_path, dates, elev):
    d = tmp_path / os.path.dirname(PANEL)
    d.mkdir(parents=True)
    pd.DataFrame({"sid": "BM1", "datetime": dates, "elev_m": elev, "x_3826": 500.0,
                  "y_3826": 500.0}).to_parquet(tmp_path / PANEL)
    return str(tmp_path)


def test_validate_against_leveling_drops_excluded_surveys(tmp_path):
    grid = FanGrid(nx=1, ny=1, dx=1000.0, x0=0.0, y0=0.0, mask=np.ones((1, 1), dtype=bool))
    model_dates = pd.date_range("2012-01-01", periods=72, freq="MS")
    surveys = pd.to_datetime([f"{y}-05-10" for y in range(2012, 2018)])
    elev = np.array([0.0, -0.01, -0.02, -0.03, -0.20, -0.05])     # 2016 is the outlier
    ddir = _panel(tmp_path, surveys, elev)
    field = np.linspace(0.0, 0.05, 72)[None, :]                   # steady sinking
    full = validate_against_leveling(ddir, grid, field, model_dates, max_rate=None)
    flags = mask_month_flags("2016-01:2016-12", model_dates)
    held = validate_against_leveling(ddir, grid, field, model_dates, max_rate=None,
                                     exclude_months=flags)
    assert full["n_sites"] == held["n_sites"] == 1
    assert held["rmse_cm"] < full["rmse_cm"]                      # the outlier is gone


def test_collapsed_three_zone_column_runs_on_a_county_split_map(tmp_path):
    f = tmp_path / "vep.json"
    f.write_text(json.dumps({"zonal": _col_params(), "column_zones_collapsed": True}))
    zone6 = np.array([0, 1, 2, 3, 4, 5])
    col, _ = load_or_fit_vep(str(f), None, None, "cpu", zone_of_cell=zone6)
    assert col is not None
    g = tmp_path / "plain.json"
    g.write_text(json.dumps({"zonal": _col_params()}))
    with pytest.raises(ValueError, match="county-split"):
        load_or_fit_vep(str(g), None, None, "cpu", zone_of_cell=zone6)
