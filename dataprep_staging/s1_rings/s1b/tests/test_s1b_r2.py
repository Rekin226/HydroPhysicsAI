"""CPU unit tests of the S1b repair-round helpers (synthetic values only, no real data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from s1_rings.s1b import s1b_r2_core as R  # noqa: E402


def test_contains_rules():
    # C_pre 40, C_post 20 (range 20..40), sigmas 5, L 250 m (30 ppm = 7.5 mm, 40 ppm = 10 mm)
    a = (40.0, 20.0, 5.0, 5.0, 250000.0)
    assert R.contains("k:0", 30.0, *a)
    assert not R.contains("k:0", 15.0, *a)
    assert R.contains("k:1", 15.0, *a)
    assert R.contains("range_ppm:30", 13.0, *a)
    assert not R.contains("range_ppm:30", 12.0, *a)
    assert R.contains("mid_ppm:40", 39.9, *a)
    assert not R.contains("mid_ppm:30", 38.0, *a)
    assert R.contains("k:2", 10.0, *a) and not R.contains("k:2", 9.9, *a)
    assert R.contains("k:0", float("nan"), *a) is None


def test_min_k_and_k_star():
    assert R.min_k(30.0, 40.0, 20.0, 5.0, 5.0) == 0.0
    assert abs(R.min_k(10.0, 40.0, 20.0, 5.0, 4.0) - 2.5) < 1e-12
    assert abs(R.min_k(50.0, 40.0, 20.0, 2.0, 5.0) - 5.0) < 1e-12
    assert R.k_star([0, 0, 0.3, 1.2, 2.0], 0.8) == 1.2       # need 4 of 5
    assert R.k_star([0.1] * 9 + [5.0], 0.8) == 0.1


def test_family_counts():
    rows = [{"evaluable": True, "site": f"s{i}", "C_primary_mm": c, "C_pre_mm": 40.0,
             "C_post_mm": 20.0, "sigma_pre_mm": 5.0, "sigma_post_mm": 5.0, "L_m": 250.0}
            for i, c in enumerate([25.0, 30.0, 35.0, 12.0, 50.0])]
    f = R.family_counts(rows, "k:0")
    assert f["n_contain"] == 3 and f["n_needed"] == 4 and not f["containment_pass"]
    assert f["sites_outside"] == ["s3", "s4"]
    f = R.family_counts(rows, "k:2")
    assert f["n_contain"] == 5 and f["containment_pass"]


def test_williams_t_signs():
    w = R.williams_t(0.7, 0.3, 0.5, 14)
    assert w["t"] > 0 and 0 < w["p_one_sided"] < 0.5
    w0 = R.williams_t(0.5, 0.5, 0.5, 14)
    assert abs(w0["t"]) < 1e-12


def test_bootstrap_diff_positive_when_one_predictor_is_better():
    rng = np.random.default_rng(3)
    x = rng.normal(size=40)
    yb = x + 0.3 * rng.normal(size=40)
    yr = 0.2 * x + rng.normal(size=40)
    b = R.corr_diff_bootstrap(x, yb, yr, n_boot=2000, seed=1)
    assert b["ci95"][0] > 0 and b["p_diff_le_0"] < 0.01


def test_bridge_increments_and_longest_gap():
    dates = pd.date_range("2022-05-01", periods=10, freq="12D")
    seg = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    d = np.arange(10, dtype=float) * 2.0
    d[2] = np.nan                                  # last finite of segment 0 is index 1
    inc = R.bridge_increments(dates, d, seg, 0, 2)
    assert len(inc) == 2
    assert inc[0]["inc_mm"] == d[3] - d[1] and inc[0]["days"] == 24
    assert inc[1]["inc_mm"] == d[6] - d[5]
    days = pd.DatetimeIndex(["2022-01-02", "2022-01-03", "2022-01-20", "2022-01-21"])
    g = R.longest_gap(days, "2022-01-01", "2022-01-31")
    assert g["max_gap_days"] == 16.0 and g["after"] == "2022-01-03"
