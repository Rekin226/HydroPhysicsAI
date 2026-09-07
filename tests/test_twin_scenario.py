"""Pumping scenarios: policy classification, scoping, and forward extension."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydrophysics.twin.scenario import (
    CLASSES,
    PumpingScenario,
    climatology,
    purpose_class,
)


def test_purpose_classification_covers_the_real_census_labels():
    """The specific agricultural sub-types must not be swallowed by the 農業用水 prefix."""
    assert purpose_class("農業用水(灌溉-旱作)") == "irrigation"
    assert purpose_class("農業用水(灌溉-一期水稻)") == "irrigation"
    assert purpose_class("農業用水(灌溉-溫室設施)") == "irrigation"
    assert purpose_class("農業用水(養殖-淡水)") == "aquaculture"
    assert purpose_class("農業用水(養殖-鹹水)") == "aquaculture"
    assert purpose_class("農業用水(畜牧-豬)") == "livestock"
    assert purpose_class("家庭用水") == "domestic"
    assert purpose_class("家用及公共給水") == "domestic"
    assert purpose_class("公共用水") == "domestic"
    assert purpose_class("工業用水") == "industry"
    assert purpose_class("其他用途") == "other"
    # Bare agricultural water is deliberately NOT guessed into irrigation.
    assert purpose_class("農業用水") == "other"
    assert purpose_class(None) == "other"


def _e(a=4, t=6):
    return {"irrigation": np.ones((a, t)) * 10.0, "domestic": np.ones((a, t)) * 2.0}


def _dates(t=6):
    return pd.date_range("2020-01-01", periods=t, freq="MS")


def test_baseline_is_the_untouched_sum():
    tot = PumpingScenario("base").apply(_e(), _dates())
    assert np.allclose(tot, 12.0)


def test_factor_applies_to_named_class_only():
    s = PumpingScenario("cut", factors={"irrigation": 0.5})
    tot = s.apply(_e(), _dates())
    assert np.allclose(tot, 5.0 + 2.0)


def test_retirement_to_zero():
    s = PumpingScenario("retire", factors={"irrigation": 0.0, "domestic": 0.0})
    assert np.allclose(s.apply(_e(), _dates()), 0.0)


def test_start_date_delays_the_policy():
    dates = _dates(6)                       # 2020-01 .. 2020-06
    s = PumpingScenario("cut", factors={"irrigation": 0.0}, start="2020-04-01")
    tot = s.apply(_e(), dates)
    assert np.allclose(tot[:, :3], 12.0), "before the start date nothing changes"
    assert np.allclose(tot[:, 3:], 2.0), "after it, irrigation is gone"


def test_zone_restriction_limits_which_cells_change():
    from hydrophysics.twin.zones import ZONE_NAMES

    zone_of_cell = np.array([0, 0, 1, 2])
    s = PumpingScenario("cut", factors={"irrigation": 0.0}, zones=(ZONE_NAMES[0],))
    tot = s.apply(_e(a=4), _dates(), zone_of_cell=zone_of_cell)
    assert np.allclose(tot[:2], 2.0), "zone-0 cells lose irrigation"
    assert np.allclose(tot[2:], 12.0), "other zones untouched"


def test_zone_without_mapping_is_an_error():
    s = PumpingScenario("x", factors={"irrigation": 0.5}, zones=("proximal",))
    with pytest.raises(ValueError):
        s.apply(_e(), _dates())


def test_rejects_unknown_class_and_negative_factor():
    with pytest.raises(ValueError):
        PumpingScenario("x", factors={"golf": 0.5})
    with pytest.raises(ValueError):
        PumpingScenario("x", factors={"irrigation": -1.0})


def test_classes_are_exhaustive_for_the_rules():
    assert set(CLASSES) >= {"irrigation", "aquaculture", "livestock", "domestic",
                            "industry", "other"}


def test_climatology_repeats_the_month_of_year_mean():
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    # A pure seasonal signal: value == month number, two identical years.
    arr = np.tile(np.arange(1, 13, dtype="float64"), 2)[None, :]
    ext, fut = climatology(arr, dates, horizon=6)
    assert ext.shape == (1, 6)
    assert fut[0] == pd.Timestamp("2022-01-01")
    # January..June of the extension must reproduce 1..6.
    assert np.allclose(ext[0], np.arange(1, 7))


def test_climatology_rejects_nonpositive_horizon():
    with pytest.raises(ValueError):
        climatology(np.ones((1, 12)), pd.date_range("2020-01-01", periods=12, freq="MS"), 0)


def test_describe_is_readable():
    s = PumpingScenario("cut30", factors={"irrigation": 0.7}, start="2026-01-01")
    assert "irrigation x0.7" in s.describe()
    assert "2026-01-01" in s.describe()
